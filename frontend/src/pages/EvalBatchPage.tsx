import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { StatTile } from '../components/StatTile';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import { SEARCH_TYPE_LABEL } from '../types';
import type { EvalBatch, EvalResult, SearchType } from '../types';

type Filter = 'all' | 'notsent' | 'missed' | 'errors';

const isExpected = (r: EvalResult, source: string | null) =>
  Boolean(r.expected_source && source && (source === r.expected_source || source.startsWith(`${r.expected_source} `)));

/** A hit's score in the batch's own scale: cosine as a percentage, BM25 to one decimal,
 *  RRF to three. */
const fmtScore = (score: number, mode: SearchType) => (
  mode === 'semantic' ? `${(score * 100).toFixed(0)}%` : mode === 'keyword' ? score.toFixed(1) : score.toFixed(3)
);

/** What happened to the expected chunk, in words, when it was not sent to the LLM. */
function NotSent({ r, mode }: { r: EvalResult; mode: SearchType }) {
  const hits = r.hits ?? [];
  const i = hits.findIndex((h) => isExpected(r, h.source));
  if (i < 0) {
    return <span className="cell-sub">expected <span className="mono">{r.expected_source}</span> not among the {hits.length || 0} retrieved</span>;
  }
  const h = hits[i];
  const top = hits[0]?.score ?? 0;
  if (mode !== 'semantic') {
    return <span className="cell-sub">expected <span className="mono">{r.expected_source}</span> was #{i + 1} but dropped by the character cap</span>;
  }
  return (
    <span className="cell-sub">
      expected <span className="mono">{r.expected_source}</span> was #{i + 1} at {fmtScore(h.score ?? 0, mode)}, below the send threshold
      {top ? ` (top was ${fmtScore(top, mode)})` : ''}
    </span>
  );
}
const pct = (n: number, of: number) => (of ? `${Math.round((100 * n) / of)}%` : '—');

function Hits({ r, mode }: { r: EvalResult; mode: SearchType }) {
  const hits = r.hits ?? [];
  if (hits.length === 0) return <span className="muted">no chunks retrieved</span>;
  return (
    <div className="kb">
      {hits.map((h, i) => {
        const expected = isExpected(r, h.source);
        const score = h.score ?? 0;
        return (
          <div className="kb__row" key={`${h.document_id}-${h.chunk_index}-${i}`} style={expected ? { fontWeight: 600 } : undefined}>
            <span className="mono">#{i + 1}</span>{' '}
            <span>{h.source ?? '?'}</span>{' '}
            <span className={`pill ${mode !== 'semantic' ? 'pill--idle' : score >= 0.5 ? 'pill--up' : score >= 0.35 ? 'pill--warn' : 'pill--idle'}`}>{fmtScore(score, mode)}</span>{' '}
            {h.used ? <span className="pill pill--info">sent</span> : <span className="pill pill--idle">not sent</span>}
            {expected && <span className="pill pill--violet">expected</span>}
          </div>
        );
      })}
    </div>
  );
}

export function EvalBatchPage() {
  const { batchId = '' } = useParams();
  const toast = useToast();
  const [batch, setBatch] = useState<EvalBatch | null>(null);
  const [results, setResults] = useState<EvalResult[]>([]);
  const [filter, setFilter] = useState<Filter>('all');
  const [open, setOpen] = useState<number | null>(null);

  const load = () => api.evalBatch(batchId).then((d) => { setBatch(d.batch); setResults(d.results); });
  useEffect(() => {
    load().catch((e) => toast.show({ tone: 'error', title: 'Could not load the batch', body: String(e) }));
  }, [batchId]); // eslint-disable-line react-hooks/exhaustive-deps

  const running = batch?.status === 'running';
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => { load().catch(() => undefined); }, 5_000);
    return () => window.clearInterval(id);
  }, [running, batchId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!batch) return <div className="page"><span className="spinner" /></div>;

  const shown = results.filter((r) => (
    filter === 'all' ? true
      : filter === 'notsent' ? (!r.expected_sent && r.expected_rank !== null)
        : filter === 'missed' ? r.expected_rank === null
          : Boolean(r.error)
  ));
  const n = batch.results;
  const mode: SearchType = batch.search_type ?? 'semantic';

  return (
    <>
      <Topbar
        title={batch.note ? batch.note : `Evaluation ${when(batch.started_at)}`}
        back={{ to: '/eval-results', label: 'Eval results' }}
        actions={(
          <>
            {batch.status === 'done' ? <span className="pill pill--up">done</span> : running ? <span className="pill pill--info">running {batch.done}/{batch.total}</span> : <span className="pill pill--idle">{batch.status}</span>}
            <span className="pill pill--idle">config #{batch.config_id ?? '?'}</span>
            <span className="pill pill--idle">{batch.concurrency} at a time</span>
            {batch.force_search && <span className="pill pill--violet" title="The first decide step was skipped: every question was searched as said">search forced</span>}
            <span className="pill pill--info" title={mode === 'semantic' ? 'Nearest embeddings, scored by cosine similarity' : mode === 'keyword' ? 'BM25 word match; every retrieved sheet is sent' : 'Semantic and BM25 lists merged by Reciprocal Rank Fusion; every retrieved sheet is sent'}>{SEARCH_TYPE_LABEL[mode]}</span>
            <span className="pill pill--idle mono">{batch.collection ?? ''}</span>
            {running && <button type="button" className="btn btn--ghost btn--sm" onClick={() => api.evalCancel(batch.id).then(load)}>Cancel</button>}
          </>
        )}
      />
      <div className="page stack">
        {batch.error && <div className="banner banner--error">{batch.error}</div>}
        <div className="stats">
          <StatTile label="Questions" icon="chat" value={`${n}/${batch.total}`} note={running ? 'in progress' : batch.finished_at ? `finished ${when(batch.finished_at)}` : ''} />
          <StatTile label="Expected chunk retrieved" icon="search" value={pct(batch.retrieved, n)} note={`${batch.retrieved} of ${n} in the search results`} />
          <StatTile label="Expected chunk sent" icon="check" value={pct(batch.sent, n)} pill={{ tone: n && batch.sent / n >= 0.8 ? 'up' : 'warn', text: 'given to the answer step' }} />
          <StatTile label="Ranked first" icon="tick" value={pct(batch.rank1, n)} note={`${batch.rank1} of ${n}`} />
          <StatTile label="Avg time" icon="refresh" value={batch.avg_ms ? `${(batch.avg_ms / 1000).toFixed(1)} s` : '—'} note={batch.errors ? `${batch.errors} errors` : 'per question'} />
        </div>

        <div className="segmented segmented--inline">
          {(['all', 'notsent', 'missed', 'errors'] as Filter[]).map((f) => (
            <button key={f} type="button" className={`segmented__opt${filter === f ? ' is-on' : ''}`} onClick={() => setFilter(f)}>
              {f === 'all' ? `All (${results.length})` : f === 'notsent' ? `Retrieved but not sent (${results.filter((r) => !r.expected_sent && r.expected_rank !== null).length})` : f === 'missed' ? `Not retrieved (${results.filter((r) => r.expected_rank === null).length})` : `Errors (${results.filter((r) => r.error).length})`}
            </button>
          ))}
        </div>

        <section className="card">
          <div className="table-wrap">
            <table className="table table--left">
              <thead><tr><th>#</th><th>Question (Tamil)</th><th>Expected</th><th>Rewritten query</th><th>Retrieved chunks</th><th>Rank</th><th>Sent</th><th>Answer</th><th>Time</th></tr></thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.id} className={r.error ? 'is-flagged' : undefined}>
                    <td className="mono">{r.question_id ?? '—'}</td>
                    <td style={{ maxWidth: 260 }}>
                      <span className="cell-link indic">{r.question_ta}</span>
                      {r.question_en && <span className="cell-sub">{r.question_en}</span>}
                    </td>
                    <td className="mono">{r.expected_source}</td>
                    <td style={{ maxWidth: 220 }}><span className="cell-sub">{r.rewritten_query ?? '—'}</span></td>
                    <td style={{ minWidth: 260 }}><Hits r={r} mode={mode} /></td>
                    <td>{r.expected_rank === null ? <span className="pill pill--down">missed</span> : <span className={`pill ${r.expected_rank === 1 ? 'pill--up' : 'pill--warn'}`}>#{r.expected_rank}</span>}</td>
                    <td style={{ maxWidth: 240 }}>
                      {r.expected_sent ? <span className="pill pill--up">sent</span> : <><span className="pill pill--down">not sent</span><NotSent r={r} mode={mode} /></>}
                    </td>
                    <td style={{ maxWidth: 360 }}>
                      {r.error && <span className="pill pill--down" title={r.error}>error</span>}
                      {r.answer && (
                        open === r.id
                          ? <span className="cell-sub indic" style={{ whiteSpace: 'pre-wrap', cursor: 'pointer' }} onClick={() => setOpen(null)}>{r.answer}</span>
                          : <span className="cell-sub indic" style={{ cursor: 'pointer' }} title="Show the whole answer" onClick={() => setOpen(r.id)}>{r.answer.length > 140 ? `${r.answer.slice(0, 140)}…` : r.answer}</span>
                      )}
                    </td>
                    <td className="mono">{r.ms ? `${(r.ms / 1000).toFixed(1)} s` : '—'}</td>
                  </tr>
                ))}
                {shown.length === 0 && <tr><td colSpan={9} className="muted">{running ? 'Waiting for the first result…' : 'Nothing to show.'}</td></tr>}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </>
  );
}
