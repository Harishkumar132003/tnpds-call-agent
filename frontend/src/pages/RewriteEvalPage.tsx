import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { StatTile } from '../components/StatTile';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import { SEARCH_TYPE_LABEL } from '../types';
import type { AgentConfig, RewriteQuestion, RewriteResult, RewriteRun, SearchType } from '../types';

const pct = (n: number, of: number) => (of ? `${Math.round((100 * n) / of)}%` : '—');

const isExpected = (expected: string, source: string | null) =>
  Boolean(source && (source === expected || source.startsWith(`${expected} `)));

const fmtScore = (score: number, mode: SearchType) => (
  mode === 'semantic' ? `${(score * 100).toFixed(0)}%` : mode === 'keyword' ? score.toFixed(1) : score.toFixed(3)
);

const promptLabel = (run: RewriteRun) => (
  run.prompt_source === 'default' ? 'built-in prompt' : run.prompt_source === 'config' ? 'configuration prompt' : 'draft prompt'
);

function RankPill({ r }: { r: RewriteResult | undefined }) {
  if (!r) return <span className="muted">—</span>;
  if (r.error) return <span className="pill pill--down" title={r.error}>error</span>;
  if (r.expected_rank === null) return <span className="pill pill--down">missed</span>;
  return <span className={`pill ${r.expected_rank === 1 ? 'pill--up' : 'pill--warn'}`}>#{r.expected_rank}{r.expected_sent ? '' : ' · not sent'}</span>;
}

function Hits({ r, mode }: { r: RewriteResult; mode: SearchType }) {
  const hits = r.hits ?? [];
  if (hits.length === 0) return <span className="muted">{r.error ? '' : 'no sheet retrieved'}</span>;
  return (
    <div className="kb">
      {hits.map((h, i) => {
        const expected = isExpected(r.expected_source, h.source);
        return (
          <div className="kb__row" key={`${h.document_id}-${i}`} style={expected ? { fontWeight: 600 } : undefined}>
            <span className="mono">#{i + 1}</span>{' '}
            <span>{(h.source ?? '?').split(' ')[0]}</span>{' '}
            <span className="pill pill--idle">{fmtScore(h.score ?? 0, mode)}</span>
            {h.used ? <span className="pill pill--info">sent</span> : null}
            {expected && <span className="pill pill--violet">expected</span>}
          </div>
        );
      })}
    </div>
  );
}

/** A fixed set of caller questions rewritten with a draft prompt, so the rewrite prompt can
 *  be tuned against the same questions every time. */
export function RewriteEvalPage() {
  const toast = useToast();
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [questions, setQuestions] = useState<RewriteQuestion[] | null>(null);
  const [runs, setRuns] = useState<RewriteRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [compare, setCompare] = useState<string>('');
  const [results, setResults] = useState<Record<string, RewriteResult[]>>({});
  const [prompt, setPrompt] = useState('');
  const [searchType, setSearchType] = useState<SearchType>('semantic');
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<RewriteQuestion | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [draft, setDraft] = useState({ question_ta: '', question_en: '', expected_source: '' });
  const [sources, setSources] = useState<string[]>([]);
  const [callTimeoutMs, setCallTimeoutMs] = useState(3000);
  const [showPrompt, setShowPrompt] = useState(false);

  const fail = (title: string) => (e: unknown) => toast.show({ tone: 'error', title, body: (e as Error).message ?? String(e) });

  const loadQuestions = () => api.rewriteQuestions().then((d) => {
    setQuestions(d.questions);
    if (d.seeded) toast.show({ tone: 'success', title: 'Question set created', body: `${d.seeded} random questions were picked from the evaluation bank. Edit them until they read like real callers.` });
  });
  const loadRuns = () => api.rewriteRuns().then((d) => {
    setRuns(d.runs);
    setCallTimeoutMs(d.call_timeout_ms);
    setSelected((cur) => cur ?? d.runs[0]?.id ?? null);
  });

  useEffect(() => {
    api.config().then((c) => { setConfig(c); setPrompt(c.rewrite_prompt ?? c.rewrite_prompt_default); }).catch(fail('Could not load the configuration'));
    loadQuestions().catch(fail('Could not load the questions'));
    loadRuns().catch(fail('Could not load the runs'));
    api.evalSources().then((d) => setSources(d.sources)).catch(() => undefined);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const ensureResults = (id: string | null) => {
    if (!id || results[id]) return;
    api.rewriteRun(id).then((d) => setResults((r) => ({ ...r, [id]: d.results }))).catch(fail('Could not load the run'));
  };
  useEffect(() => { ensureResults(selected); }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { ensureResults(compare || null); }, [compare]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async () => {
    setBusy(true);
    try {
      const d = await api.rewriteRunStart(prompt.trim() || null, searchType);
      setResults((r) => ({ ...r, [d.run.id]: d.results }));
      await loadRuns();
      setCompare(selected && selected !== d.run.id ? selected : '');
      setSelected(d.run.id);
      toast.show({ tone: 'success', title: 'Run finished', body: `${d.run.retrieved} of ${d.run.results} expected sheets retrieved, ${d.run.rank1} ranked first.` });
    } catch (e) {
      fail('Run failed')(e);
    } finally {
      setBusy(false);
    }
  };

  const add = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api.rewriteAddQuestion({ question_ta: draft.question_ta, question_en: draft.question_en || null, expected_source: draft.expected_source });
      setDraft({ question_ta: '', question_en: '', expected_source: '' });
      setShowAdd(false);
      await loadQuestions();
    } catch (err) { fail('Could not add')(err); }
  };

  const saveEdit = async (e: FormEvent) => {
    e.preventDefault();
    if (!editing) return;
    try {
      await api.rewriteUpdateQuestion(editing.id, { question_ta: editing.question_ta, question_en: editing.question_en, expected_source: editing.expected_source });
      setEditing(null);
      await loadQuestions();
    } catch (err) { fail('Could not save')(err); }
  };

  const remove = async (q: RewriteQuestion) => {
    if (!window.confirm('Remove this question from the set?')) return;
    try { await api.rewriteDeleteQuestion(q.id); await loadQuestions(); } catch (err) { fail('Could not remove')(err); }
  };

  const judge = async (r: RewriteResult, ok: boolean) => {
    setResults((all) => ({ ...all, [r.run_id]: (all[r.run_id] ?? []).map((x) => (x.id === r.id ? { ...x, judged_ok: ok } : x)) }));
    try {
      const d = await api.rewriteJudge(r.run_id, r.id, ok);
      setRuns((rs) => rs.map((x) => (x.id === d.run.id ? d.run : x)));
    } catch (err) { fail('Could not save the tick')(err); }
  };

  const addRandom = async () => {
    try {
      const d = await api.rewriteAddRandom(5);
      toast.show({ tone: 'success', title: `${d.added} question${d.added === 1 ? '' : 's'} added`, body: 'Picked at random from the enabled evaluation questions.' });
      await loadQuestions();
    } catch (err) { fail('Could not add')(err); }
  };

  const selectedRun = runs.find((r) => r.id === selected) ?? null;
  const compareRun = runs.find((r) => r.id === compare) ?? null;
  const byQuestion = (id: string | null) => {
    const map = new Map<number | null, RewriteResult>();
    for (const r of (id && results[id]) || []) map.set(r.question_id, r);
    return map;
  };
  const cur = useMemo(() => byQuestion(selected), [selected, results]); // eslint-disable-line react-hooks/exhaustive-deps
  const prev = useMemo(() => byQuestion(compare || null), [compare, results]); // eslint-disable-line react-hooks/exhaustive-deps
  const mode: SearchType = selectedRun?.search_type ?? 'semantic';

  const trimmed = prompt.trim();
  const promptState = !config ? '' : trimmed === config.rewrite_prompt_default.trim() ? 'built-in default' : trimmed === (config.rewrite_prompt ?? '').trim() ? 'same as the configuration' : 'draft, not saved';

  return (
    <>
      <Topbar
        title="Rewrite prompt"
        actions={(
          <>
            {questions && <span className="pill pill--idle">{questions.length} fixed questions</span>}
            <button type="button" className="btn btn--ghost btn--sm" onClick={addRandom}>Pick 5 random</button>
            <button type="button" className="btn btn--ghost btn--sm" onClick={() => { setEditing(null); setShowAdd(true); }}>Add question</button>
          </>
        )}
      />
      <div className="page stack">
        <section className="card">
          <div className="card__head">
            <span className="card__label">Rewrite prompt to try</span>
            {promptState && <span className={`pill ${promptState.startsWith('draft') ? 'pill--violet' : 'pill--idle'}`}>{promptState}</span>}
            <span className="topbar__spacer" />
            {config && <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(config.rewrite_prompt ?? config.rewrite_prompt_default)}>Use the configuration's</button>}
            {config && <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(config.rewrite_prompt_default)}>Use the built-in</button>}
          </div>
          <div className="card__body stack">
            <textarea className="field__textarea" rows={14} value={prompt} onChange={(e) => setPrompt(e.target.value)} spellCheck={false} />
            <div className="row" style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
              <div className="field" style={{ minWidth: 240 }}>
                <label className="field__label" htmlFor="rw-search">Search after the rewrite</label>
                <select id="rw-search" className="field__select" value={searchType} onChange={(e) => setSearchType(e.target.value as SearchType)}>
                  <option value="semantic">Semantic · embeddings, as calls do</option>
                  <option value="keyword">Keyword · BM25 word match</option>
                  <option value="hybrid">Hybrid · both, merged by rank (RRF)</option>
                </select>
              </div>
              <span className="topbar__spacer" />
              <button type="button" className="btn btn--primary" disabled={busy || !questions?.length} onClick={run}>{busy ? 'Running…' : `Run ${questions?.length ?? 0} questions`}</button>
            </div>
            <span className="field__hint">Each Tamil question is rewritten with this prompt by the active model, then searched. Nothing is saved to the configuration: when a prompt works, paste it on the Configuration page. The same questions run every time until you edit them.</span>
          </div>
        </section>

        {selectedRun && (
          <div className="stats">
            <StatTile label="Expected sheet retrieved" icon="search" value={pct(selectedRun.retrieved, selectedRun.results)} note={`${selectedRun.retrieved} of ${selectedRun.results}`} />
            <StatTile label="Ranked first" icon="tick" value={pct(selectedRun.rank1, selectedRun.results)} note={`${selectedRun.rank1} of ${selectedRun.results}`} />
            <StatTile label="Rewrite ticked correct" icon="check" value={pct(selectedRun.judged_ok, selectedRun.results)} note={`${selectedRun.judged_ok} of ${selectedRun.results} ticked by you`} pill={selectedRun.judged_ok === 0 ? undefined : { tone: selectedRun.judged_ok / selectedRun.results >= 0.8 ? 'up' : 'warn', text: 'by hand' }} />
            <StatTile label="Sent to the answer step" icon="check" value={pct(selectedRun.sent, selectedRun.results)} note={`${selectedRun.sent} of ${selectedRun.results}`} />
            <StatTile label="Rewrite time" icon="refresh" value={selectedRun.avg_rewrite_ms ? `${(selectedRun.avg_rewrite_ms / 1000).toFixed(1)} s` : '—'} note={selectedRun.errors ? `${selectedRun.errors} errors` : 'average per question'} />
          </div>
        )}

        <section className="card">
          <div className="card__head">
            <span className="card__label">Questions</span>
            {selectedRun && <span className="card__note">showing run {when(selectedRun.started_at)} · {promptLabel(selectedRun)} · {SEARCH_TYPE_LABEL[mode]}</span>}
            {selectedRun && <button type="button" className="btn btn--ghost btn--sm" onClick={() => setShowPrompt((v) => !v)}>{showPrompt ? 'Hide prompt' : 'Show prompt'}</button>}
            <span className="topbar__spacer" />
            {runs.length > 1 && (
              <label className="field__hint" htmlFor="rw-compare">compare with{' '}
                <select id="rw-compare" className="field__select" value={compare} onChange={(e) => setCompare(e.target.value)}>
                  <option value="">—</option>
                  {runs.filter((r) => r.id !== selected).map((r) => <option key={r.id} value={r.id}>{when(r.started_at)} · {promptLabel(r)}</option>)}
                </select>
              </label>
            )}
          </div>
          {showPrompt && selectedRun && (
            <div className="card__body stack" style={{ borderBottom: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <span className="card__label">Prompt used by this run</span>
                <span className={`pill ${selectedRun.prompt_source === 'draft' ? 'pill--violet' : 'pill--idle'}`}>{promptLabel(selectedRun)}</span>
                {selectedRun.model && <span className="pill pill--idle mono">{selectedRun.model}</span>}
                {selectedRun.prompt.trim() === prompt.trim()
                  ? <span className="pill pill--up">same as the editor</span>
                  : <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(selectedRun.prompt)}>Load into the editor</button>}
                {compareRun && (compareRun.prompt.trim() === selectedRun.prompt.trim()
                  ? <span className="pill pill--idle">compared run used the same prompt</span>
                  : <span className="pill pill--warn">compared run used a different prompt</span>)}
              </div>
              <pre className="preview-prompt" style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{selectedRun.prompt}</pre>
            </div>
          )}
          <div className="table-wrap">
            <table className="table table--left">
              <thead><tr><th>#</th><th>Question (Tamil)</th><th>Expected</th><th>Rewritten query</th><th title="Tick when the rewritten query says what the caller asked">Correct?</th><th>Retrieved sheets</th><th>Rank</th><th></th></tr></thead>
              <tbody>
                {(questions ?? []).map((q, i) => {
                  const r = cur.get(q.id);
                  const p = prev.get(q.id);
                  return (
                    <tr key={q.id}>
                      <td className="mono">{i + 1}</td>
                      <td style={{ maxWidth: 280 }}>
                        <span className="cell-link indic">{q.question_ta}</span>
                        {q.question_en && <span className="cell-sub">{q.question_en}</span>}
                      </td>
                      <td className="mono">{q.expected_source}</td>
                      <td style={{ maxWidth: 260 }}>
                        <span className={r?.rewritten_query ? '' : 'muted'}>{r?.rewritten_query ?? (r?.error ? '' : '—')}</span>
                        {r?.error && <span className="cell-sub" style={{ color: 'var(--down)' }}>{r.error}</span>}
                        {r && !r.error && (r.rewrite_ms ?? 0) > callTimeoutMs && <span className="cell-sub" style={{ color: 'var(--warn, #b26b00)' }}>took {((r.rewrite_ms ?? 0) / 1000).toFixed(1)} s, over the {(callTimeoutMs / 1000).toFixed(0)} s limit a call allows</span>}
                        {compareRun && p && <span className="cell-sub" title="the run compared with">↳ {p.rewritten_query ?? (p.error ? 'error' : '—')}</span>}
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        {r && !r.error
                          ? <input type="checkbox" className="judge" checked={r.judged_ok} onChange={(e) => judge(r, e.target.checked)} title="The rewritten query is correct for what the caller said" />
                          : <span className="muted">—</span>}
                        {compareRun && p && !p.error && <div className="cell-sub">was {p.judged_ok ? 'ticked' : 'unticked'}</div>}
                      </td>
                      <td style={{ minWidth: 220 }}>{r ? <Hits r={r} mode={mode} /> : <span className="muted">—</span>}</td>
                      <td>
                        <RankPill r={r} />
                        {compareRun && <div className="cell-sub">was <RankPill r={p} /></div>}
                      </td>
                      <td style={{ whiteSpace: 'nowrap' }}>
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => { setShowAdd(false); setEditing({ ...q }); }}>Edit</button>{' '}
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => remove(q)}>Remove</button>
                      </td>
                    </tr>
                  );
                })}
                {questions && questions.length === 0 && <tr><td colSpan={7} className="muted">No questions: add one, or pick random ones from the evaluation bank.</td></tr>}
                {questions === null && <tr><td colSpan={7}><span className="spinner" /></td></tr>}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <div className="card__head">
            <span className="card__label">Runs</span>
            <span className="topbar__spacer" />
            <span className="card__note">click a run to show its results above · "Load prompt" puts that run's prompt in the editor</span>
          </div>
          {runs.length === 0 ? (
            <div className="card__body muted">No runs yet.</div>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Started</th><th>Prompt</th><th>Search</th><th>Questions</th><th>Retrieved</th><th>Ranked #1</th><th>Ticked correct</th><th>Sent</th><th>Errors</th><th>Rewrite</th><th>Config</th><th></th></tr></thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id} className={`row-link${r.id === selected ? ' is-selected' : ''}`} onClick={() => setSelected(r.id)}>
                      <td><span className="cell-link">{when(r.started_at)}</span>{r.status !== 'done' && <span className="cell-sub">{r.status}{r.error ? `: ${r.error}` : ''}</span>}</td>
                      <td><span className={`pill ${r.prompt_source === 'draft' ? 'pill--violet' : 'pill--idle'}`} title={r.prompt}>{promptLabel(r)}</span></td>
                      <td>{SEARCH_TYPE_LABEL[r.search_type]}</td>
                      <td className="mono">{r.results}/{r.total}</td>
                      <td>{pct(r.retrieved, r.results)}</td>
                      <td>{pct(r.rank1, r.results)}</td>
                      <td>{r.judged_ok ? pct(r.judged_ok, r.results) : '—'}</td>
                      <td>{pct(r.sent, r.results)}</td>
                      <td>{r.errors || '—'}</td>
                      <td>{r.avg_rewrite_ms ? `${(r.avg_rewrite_ms / 1000).toFixed(1)} s` : '—'}</td>
                      <td className="mono">{r.config_id ? `#${r.config_id}` : '—'}</td>
                      <td onClick={(e) => e.stopPropagation()} style={{ whiteSpace: 'nowrap' }}>
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(r.prompt)}>Load prompt</button>{' '}
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => { if (window.confirm('Delete this run?')) api.rewriteDeleteRun(r.id).then(() => { if (selected === r.id) setSelected(null); if (compare === r.id) setCompare(''); return loadRuns(); }).catch(fail('Could not delete')); }}>Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {(editing || showAdd) && (
          <div className="scrim" onClick={() => { setEditing(null); setShowAdd(false); }}>
            <form className="modal" onSubmit={editing ? saveEdit : add} onClick={(e) => e.stopPropagation()}>
              <div className="modal__head">
                <span className="card__label">{editing ? `Edit question ${editing.id}` : 'Add a question'}</span>
                <span className="topbar__spacer" />
                <button type="button" className="btn btn--ghost btn--sm" onClick={() => { setEditing(null); setShowAdd(false); }}>Close</button>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="rw-ta">Tamil, as a caller would say it</label>
                <textarea id="rw-ta" className="field__textarea indic" rows={3} autoFocus value={editing ? editing.question_ta : draft.question_ta} onChange={(e) => (editing ? setEditing({ ...editing, question_ta: e.target.value }) : setDraft({ ...draft, question_ta: e.target.value }))} />
              </div>
              <div className="field">
                <label className="field__label" htmlFor="rw-en">English <span className="muted">(optional, for your reference)</span></label>
                <input id="rw-en" className="field__input" value={editing ? editing.question_en ?? '' : draft.question_en} onChange={(e) => (editing ? setEditing({ ...editing, question_en: e.target.value }) : setDraft({ ...draft, question_en: e.target.value }))} />
              </div>
              <div className="field">
                <label className="field__label" htmlFor="rw-src">Expected sheet</label>
                <input id="rw-src" className="field__input mono" list="rw-sources" value={editing ? editing.expected_source : draft.expected_source} onChange={(e) => (editing ? setEditing({ ...editing, expected_source: e.target.value }) : setDraft({ ...draft, expected_source: e.target.value }))} placeholder="e.g. pb87-2" required />
                <datalist id="rw-sources">{sources.map((s) => <option key={s} value={s} />)}</datalist>
              </div>
              <div className="modal__foot">
                <button type="button" className="btn btn--ghost" onClick={() => { setEditing(null); setShowAdd(false); }}>Cancel</button>
                <button type="submit" className="btn btn--primary">{editing ? 'Save' : 'Add question'}</button>
              </div>
            </form>
          </div>
        )}
      </div>
    </>
  );
}
