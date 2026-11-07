import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Empty } from '../components/Empty';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import { SEARCH_TYPE_LABEL } from '../types';
import type { EvalBatch } from '../types';

const pct = (n: number, of: number) => (of ? `${Math.round((100 * n) / of)}%` : '—');

export function EvalResultsPage() {
  const toast = useToast();
  const navigate = useNavigate();
  const [batches, setBatches] = useState<EvalBatch[] | null>(null);
  const [running, setRunning] = useState<string | null>(null);

  const load = () => api.evalBatches().then((b) => { setBatches(b.batches); setRunning(b.running); });
  useEffect(() => {
    load().catch((e) => toast.show({ tone: 'error', title: 'Could not load results', body: String(e) }));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const live = running !== null;
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => { load().catch(() => undefined); }, 5_000);
    return () => window.clearInterval(id);
  }, [live]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <Topbar title="Evaluation results" actions={running ? <span className="pill pill--info">a batch is running</span> : undefined} />
      <div className="page stack">
        <section className="card">
          <div className="card__head">
            <span className="card__label">Batches</span>
            <span className="topbar__spacer" />
            <span className="card__note">one row per run · click a row for its per-question results</span>
          </div>
          {batches === null ? (
            <div className="card__body"><span className="spinner" /></div>
          ) : batches.length === 0 ? (
            <Empty icon="table" title="No runs yet" body="Start a run from the Evaluation page." />
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Started</th><th>Status</th><th>Progress</th><th>At a time</th><th>Search</th><th>Retrieved</th><th>Sent to LLM</th><th>Ranked #1</th><th>Errors</th><th>Avg time</th><th>Config</th><th>Collection</th><th></th></tr></thead>
                <tbody>
                  {batches.map((b) => (
                    <tr key={b.id} className="row-link" onClick={() => navigate(`/eval-results/${b.id}`)}>
                      <td><span className="cell-link">{when(b.started_at)}</span>{b.note && <span className="cell-sub">{b.note}</span>}</td>
                      <td>{b.status === 'done' ? <span className="pill pill--up">done</span> : b.status === 'running' ? <span className="pill pill--info">running</span> : b.status === 'cancelled' ? <span className="pill pill--idle">cancelled</span> : <span className="pill pill--down" title={b.error ?? ''}>{b.status}</span>}</td>
                      <td style={{ minWidth: 140 }}>
                        <span className="mono">{b.done}/{b.total}</span>
                        <div className="bar__track"><div className="bar__fill" style={{ width: `${b.total ? Math.round((100 * b.done) / b.total) : 0}%` }} /></div>
                      </td>
                      <td className="mono">{b.concurrency}</td>
                      <td>{SEARCH_TYPE_LABEL[b.search_type ?? 'semantic']}</td>
                      <td>{pct(b.retrieved, b.results)}</td>
                      <td>{pct(b.sent, b.results)}</td>
                      <td>{pct(b.rank1, b.results)}</td>
                      <td>{b.errors || '—'}</td>
                      <td>{b.avg_ms ? `${(b.avg_ms / 1000).toFixed(1)} s` : '—'}</td>
                      <td className="mono">{b.config_id ? `#${b.config_id}` : '—'}</td>
                      <td className="mono">{b.collection ?? '—'}</td>
                      <td onClick={(e) => e.stopPropagation()}>
                        {b.status === 'running'
                          ? <button type="button" className="btn btn--ghost btn--sm" onClick={() => api.evalCancel(b.id).then(load)}>Cancel</button>
                          : <button type="button" className="btn btn--ghost btn--sm" onClick={() => { if (window.confirm('Delete this batch and its results?')) api.evalDeleteBatch(b.id).then(load); }}>Delete</button>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </>
  );
}
