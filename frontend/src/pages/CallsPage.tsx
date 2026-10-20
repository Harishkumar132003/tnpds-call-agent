import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Empty } from '../components/Empty';
import { Icon } from '../components/Icon';
import { LatencyPill } from '../components/LatencyPill';
import { Topbar } from '../components/Topbar';
import { api } from '../api';
import { ms, secs, when } from '../format';
import type { CallSummary } from '../types';

type Filter = 'all' | 'webrtc' | 'plivo';

export function CallsPage() {
  const [calls, setCalls] = useState<CallSummary[] | null>(null);
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<Filter>('all');
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    const load = () => api.calls(500).then((c) => { if (!cancelled) setCalls(c); }).catch(() => undefined);
    load();
    const id = window.setInterval(load, 15_000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, []);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (calls ?? []).filter((c) => {
      if (filter !== 'all' && c.transport !== filter) return false;
      if (!q) return true;
      return [c.from_number, c.to_number, c.direction, c.language, c.end_reason, c.id]
        .some((v) => (v ?? '').toLowerCase().includes(q));
    });
  }, [calls, query, filter]);

  return (
    <>
      <Topbar title="Calls" onSearch={setQuery} searchPlaceholder="Number, language, reason…" />
      <div className="page stack">
        <div className="segmented segmented--inline">
          {(['all', 'webrtc', 'plivo'] as Filter[]).map((f) => (
            <button key={f} type="button" className={`segmented__opt${filter === f ? ' is-on' : ''}`} onClick={() => setFilter(f)}>
              {f === 'all' ? 'All channels' : f === 'webrtc' ? 'Web' : 'Phone'}
            </button>
          ))}
        </div>
        <section className="card">
          {calls === null ? (
            <div className="card__body"><span className="spinner" /></div>
          ) : shown.length === 0 ? (
            <Empty icon="chat" title={calls.length ? 'Nothing matches' : 'No calls yet'} body={calls.length ? 'Try a different search or channel.' : 'Calls appear here as soon as the bot handles one.'} />
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>Started</th><th>Channel</th><th>From → To</th><th>Lang</th><th>Duration</th><th>Turns</th><th>Avg latency</th><th>Max</th><th>State</th><th>Conversation</th></tr>
                </thead>
                <tbody>
                  {shown.map((c) => (
                    <tr key={c.id} className="row-link" onClick={() => navigate(`/calls/${c.id}`)}>
                      <td><Link className="cell-link" to={`/calls/${c.id}`}>{when(c.started_at)}</Link></td>
                      <td><span className="pill pill--idle">{c.transport}</span>{c.direction && <span className="cell-sub">{c.direction}</span>}</td>
                      <td className="mono">{c.from_number ?? '—'} → {c.to_number ?? '—'}</td>
                      <td>{c.language}</td>
                      <td>{secs(c.duration_secs)}</td>
                      <td>{c.turn_count}</td>
                      <td><LatencyPill value={c.avg_latency_ms} /></td>
                      <td>{ms(c.max_latency_ms)}</td>
                      <td>
                        {c.ended_at ? <span className="pill pill--idle">{c.end_reason}</span> : <span className="pill pill--up">live</span>}
                        {c.had_interruption && <span className="cell-sub">barge-in</span>}
                      </td>
                      <td><Link className="btn btn--ghost btn--sm" to={`/calls/${c.id}`} onClick={(e) => e.stopPropagation()}>Open <Icon name="chevron" size={12} /></Link></td>
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
