import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { BarList } from '../components/BarList';
import { Empty } from '../components/Empty';
import { Icon } from '../components/Icon';
import { LatencyPill } from '../components/LatencyPill';
import { StatTile } from '../components/StatTile';
import { Topbar } from '../components/Topbar';
import { api } from '../api';
import { latencyTone, ms, secs, when } from '../format';
import type { CallSummary, Stats } from '../types';

export function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [recent, setRecent] = useState<CallSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [s, c] = await Promise.all([api.stats(), api.calls(8)]);
        if (!cancelled) {
          setStats(s);
          setRecent(c);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };
    void load();
    const id = window.setInterval(load, 15_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const avgTone = latencyTone(stats?.avg_latency_ms);
  const avgText = { up: 'within target', warn: 'slower than 1.5 s', down: 'over 3 s', idle: 'no turns yet' }[avgTone];

  return (
    <>
      <Topbar
        title="Dashboard"
        actions={
          <Link className="btn btn--primary btn--sm" to="/config">
            Edit agent
          </Link>
        }
      />
      <div className="page stack">
        {error && <div className="banner banner--error">Cannot reach the portal backend: {error}</div>}

        <div className="stats">
          <StatTile
            label="Calls today"
            icon="calendar"
            value={stats?.calls_today ?? '—'}
            pill={stats && stats.live_calls > 0 ? { tone: 'up', text: `${stats.live_calls} live` } : { tone: 'idle', text: 'none live' }}
            note={stats ? `${stats.calls_total.toLocaleString()} all time` : undefined}
          />
          <StatTile
            label="Avg response latency"
            icon="refresh"
            value={ms(stats?.avg_latency_ms)}
            pill={{ tone: avgTone, text: avgText }}
            note={stats?.p95_latency_ms != null ? `p95 ${ms(stats.p95_latency_ms)}` : undefined}
          />
          <StatTile
            label="Turns"
            icon="chat"
            value={stats?.turns_total.toLocaleString() ?? '—'}
            pill={{ tone: 'violet', text: stats && stats.calls_total ? `${(stats.turns_total / stats.calls_total).toFixed(1)} per call` : '—' }}
            note="user and assistant exchanges"
          />
          <StatTile
            label="Barge-ins"
            icon="alert"
            value={stats?.interrupted_turns ?? '—'}
            pill={{ tone: stats && stats.interrupted_turns > 0 ? 'warn' : 'idle', text: 'turns interrupted' }}
            note="user spoke over the agent"
          />
        </div>

        <div className="grid grid--main">
          <section className="card">
            <div className="card__head">
              <span className="card__label">Recent calls</span>
              <span className="topbar__spacer" />
              <Link className="btn btn--ghost btn--sm" to="/calls">All calls</Link>
            </div>
            {recent.length === 0 ? (
              <Empty icon="chat" title="No calls yet" body="Start the bot with DATABASE_URL set and make a call; it will show up here." />
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr><th>Started</th><th>Channel</th><th>Duration</th><th>Turns</th><th>Avg latency</th><th>State</th><th>Conversation</th></tr>
                  </thead>
                  <tbody>
                    {recent.map((c) => (
                      <tr key={c.id} className="row-link" onClick={() => navigate(`/calls/${c.id}`)}>
                        <td><Link className="cell-link" to={`/calls/${c.id}`}>{when(c.started_at)}</Link><span className="cell-sub">{c.from_number ?? 'web'} → {c.to_number ?? 'agent'}</span></td>
                        <td><span className="pill pill--idle">{c.transport}</span></td>
                        <td>{secs(c.duration_secs)}</td>
                        <td>{c.turn_count}</td>
                        <td><LatencyPill value={c.avg_latency_ms} /></td>
                        <td>{c.ended_at ? <span className="pill pill--idle">{c.end_reason}</span> : <span className="pill pill--up">live</span>}</td>
                        <td><Link className="btn btn--ghost btn--sm" to={`/calls/${c.id}`} onClick={(e) => e.stopPropagation()}>Open <Icon name="chevron" size={12} /></Link></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <div className="stack">
            <section className="card">
              <div className="card__head"><span className="card__label">Latency distribution</span></div>
              <div className="card__body">
                {stats && stats.turns_total > 0 ? (
                  <BarList items={stats.latency_buckets.map((b) => ({ name: b.label, value: b.turns }))} unit="turns" />
                ) : (
                  <p className="card__note">No turns recorded yet.</p>
                )}
              </div>
            </section>
            <section className="card">
              <div className="card__head"><span className="card__label">By channel</span></div>
              <div className="card__body">
                {stats && stats.by_transport.length > 0 ? (
                  <BarList items={stats.by_transport.map((t) => ({ name: t.transport, value: t.calls, meta: `avg ${ms(t.avg_latency_ms)}` }))} unit="calls" />
                ) : (
                  <p className="card__note">No calls recorded yet.</p>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
