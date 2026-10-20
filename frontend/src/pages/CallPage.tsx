import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Empty } from '../components/Empty';
import { Icon } from '../components/Icon';
import { LatencyPill } from '../components/LatencyPill';
import { StatTile } from '../components/StatTile';
import { Topbar } from '../components/Topbar';
import { api } from '../api';
import { clock, latencyTone, ms, secs, when } from '../format';
import type { CallDetail, Turn, TurnSearch } from '../types';

/* Colour per contribution kind. Violet steps for the model work, greys for waiting. */
const SEGMENT_CLASS: Record<string, string> = {
  endpointing_wait: 'wait',
  turn_detection: 'wait',
  waiting_for_user: 'wait',
  first_request: 'wait',
  transcription: 'stt',
  llm_inference: 'llm',
  turn_completion: 'llm',
  llm_tool_call: 'llm',
  function_handler: 'llm',
  awaiting_speakable_text: 'agg',
  sentence_aggregation: 'agg',
  speech_synthesis: 'tts',
  output_transport: 'out',
};

function Breakdown({ t }: { t: Turn }) {
  const total = t.response_latency_ms;
  if (total === null) return <div className="lat"><span className="card__note">no latency measured</span></div>;
  const parts = (t.contributions ?? []).filter((c) => c.ms > 0);
  const known = parts.reduce((n, p) => n + p.ms, 0);
  const rest = Math.max(0, total - known);
  const legacy = parts.length === 0;
  const legacyParts = legacy
    ? ([
        { key: 'transcription', label: 'STT first byte', ms: t.stt_ttfb_ms },
        { key: 'llm_inference', label: 'LLM first byte', ms: t.llm_ttfb_ms },
        { key: 'speech_synthesis', label: 'TTS first byte', ms: t.tts_ttfb_ms },
      ].filter((p) => p.ms !== null) as { key: string; label: string; ms: number }[])
    : [];
  const shown = legacy ? legacyParts : parts;
  const shownTotal = legacy ? total : Math.max(total, known);
  return (
    <div className="lat">
      <div className="lat__head">
        <span className="card__label">{t.turn_index === 0 ? 'connect → greeting' : 'silence → reply'}</span>
        <span className="lat__total" style={{ color: `var(--${latencyTone(total) === 'up' ? 'up' : latencyTone(total) === 'warn' ? 'warn' : 'down'})` }}>{ms(total)}</span>
      </div>
      <div className="lat__track" title={`${ms(total)} total`}>
        {shown.map((p, i) => <span key={`${p.key}-${i}`} className={`lat__seg lat__seg--${SEGMENT_CLASS[p.key] ?? 'rest'}`} style={{ width: `${(p.ms / shownTotal) * 100}%` }} title={`${p.label} ${ms(p.ms)}`} />)}
        {!legacy && rest > 0 && <span className="lat__seg lat__seg--rest" style={{ width: `${(rest / shownTotal) * 100}%` }} title={`unattributed ${ms(rest)}`} />}
      </div>
      <div className="lat__legend">
        {shown.map((p, i) => <span key={`${p.key}-${i}`}><i className={`lat__seg lat__seg--${SEGMENT_CLASS[p.key] ?? 'rest'}`} />{p.label} {ms(p.ms)}</span>)}
        {!legacy && rest > 0 && <span><i className="lat__seg lat__seg--rest" />unattributed {ms(rest)}</span>}
        {legacy && <span className="muted">(older record: first-byte times only)</span>}
      </div>
    </div>
  );
}

function Searches({ searches }: { searches: TurnSearch[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!searches?.length) return null;
  return (
    <div className="kb">
      {searches.map((s, i) => (
        <div className="kb__search" key={`${s.at}-${i}`}>
          <div className="kb__head">
            <Icon name="search" size={13} />
            <span className="kb__q">Searched for <strong>{s.question}</strong></span>
            {s.translated_query && <span className="kb__q muted">→ {s.translated_query}</span>}
            <span className="topbar__spacer" />
            <span className="kb__meta">{s.translate_ms > 0 ? `translate ${s.translate_ms} ms · ` : ''}embed {s.embed_ms} ms · search {s.search_ms} ms</span>
          </div>
          {s.error && <div className="banner banner--error">search failed: {s.error}</div>}
          {s.hits.map((h) => {
            const key = `${s.at}-${h.source}-${h.chunk_index}`;
            return (
              <div className={`kb__hit${h.used ? '' : ' kb__hit--skipped'}`} key={key}>
                <button type="button" className="kb__row" onClick={() => setOpen(open === key ? null : key)}>
                  <span className={`pill ${h.score >= 0.5 ? 'pill--up' : h.score >= 0.25 ? 'pill--warn' : 'pill--idle'}`}>{(h.score * 100).toFixed(0)}%</span>
                  <span className="kb__source">{h.source} <span className="muted">· chunk {h.chunk_index}</span></span>
                  {!h.used && <span className="pill pill--idle">below floor</span>}
                  <span className="topbar__spacer" />
                  <span className="muted">{open === key ? 'hide' : 'show text'}</span>
                </button>
                {open === key && <p className="kb__text">{h.text}</p>}
              </div>
            );
          })}
          {s.hits.length === 0 && !s.error && <p className="card__note">No chunks returned.</p>}
        </div>
      ))}
    </div>
  );
}

export function CallPage() {
  const { callId } = useParams();
  const [data, setData] = useState<{ call: CallDetail; turns: Turn[] } | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    if (!callId) return;
    let cancelled = false;
    const load = () => api.call(callId).then((d) => { if (!cancelled) setData(d); }).catch(() => { if (!cancelled) setMissing(true); });
    load();
    const id = window.setInterval(() => { if (!data?.call.ended_at) load(); }, 5_000);
    return () => { cancelled = true; window.clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [callId]);

  if (missing) {
    return (
      <>
        <Topbar title="Call" back={{ to: '/calls', label: 'Calls' }} />
        <div className="page"><Empty icon="alert" title="Call not found" body="It may have been deleted." actions={<Link className="btn btn--ghost btn--sm" to="/calls">Back to calls</Link>} /></div>
      </>
    );
  }
  if (!data) return <div className="loading">Loading call…</div>;

  const { call, turns } = data;
  const latencies = turns.map((t) => t.response_latency_ms).filter((v): v is number => v !== null);
  const avg = latencies.length ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null;
  const max = latencies.length ? Math.max(...latencies) : null;
  const duration = call.ended_at ? Math.round((new Date(call.ended_at).getTime() - new Date(call.started_at).getTime()) / 1000) : null;

  return (
    <>
      <Topbar
        title={`${call.transport === 'webrtc' ? 'Web session' : 'Phone call'} · ${when(call.started_at)}`}
        back={{ to: '/calls', label: 'Calls' }}
        actions={call.ended_at ? <span className="pill pill--idle">{call.end_reason}</span> : <span className="pill pill--up">live</span>}
      />
      <div className="page stack">
        <div className="stats">
          <StatTile label="Duration" icon="calendar" value={duration === null ? 'live' : secs(duration)} note={`${turns.length} turns`} />
          <StatTile label="Avg latency" icon="refresh" value={ms(avg)} pill={{ tone: latencyTone(avg), text: avg === null ? 'no turns' : avg < 1500 ? 'within target' : avg < 3000 ? 'slow' : 'too slow' }} />
          <StatTile label="Slowest turn" icon="alert" value={ms(max)} note="silence to first bot audio" />
          <StatTile label="Barge-ins" icon="chat" value={turns.filter((t) => t.interrupted).length} note="assistant interrupted" />
        </div>

        <div className="facts">
          <div className="fact"><div className="fact__label">Channel</div><div className="fact__value">{call.transport} {call.direction ?? ''}</div></div>
          <div className="fact"><div className="fact__label">From → To</div><div className="fact__value mono">{call.from_number ?? '—'} → {call.to_number ?? '—'}</div></div>
          <div className="fact"><div className="fact__label">Language</div><div className="fact__value">{call.language}</div></div>
          <div className="fact"><div className="fact__label">Agent config</div><div className="fact__value">{call.config_id ? <Link className="link" to="/config">version #{call.config_id}</Link> : 'environment defaults'}</div></div>
          <div className="fact"><div className="fact__label">Plivo call</div><div className="fact__value mono">{call.call_id ?? '—'}</div></div>
          <div className="fact"><div className="fact__label">Ended</div><div className="fact__value">{call.ended_at ? when(call.ended_at) : 'in progress'}</div></div>
        </div>

        <section className="card">
          <div className="card__head">
            <span className="card__label">Conversation</span>
            <span className="topbar__spacer" />
            <span className="card__note">Latency is measured from the user falling silent to the first bot audio; turn 0 from connect to greeting.</span>
          </div>
          {turns.length === 0 ? (
            <Empty icon="chat" title="No turns yet" body="Turns appear as the conversation progresses." />
          ) : turns.map((t) => (
            <div className="exchange" key={t.turn_index}>
              <div className="exchange__n">#{t.turn_index}<br />{clock(t.completed_at)}</div>
              <div className="exchange__lines">
                {t.turn_index > 0 && (
                  <div className="line line--user">
                    <span className="line__who">User</span>
                    <span className={`line__text${t.user_text ? '' : ' line__text--empty'}`}>{t.user_text || 'nothing transcribed'}</span>
                  </div>
                )}
                <div className="line line--bot">
                  <span className="line__who">Agent</span>
                  <span className={`line__text${t.assistant_text ? '' : ' line__text--empty'}`}>
                    {t.assistant_text || 'no reply'}
                    {t.interrupted && <> <span className="pill pill--warn">interrupted</span></>}
                  </span>
                </div>
                <Searches searches={t.searches} />
              </div>
              <Breakdown t={t} />
            </div>
          ))}
        </section>
        {turns.length > 0 && (
          <section className="card">
            <div className="card__head"><span className="card__label">Turn timings</span></div>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>#</th><th>Latency</th><th>End of turn</th><th>STT first byte</th><th>LLM first byte</th><th>TTS first byte</th><th>Completed</th></tr></thead>
                <tbody>
                  {turns.map((t) => (
                    <tr key={t.turn_index}>
                      <td>{t.turn_index}</td>
                      <td><LatencyPill value={t.response_latency_ms} /></td>
                      <td>{ms(t.turn_release_ms)}</td>
                      <td>{ms(t.stt_ttfb_ms)}</td>
                      <td>{ms(t.llm_ttfb_ms)}</td>
                      <td>{ms(t.tts_ttfb_ms)}</td>
                      <td>{clock(t.completed_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </>
  );
}
