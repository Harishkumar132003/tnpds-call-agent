import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { PipecatClient, type TransportState } from '@pipecat-ai/client-js';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import { Empty } from '../components/Empty';
import { Icon } from '../components/Icon';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import type { AgentConfig, CallSummary, TestCallInfo } from '../types';

interface Line {
  key: number;
  who: 'user' | 'bot';
  text: string;
  final: boolean;
  at: number;
}

const clock = (ms: number) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const STATE_LABEL: Record<TransportState, string> = {
  disconnected: 'idle', initializing: 'starting mic', initialized: 'mic ready', authenticating: 'starting bot', authenticated: 'bot started',
  connecting: 'connecting', connected: 'connected', ready: 'in call', disconnecting: 'hanging up', error: 'error',
};

/** Place a WebRTC call to the running bot from the browser, with a live transcript,
 *  and link to the tracked call record afterwards. */
export function TestCallPage() {
  const toast = useToast();
  const [info, setInfo] = useState<TestCallInfo | null>(null);
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [state, setState] = useState<TransportState>('disconnected');
  const [lines, setLines] = useState<Line[]>([]);
  const [micLevel, setMicLevel] = useState(0);
  const [botLevel, setBotLevel] = useState(0);
  const [botSpeaking, setBotSpeaking] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [muted, setMuted] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [record, setRecord] = useState<CallSummary | null | 'looking'>(null);
  const client = useRef<PipecatClient | null>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const keyRef = useRef(1);
  const scroller = useRef<HTMLDivElement>(null);
  const meter = useRef<{ ctx: AudioContext; raf: number } | null>(null);

  /** The transport reports no remote audio level, so measure the bot's track here. */
  const watchBotLevel = (stream: MediaStream) => {
    stopBotLevel();
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) { const d = (v - 128) / 128; sum += d * d; }
      setBotLevel(Math.min(1, Math.sqrt(sum / buf.length) * 4));
      if (meter.current) meter.current.raf = requestAnimationFrame(tick);
    };
    meter.current = { ctx, raf: requestAnimationFrame(tick) };
  };
  const stopBotLevel = () => {
    if (!meter.current) return;
    cancelAnimationFrame(meter.current.raf);
    void meter.current.ctx.close().catch(() => undefined);
    meter.current = null;
    setBotLevel(0);
    setMicLevel(0);
  };

  const refresh = () => Promise.all([api.testCall(), api.config()]).then(([i, c]) => { setInfo(i); setConfig(c); });
  useEffect(() => { refresh().catch((e) => toast.show({ tone: 'error', title: 'Could not reach the portal', body: String(e) })); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { void client.current?.disconnect(); stopBotLevel(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (startedAt === null) { setElapsed(0); return; }
    const id = window.setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(id);
  }, [startedAt]);
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight }); }, [lines]);

  const push = (who: Line['who'], text: string, final: boolean) => {
    setLines((ls) => {
      const last = ls[ls.length - 1];
      // interim user transcripts replace the previous interim; bot text accumulates per turn
      if (who === 'user' && last?.who === 'user' && !last.final) return [...ls.slice(0, -1), { ...last, text, final }];
      if (who === 'bot' && last?.who === 'bot' && !last.final) return [...ls.slice(0, -1), { ...last, text: `${last.text} ${text}`.replace(/\s+/g, ' ') }];
      return [...ls, { key: keyRef.current++, who, text, final, at: Date.now() }];
    });
  };
  const closeBotLine = () => setLines((ls) => { const last = ls[ls.length - 1]; return last?.who === 'bot' && !last.final ? [...ls.slice(0, -1), { ...last, final: true }] : ls; });

  const start = async () => {
    if (!info) return;
    setLines([]);
    setRecord(null);
    setMuted(false);
    const transport = new SmallWebRTCTransport();
    const pc = new PipecatClient({
      transport,
      enableMic: true,
      enableCam: false,
      callbacks: {
        onTransportStateChanged: (s) => setState(s),
        onBotReady: () => setStartedAt((t) => t ?? Date.now()),
        onConnected: () => setStartedAt((t) => t ?? Date.now()),
        onDisconnected: () => { setStartedAt(null); setBotSpeaking(false); setUserSpeaking(false); closeBotLine(); stopBotLevel(); },
        onError: (m) => toast.show({ tone: 'error', title: 'Call error', body: JSON.stringify(m.data ?? m) }),
        onLocalAudioLevel: (l) => setMicLevel(l),
        onRemoteAudioLevel: (l) => setBotLevel(l),
        onUserStartedSpeaking: () => setUserSpeaking(true),
        onUserStoppedSpeaking: () => setUserSpeaking(false),
        onBotStartedSpeaking: () => setBotSpeaking(true),
        onBotStoppedSpeaking: () => { setBotSpeaking(false); closeBotLine(); },
        onUserTranscript: (d) => push('user', d.text, d.final),
        onBotTtsText: (d) => push('bot', d.text, false),
        onTrackStarted: (track, participant) => {
          if (participant?.local || track.kind !== 'audio' || !audio.current) return;
          const stream = new MediaStream([track]);
          audio.current.srcObject = stream;
          void audio.current.play().catch(() => undefined);
          watchBotLevel(stream);
        },
      },
    });
    client.current = pc;
    try {
      await pc.startBotAndConnect({ endpoint: `${info.bot_url}/start`, requestData: { transport: 'webrtc', enableDefaultIceServers: true } });
    } catch (e) {
      toast.show({ tone: 'error', title: 'Could not start the call', body: (e as Error).message });
      setState('disconnected');
      client.current = null;
    }
  };

  const hangUp = async () => {
    const pc = client.current;
    client.current = null;
    const since = startedAt ?? Date.now() - 60_000;
    try { await pc?.disconnect(); } catch { /* already closed */ }
    setState('disconnected');
    setStartedAt(null);
    // the recorder writes the call row at connect; give it a moment, then find it
    setRecord('looking');
    for (let attempt = 0; attempt < 5; attempt++) {
      await new Promise((r) => setTimeout(r, 800));
      const calls = await api.calls(10).catch(() => [] as CallSummary[]);
      const mine = calls.find((c) => c.transport === 'webrtc' && new Date(c.started_at).getTime() >= since - 15_000);
      if (mine) { setRecord(mine); return; }
    }
    setRecord(null);
  };

  const toggleMute = () => { const next = !muted; setMuted(next); client.current?.enableMic(!next); };

  const inCall = state !== 'disconnected' && state !== 'error';
  const language = config ? (config.languages.find((l) => l.code === (config.language ?? config.default_language))?.english_name ?? config.language) : '…';
  const ready = !!info?.reachable && info.webrtc !== false;
  const mm = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, '0')}`;

  return (
    <>
      <Topbar
        title="Test call"
        actions={info && (
          <>
            <span className={`pill ${info.reachable ? 'pill--up' : 'pill--down'}`}>bot {info.reachable ? 'online' : 'unreachable'}</span>
            {info.reachable && info.webrtc === false && <span className="pill pill--warn">WebRTC off (BOT_TRANSPORT)</span>}
            <span className="pill pill--info mono">{info.bot_url}</span>
          </>
        )}
      />
      <div className="page stack">
        {info && !info.reachable && <div className="banner banner--error">The bot at {info.bot_url} is not answering. Start it (python bot.py) or set BOT_PUBLIC_URL in backend/.env to where it runs.</div>}
        {info?.reachable && info.webrtc === false && <div className="banner banner--warn">The bot runs with BOT_TRANSPORT set, which disables WebRTC. Unset it so the same process takes phone calls and browser calls.</div>}

        <div className="grid grid--main">
          <section className="card" style={{ alignSelf: 'start' }}>
            <div className="card__head">
              <span className="card__label">Call</span>
              <span className="topbar__spacer" />
              <span className={`pill ${state === 'ready' ? 'pill--up' : inCall ? 'pill--info' : state === 'error' ? 'pill--down' : 'pill--idle'}`}>{STATE_LABEL[state]}{startedAt ? ` · ${mm}` : ''}</span>
            </div>
            <div className="card__body">
              <div className="tc">
                <div className={`tc__party${userSpeaking ? ' is-speaking' : ''}`}>
                  <div className="tc__avatar"><Icon name="chat" size={18} /></div>
                  <div className="tc__who">You</div>
                  <div className="tc__meter"><span style={{ width: `${Math.min(100, Math.round(micLevel * 100))}%` }} /></div>
                  <div className="tc__sub">{inCall ? (muted ? 'muted' : userSpeaking ? 'speaking' : 'listening') : 'microphone off'}</div>
                </div>
                <div className="tc__link"><span /><span /><span /></div>
                <div className={`tc__party${botSpeaking ? ' is-speaking' : ''}`}>
                  <div className="tc__avatar tc__avatar--bot"><Icon name="sparkle" size={18} /></div>
                  <div className="tc__who">{config?.agent_name ?? 'Agent'}</div>
                  <div className="tc__meter"><span style={{ width: `${Math.min(100, Math.round(botLevel * 100))}%` }} /></div>
                  <div className="tc__sub">{inCall ? (botSpeaking ? 'speaking' : 'thinking / listening') : 'not connected'}</div>
                </div>
              </div>
              <div className="tc__controls">
                {!inCall ? (
                  <button type="button" className="btn btn--primary" disabled={!ready} onClick={start}><Icon name="send" size={14} /> Start call</button>
                ) : (
                  <>
                    <button type="button" className="btn btn--danger" onClick={hangUp}><Icon name="close" size={14} /> Hang up</button>
                    <button type="button" className={`btn ${muted ? 'btn--solid' : 'btn--ghost'}`} onClick={toggleMute}>{muted ? 'Unmute' : 'Mute'}</button>
                  </>
                )}
                <span className="field__hint">Uses this browser's microphone and the same pipeline as a phone call. Latency, searches and cost are tracked like any call.</span>
              </div>
              <audio ref={audio} autoPlay playsInline />
            </div>
            <div className="card__foot">
              {record === 'looking' ? <span className="card__note">Looking for the call record…</span>
                : record ? <Link className="btn btn--ghost btn--sm" to={`/calls/${record.id}`}>Open call record · {record.turn_count} turns{record.avg_latency_ms ? ` · avg ${record.avg_latency_ms} ms` : ''}</Link>
                : <span className="card__note">{inCall ? 'Live: the record appears under Calls as the conversation progresses.' : 'After you hang up, the tracked call record is linked here.'}</span>}
            </div>
          </section>

          <div className="stack">
            <section className="card">
              <div className="card__head">
                <span className="card__label">Live transcript</span>
                <span className="topbar__spacer" />
                {lines.length > 0 && <button type="button" className="btn btn--quiet btn--sm" onClick={() => setLines([])}>Clear</button>}
              </div>
              <div className="tc__transcript" ref={scroller}>
                {lines.length === 0 ? (
                  <Empty icon="chat" title={inCall ? 'Waiting for speech' : 'No call yet'} body={inCall ? 'Say something; what the agent hears and says shows up here.' : 'Start a call to see the conversation as it happens.'} />
                ) : lines.map((l) => (
                  <div className={`line line--${l.who}${l.final ? '' : ' is-interim'}`} key={l.key}>
                    <span className="line__who">{l.who === 'user' ? 'You' : 'Agent'} <span className="line__time">{clock(l.at)}</span></span>
                    <span className="line__text">{l.text}</span>
                  </div>
                ))}
              </div>
            </section>

            <section className="card">
              <div className="card__head"><span className="card__label">What you are testing</span></div>
              <div className="card__body">
                <div className="facts facts--tight">
                  <div className="fact"><div className="fact__label">Config</div><div className="fact__value">{config ? <Link className="link" to="/config">version #{config.id ?? '—'}</Link> : '…'}</div></div>
                  <div className="fact"><div className="fact__label">Language</div><div className="fact__value">{language}</div></div>
                  <div className="fact"><div className="fact__label">Voice dictionary</div><div className="fact__value">{config?.tts_dict_id ? <Link className="link mono" to="/pronunciation">{config.tts_dict_id}</Link> : 'none'}</div></div>
                  <div className="fact"><div className="fact__label">Knowledge search</div><div className="fact__value">{config ? (config.knowledge_on ? 'on' : 'off') : '…'}</div></div>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
