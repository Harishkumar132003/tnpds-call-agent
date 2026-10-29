import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Empty } from '../components/Empty';
import { Icon } from '../components/Icon';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { chatApi, type ChatSessionInfo, type ChatStatus, type ChatStep, type ChatTurn } from './chatApi';

/** A message as the conversation shows it. `turn` links the agent's reply to its steps. */
interface Bubble {
  key: number;
  who: 'caller' | 'agent';
  text: string;
  turn?: number;
}

/** Real caller questions, so the chat can be tested without a Tamil keyboard. */
const STARTERS = [
  'புது ஸ்மார்ட் கார்டு அப்ளை பண்றது எப்படி?',
  'என்னோட பேரை கார்டுல இருந்து எப்படி நீக்குறது?',
  'குடும்பத் தலைவர் பேரை மாற்றணும், என்ன பண்ணனும்?',
  'ஆயிரம் ரூபாய் எப்படி வாங்குறது?',
  'என்னோட பயோமெட்ரிக் வேலை செய்யல, பொருட்கள் எப்படி வாங்குறது?',
];

const STEP_LABEL: Record<string, string> = {
  'topic check': 'Topic check',
  'llm decide': 'LLM 1 · decide',
  search: 'Search',
  'llm answer': 'LLM 2 · answer',
};

const clock = (iso: string) =>
  new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** One line of plain English for what a step did. */
function stepNote(step: ChatStep): string {
  switch (step.step) {
    case 'topic check':
      return step.error
        ? `check failed, context kept (${step.error})`
        : `${step.reason ?? '—'}${step.restarted ? ' · context restarted here' : ' · context kept'}`;
    case 'llm decide':
      if (step.error) return `failed: ${step.error}`;
      return step.searched
        ? `called the search tool · ${step.messages_sent ?? 0} messages in context`
        : `answered without searching · ${step.messages_sent ?? 0} messages in context`;
    case 'search':
      if (!step.found) return `nothing above the score cut · ${step.reason ?? 'no relevant information'}`;
      return `${step.sheets_sent ?? 0} of ${step.sheets_retrieved ?? 0} sheets sent to LLM 2`;
    case 'llm answer':
      return step.error
        ? `failed: ${step.error}`
        : `wrote the reply from the sheets · ${step.messages_sent ?? 0} messages in context`;
    default:
      return '';
  }
}

/** Text chat that runs a call's four steps: topic check, LLM 1, search, LLM 2. */
export function ChatPage() {
  const toast = useToast();
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [session, setSession] = useState<ChatSessionInfo | null>(null);
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [starting, setStarting] = useState(true);
  const keyRef = useRef(1);
  const scroller = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  const open = async (previous?: string) => {
    setStarting(true);
    try {
      if (previous) await chatApi.end(previous).catch(() => undefined);
      const fresh = await chatApi.start();
      setSession(fresh);
      setTurns([]);
      setSelected(null);
      setBubbles(
        fresh.greeting ? [{ key: keyRef.current++, who: 'agent', text: fresh.greeting }] : [],
      );
      box.current?.focus();
    } catch (e) {
      toast.show({ tone: 'error', title: 'Could not start the chat', body: (e as Error).message });
    } finally {
      setStarting(false);
    }
  };

  useEffect(() => {
    chatApi
      .status()
      .then((s) => {
        setStatus(s);
        return s.available ? open() : setStarting(false);
      })
      .catch((e) => {
        setStarting(false);
        toast.show({ tone: 'error', title: 'Could not reach the portal', body: String(e) });
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' });
  }, [bubbles, sending]);

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || !session || sending) return;
    setDraft('');
    setBubbles((bs) => [...bs, { key: keyRef.current++, who: 'caller', text: message }]);
    setSending(true);
    try {
      const turn = await chatApi.send(session.id, message);
      setTurns((ts) => [...ts, turn]);
      setSelected(turn.index);
      setBubbles((bs) => [
        ...bs,
        {
          key: keyRef.current++,
          who: 'agent',
          text: turn.reply || '(the model returned no text)',
          turn: turn.index,
        },
      ]);
      if (turn.error) toast.show({ tone: 'error', title: `Turn ${turn.index}`, body: turn.error });
    } catch (e) {
      toast.show({ tone: 'error', title: 'The turn failed', body: (e as Error).message });
      setBubbles((bs) => bs.slice(0, -1));
      setDraft(message);
    } finally {
      setSending(false);
      box.current?.focus();
    }
  };

  const turn = turns.find((t) => t.index === selected) ?? turns[turns.length - 1] ?? null;
  const language = session?.language_name ?? status?.language ?? '…';
  const unavailable = status !== null && !status.available;

  return (
    <>
      <Topbar
        title="Chat"
        actions={
          status && (
            <>
              <span className="pill pill--idle">text, no audio</span>
              <span className={`pill ${status.knowledge_on ? 'pill--up' : 'pill--warn'}`}>
                knowledge {status.knowledge_on ? 'on' : 'off'}
              </span>
              <span className={`pill ${status.topic_context_on ? 'pill--info' : 'pill--idle'}`}>
                topic context {status.topic_context_on ? 'on' : 'off'}
              </span>
              {status.min_score !== null && (
                <span className="pill pill--idle">cut {Math.round(status.min_score * 100)}%</span>
              )}
            </>
          )
        }
      />
      <div className="page stack">
        {unavailable && (
          <div className="banner banner--error">
            The chat needs the Sarvam key in backend/.env, the same one calls use.
          </div>
        )}

        <div className="grid grid--main">
          <section className="card chat">
            <div className="card__head">
              <span className="card__label">Conversation</span>
              <span className="topbar__spacer" />
              {session && <span className="card__note mono">{session.id.slice(0, 8)}</span>}
              <button
                type="button"
                className="btn btn--quiet btn--sm"
                disabled={starting || sending}
                onClick={() => open(session?.id)}
              >
                <Icon name="refresh" size={13} /> New chat
              </button>
            </div>

            <div className="chat__scroll" ref={scroller}>
              {bubbles.length === 0 && !sending ? (
                <Empty
                  icon="chat"
                  title={starting ? 'Starting the chat…' : 'Ask something in Tamil'}
                  body="The question runs through the same steps as a phone call: topic check, LLM 1 decides whether to search, the knowledge search, then LLM 2 writes the answer from the sheets."
                />
              ) : (
                bubbles.map((b) => {
                  const its = b.turn ? turns.find((t) => t.index === b.turn) : undefined;
                  return (
                    <div
                      key={b.key}
                      className={`chat__row chat__row--${b.who}${
                        b.turn && b.turn === turn?.index ? ' is-picked' : ''
                      }`}
                      onClick={() => b.turn && setSelected(b.turn)}
                    >
                      {its?.searched && its.filler && (
                        <span className="chat__filler">
                          while searching, the caller hears: {its.filler}
                        </span>
                      )}
                      <span className="chat__bubble">{b.text}</span>
                      <span className="chat__meta">
                        {b.who === 'caller' ? 'Caller' : 'Agent'}
                        {its && (
                          <>
                            {' · '}
                            {its.ms} ms
                            {its.searched ? ' · searched' : ' · no search'}
                            {' · '}
                            {clock(its.at)}
                          </>
                        )}
                      </span>
                    </div>
                  );
                })
              )}
              {sending && (
                <div className="chat__row chat__row--agent">
                  <span className="chat__bubble chat__bubble--wait">
                    <i className="chat__dots">
                      <span />
                      <span />
                      <span />
                    </i>
                    running the steps
                  </span>
                </div>
              )}
            </div>

            <div className="card__foot chat__composer">
              <div className="chat__chips">
                {STARTERS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className="chat__chip"
                    disabled={!session || sending}
                    onClick={() => void send(s)}
                    title={s}
                  >
                    {s}
                  </button>
                ))}
              </div>
              <div className="chat__input">
                <textarea
                  ref={box}
                  className="field__textarea chat__box"
                  rows={2}
                  placeholder={session ? 'கேள்விய தமிழ்ல டைப் பண்ணுங்க…' : 'Starting…'}
                  value={draft}
                  disabled={!session || sending}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      void send(draft);
                    }
                  }}
                />
                <button
                  type="button"
                  className="btn btn--primary"
                  disabled={!session || sending || !draft.trim()}
                  onClick={() => void send(draft)}
                >
                  <Icon name="send" size={14} /> Send
                </button>
              </div>
              <span className="field__hint">
                Enter sends, Shift+Enter adds a line. The agent answers in Tamil, from the uploaded
                sheets only.
              </span>
            </div>
          </section>

          <div className="stack">
            <section className="card">
              <div className="card__head">
                <span className="card__label">Steps</span>
                <span className="topbar__spacer" />
                {turn && <span className="card__note">turn {turn.index} · {turn.ms} ms</span>}
              </div>
              {!turn ? (
                <div className="card__body">
                  <span className="card__note">Send a message to see each step it ran.</span>
                </div>
              ) : (
                <div className="card__body stack">
                  {turn.steps.map((step, i) => (
                    <div className="chat__step" key={`${step.step}-${i}`}>
                      <div className="chat__step-head">
                        <span className="chat__step-name">{STEP_LABEL[step.step] ?? step.step}</span>
                        {step.step === 'topic check' && step.decision && (
                          <span className={`pill ${step.decision === 'NEW' ? 'pill--warn' : 'pill--idle'}`}>
                            {step.decision}
                          </span>
                        )}
                        {step.step === 'search' && (
                          <span className={`pill ${step.found ? 'pill--up' : 'pill--down'}`}>
                            {step.found ? 'found' : 'nothing'}
                          </span>
                        )}
                        <span className="topbar__spacer" />
                        <span className="chat__step-ms mono">{step.ms} ms</span>
                      </div>
                      <div className="chat__step-note">{stepNote(step)}</div>
                      {step.note && <div className="chat__step-note">{step.note}</div>}
                    </div>
                  ))}
                  {turn.dropped_text && (
                    <div className="banner banner--warn">
                      LLM 1 also wrote text, dropped as on a call: “{turn.dropped_text}”
                    </div>
                  )}
                  <div className="facts facts--tight">
                    <div className="fact">
                      <div className="fact__label">Prompt tokens</div>
                      <div className="fact__value">{turn.prompt_tokens}</div>
                    </div>
                    <div className="fact">
                      <div className="fact__label">Reply tokens</div>
                      <div className="fact__value">{turn.completion_tokens}</div>
                    </div>
                  </div>
                </div>
              )}
            </section>

            {turn && turn.searches.length > 0 && (
              <section className="card">
                <div className="card__head">
                  <span className="card__label">Sheets retrieved</span>
                </div>
                <div className="card__body stack">
                  {turn.searches.map((search, i) => (
                    <div key={`${search.question}-${i}`} className="stack">
                      <div className="chat__query">
                        <span className="chat__query-label">searched</span> {search.question}
                      </div>
                      {(search.rewritten_query ?? search.translated_query) && (
                        <div className="chat__query">
                          <span className="chat__query-label">as</span>{' '}
                          {search.rewritten_query ?? search.translated_query}
                        </div>
                      )}
                      {search.error && <div className="banner banner--error">{search.error}</div>}
                      {search.hits.map((hit) => (
                        <details className={`chat__hit${hit.used ? ' is-used' : ''}`} key={`${hit.document_id}-${hit.chunk_index}`}>
                          <summary>
                            <span className="chat__hit-score mono">
                              {Math.round(hit.score * 100)}%
                            </span>
                            <span className="chat__hit-name">{hit.source}</span>
                            <span className={`pill ${hit.used ? 'pill--up' : 'pill--idle'}`}>
                              {hit.used ? 'sent' : 'below cut'}
                            </span>
                          </summary>
                          <p className="chat__hit-text">{hit.text}</p>
                        </details>
                      ))}
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="card">
              <div className="card__head">
                <span className="card__label">What you are testing</span>
              </div>
              <div className="card__body">
                <div className="facts facts--tight">
                  <div className="fact">
                    <div className="fact__label">Config</div>
                    <div className="fact__value">
                      <Link className="link" to="/config">
                        version #{session?.config_id ?? status?.config_id ?? '—'}
                      </Link>
                    </div>
                  </div>
                  <div className="fact">
                    <div className="fact__label">Language</div>
                    <div className="fact__value">{language}</div>
                  </div>
                  <div className="fact">
                    <div className="fact__label">Model</div>
                    <div className="fact__value mono">{session?.model ?? status?.model ?? '…'}</div>
                  </div>
                  <div className="fact">
                    <div className="fact__label">Temperature</div>
                    <div className="fact__value">{session?.temperature ?? status?.temperature ?? '—'}</div>
                  </div>
                  <div className="fact">
                    <div className="fact__label">Sheets per search</div>
                    <div className="fact__value">top {status?.top_k ?? '—'}</div>
                  </div>
                  <div className="fact">
                    <div className="fact__label">Score cut</div>
                    <div className="fact__value">
                      {status?.min_score != null ? `${Math.round(status.min_score * 100)}%` : '—'}
                    </div>
                  </div>
                  <div className="fact">
                    <div className="fact__label">Query rewrite</div>
                    <div className="fact__value">{status?.query_rewrite_on ? 'on' : 'off'}</div>
                  </div>
                  <div className="fact">
                    <div className="fact__label">Turns</div>
                    <div className="fact__value">{turns.length}</div>
                  </div>
                </div>
              </div>
              <div className="card__foot">
                <span className="card__note">
                  The same prompts, knowledge search and score threshold as a call. Nothing here is
                  recorded as a call, so <Link className="link" to="/calls">Calls</Link> stays audio
                  only.
                </span>
              </div>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
