import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import type { AgentConfig, ConfigVersion } from '../types';

/* The page is organised the way the agent runs: a pipeline of prompt steps in
 * execution order, one editor at a time (master–detail), with the composed
 * text the model actually receives beside it. Every save is an immutable
 * config version with a note, as before; a section save only touches its own
 * fields. */

type Section = 'language' | 'idle' | 'prompt' | 'topic' | 'knowledge' | 'rewrite';
const SECTION_LABEL: Record<Section, string> = { language: 'Language & opening line', idle: 'Idle handling', prompt: 'Persona & temperature', topic: 'Topic check prompt', knowledge: 'Knowledge rules', rewrite: 'Query rewrite prompt' };

type StepId = 'persona' | 'topic' | 'router' | 'rewrite' | 'answer';
type View = 'prompts' | 'call' | 'history';

/** Everything a version stores; a section save changes only its own fields. */
interface Draft {
  language: string;
  greeting: string;
  idleSecs: string;
  idleCheck: string;
  idleBye: string;
  prompt: string;
  useTemp: boolean;
  temp: number;
  topicPrompt: string;
  routerPrompt: string;
  answerPrompt: string;
  rewritePrompt: string;
}

/** Built-in idle values for a language: what runs when nothing is stored. */
function idleDefaults(c: AgentConfig, language: string) {
  return {
    secs: String(c.idle_defaults.prompt_secs),
    check: c.idle_defaults.check_lines[language] ?? c.idle_defaults.check_lines['en-IN'],
    bye: c.idle_defaults.goodbye_lines[language] ?? c.idle_defaults.goodbye_lines['en-IN'],
  };
}

function draftFrom(c: AgentConfig): Draft {
  const language = c.language ?? c.default_language;
  const d = idleDefaults(c, language);
  return {
    language,
    greeting: c.greeting ?? '',
    // show the values in effect, editable; equal-to-default is stored as "use default"
    idleSecs: c.idle_prompt_secs === null ? d.secs : String(c.idle_prompt_secs),
    idleCheck: c.idle_check_line ?? d.check,
    idleBye: c.idle_goodbye_line ?? d.bye,
    prompt: c.system_prompt ?? '',
    useTemp: c.temperature !== null,
    temp: c.temperature ?? 0.7,
    topicPrompt: c.topic_prompt ?? c.topic_prompt_default,
    routerPrompt: c.router_prompt ?? c.router_prompt_default,
    answerPrompt: c.answer_prompt ?? c.answer_prompt_default,
    rewritePrompt: c.rewrite_prompt ?? c.rewrite_prompt_default,
  };
}

function SaveBar({ dirty, saving, note, setNote, label }: { dirty: boolean; saving: boolean; note: string; setNote: (v: string) => void; label: string }) {
  return (
    <div className="form-actions">
      <button type="submit" className="btn btn--primary" disabled={saving || !dirty}>{saving ? 'Saving…' : `Save ${label.toLowerCase()}`}</button>
      <input className="field__input" style={{ maxWidth: 320 }} value={note} onChange={(e) => setNote(e.target.value)} placeholder="change note (optional)" />
      {dirty && <span className="card__note">unsaved changes</span>}
    </div>
  );
}

export function ConfigPage() {
  const toast = useToast();
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [history, setHistory] = useState<ConfigVersion[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [notes, setNotes] = useState<Record<Section, string>>({ language: '', idle: '', prompt: '', topic: '', knowledge: '', rewrite: '' });
  const [saving, setSaving] = useState<Section | null>(null);
  const [preview, setPreview] = useState<{ prompt: string; answer: string | null; error: string | null }>({ prompt: '', answer: null, error: null });
  const [open, setOpen] = useState<number | null>(null);
  const [view, setView] = useState<View>('prompts');
  const [step, setStep] = useState<StepId>('persona');

  const load = async () => {
    const [c, h] = await Promise.all([api.config(), api.configHistory()]);
    setConfig(c);
    setHistory(h);
    setDraft(draftFrom(c));
  };
  useEffect(() => { load().catch((e) => toast.show({ tone: 'error', title: 'Could not load config', body: String(e) })); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Live preview of the two composed prompts the agent will receive.
  useEffect(() => {
    if (!draft?.language) return;
    const id = window.setTimeout(() => {
      api.previewPrompt(draft.language, draft.prompt.trim() || null, draft.routerPrompt.trim() || null, draft.answerPrompt.trim() || null)
        .then((r) => setPreview({ prompt: r.prompt, answer: r.answer_prompt, error: null }))
        .catch((e) => setPreview((p) => ({ ...p, error: (e as Error).message })));
    }, 250);
    return () => window.clearTimeout(id);
  }, [draft?.language, draft?.prompt, draft?.routerPrompt, draft?.answerPrompt]);

  if (!config || !draft) return <div className="loading">Loading config…</div>;

  const saved = draftFrom(config);
  const set = <K extends keyof Draft>(key: K, value: Draft[K]) => setDraft((d) => (d ? { ...d, [key]: value } : d));
  const defaults = idleDefaults(config, draft.language);
  const isDefault = { secs: draft.idleSecs.trim() === defaults.secs, check: draft.idleCheck.trim() === defaults.check, bye: draft.idleBye.trim() === defaults.bye };
  const selected = config.languages.find((l) => l.code === draft.language);
  const langName = (code: string | null) => config.languages.find((x) => x.code === (code ?? config.default_language))?.english_name ?? code ?? 'default';

  const dirty: Record<Section, boolean> = {
    language: draft.language !== saved.language || draft.greeting.trim() !== saved.greeting.trim(),
    idle: draft.idleSecs.trim() !== saved.idleSecs || draft.idleCheck.trim() !== saved.idleCheck.trim() || draft.idleBye.trim() !== saved.idleBye.trim(),
    prompt: draft.prompt.trim() !== saved.prompt.trim() || draft.useTemp !== saved.useTemp || (draft.useTemp && draft.temp !== saved.temp),
    topic: draft.topicPrompt.trim() !== saved.topicPrompt.trim(),
    knowledge: draft.routerPrompt.trim() !== saved.routerPrompt.trim() || draft.answerPrompt.trim() !== saved.answerPrompt.trim(),
    rewrite: draft.rewritePrompt.trim() !== saved.rewritePrompt.trim(),
  };
  const topicIsDefault = draft.topicPrompt.trim() === config.topic_prompt_default.trim();
  const routerIsDefault = draft.routerPrompt.trim() === config.router_prompt_default.trim();
  const answerIsDefault = draft.answerPrompt.trim() === config.answer_prompt_default.trim();
  const rewriteIsDefault = draft.rewritePrompt.trim() === config.rewrite_prompt_default.trim();

  /** Build the full version body: the saved values everywhere, except the section being saved. */
  const bodyFor = (section: Section) => {
    const d = { ...saved, ...(section === 'language' ? { language: draft.language, greeting: draft.greeting } : {}),
      ...(section === 'idle' ? { idleSecs: draft.idleSecs, idleCheck: draft.idleCheck, idleBye: draft.idleBye } : {}),
      ...(section === 'prompt' ? { prompt: draft.prompt, useTemp: draft.useTemp, temp: draft.temp } : {}),
      ...(section === 'knowledge' ? { routerPrompt: draft.routerPrompt, answerPrompt: draft.answerPrompt } : {}) };
    return {
      system_prompt: d.prompt.trim() || null,
      temperature: d.useTemp ? d.temp : null,
      language: d.language === config.default_language && !config.language && section !== 'language' ? null : d.language,
      greeting: d.greeting.trim() || null,
      // values equal to the built-in default are stored as null, so they keep following
      // the environment / language default; anything else is stored as typed
      idle_prompt_secs: d.idleSecs.trim() === '' || d.idleSecs.trim() === defaults.secs ? null : Number(d.idleSecs),
      idle_check_line: !d.idleCheck.trim() || d.idleCheck.trim() === defaults.check ? null : d.idleCheck.trim(),
      idle_goodbye_line: !d.idleBye.trim() || d.idleBye.trim() === defaults.bye ? null : d.idleBye.trim(),
      router_prompt: !d.routerPrompt.trim() || d.routerPrompt.trim() === config.router_prompt_default.trim() ? null : d.routerPrompt.trim(),
      answer_prompt: !d.answerPrompt.trim() || d.answerPrompt.trim() === config.answer_prompt_default.trim() ? null : d.answerPrompt.trim(),
      // only this section changes the rewrite prompt; "" (empty or equal to the built-in) = built-in
      rewrite_prompt: section !== 'rewrite' ? null : !draft.rewritePrompt.trim() || rewriteIsDefault ? '' : draft.rewritePrompt.trim(),
      // same "null keeps, empty resets" rule for the topic classifier's prompt
      topic_prompt: section !== 'topic' ? null : !draft.topicPrompt.trim() || topicIsDefault ? '' : draft.topicPrompt.trim(),
      // the Sarvam dictionary is managed on the Pronunciation page; null keeps the active one
      pronunciations: null,
      // models are managed on the Models page; the active version's choices are kept
      stt_model: config.stt_model,
      llm_model: config.llm_model,
      tts_model: config.tts_model,
      tts_speaker: config.tts_speaker,
      translate_model: config.translate_model,
      embedding_model: config.embedding_model,
      note: notes[section].trim() || `${SECTION_LABEL[section]} updated`,
    };
  };

  const save = (section: Section) => async (e: FormEvent) => {
    e.preventDefault();
    setSaving(section);
    try {
      const { id } = await api.saveConfig(bodyFor(section));
      toast.show({ tone: 'success', title: `Saved version #${id}`, body: `${SECTION_LABEL[section]} updated; other sections unchanged. New sessions use it immediately.` });
      setNotes((n) => ({ ...n, [section]: '' }));
      await load();
    } catch (err) {
      toast.show({ tone: 'error', title: 'Not saved', body: (err as Error).message });
    } finally {
      setSaving(null);
    }
  };

  const restore = (v: ConfigVersion) => {
    setDraft({
      language: v.language ?? config.default_language,
      greeting: v.greeting ?? '',
      idleSecs: v.idle_prompt_secs === null ? idleDefaults(config, v.language ?? config.default_language).secs : String(v.idle_prompt_secs),
      idleCheck: v.idle_check_line ?? idleDefaults(config, v.language ?? config.default_language).check,
      idleBye: v.idle_goodbye_line ?? idleDefaults(config, v.language ?? config.default_language).bye,
      prompt: v.system_prompt ?? '',
      useTemp: v.temperature !== null,
      temp: v.temperature ?? 0.7,
      topicPrompt: v.topic_prompt ?? config.topic_prompt_default,
      routerPrompt: v.router_prompt ?? config.router_prompt_default,
      answerPrompt: v.answer_prompt ?? config.answer_prompt_default,
      rewritePrompt: v.rewrite_prompt ?? config.rewrite_prompt_default,
    });
    setView('prompts');
    toast.show({ tone: 'info', title: `Loaded version #${v.id} into the editors`, body: 'Nothing is saved yet: review each section and save the ones you want to restore.' });
  };

  /* ---- the pipeline steps, in the order they run on a caller message ---- */
  const steps: { id: StepId; name: string; role: string; on: boolean; section: Section; custom: boolean }[] = [
    { id: 'persona', name: 'Persona', role: 'voice & behaviour, base of LLM 1 + 2', on: true, section: 'prompt', custom: config.system_prompt !== null },
    { id: 'topic', name: 'Topic check', role: 'new question or follow-up → context', on: config.topic_context_on, section: 'topic', custom: config.topic_prompt !== null },
    { id: 'router', name: 'LLM 1 · Decide', role: 'search the FAQ, or reply without facts', on: config.knowledge_on, section: 'knowledge', custom: config.router_prompt !== null },
    { id: 'rewrite', name: 'Query rewrite', role: 'caller’s words → one English query', on: config.query_rewrite_on, section: 'rewrite', custom: config.rewrite_prompt !== null },
    { id: 'answer', name: 'LLM 2 · Answer', role: 'grounded answer from the sheets only', on: config.knowledge_on, section: 'knowledge', custom: config.answer_prompt !== null },
  ];
  const visibleSteps = steps.filter((s) => s.on);
  const active: StepId = visibleSteps.some((s) => s.id === step) ? step : 'persona';

  const sectionForm = (section: Section, body: ReactNode) => (
    <form className="card" onSubmit={save(section)}>
      <div className="card__body">
        {body}
        <SaveBar dirty={dirty[section]} saving={saving === section} note={notes[section]} setNote={(v) => setNotes((n) => ({ ...n, [section]: v }))} label={SECTION_LABEL[section]} />
      </div>
    </form>
  );

  const resetPill = (isDef: boolean, onReset: () => void) =>
    isDef ? <span className="pill pill--idle">built-in</span> : (
      <>
        <span className="pill pill--violet">customised</span>
        <button type="button" className="btn btn--quiet" onClick={onReset}>reset to built-in</button>
      </>
    );

  /* ---- per-step editor ---- */
  const editor: Record<StepId, ReactNode> = {
    persona: sectionForm('prompt', (
      <>
        <div className="flow-editor__head">
          <div className="flow-editor__text">
            <h2 className="flow-editor__title">Persona — the agent's voice</h2>
            <p className="flow-editor__sub">Who is speaking and how: identity, Tamil speaking style, phone etiquette. It is the shared base of both LLM steps; facts never belong here.</p>
          </div>
          {resetPill(!draft.prompt.trim(), () => set('prompt', ''))}
          <button type="button" className="btn btn--quiet" onClick={() => set('prompt', config.default_prompt)}>load built-in to edit</button>
        </div>
        <div className="field">
          <textarea className="field__textarea" rows={16} value={draft.prompt} onChange={(e) => set('prompt', e.target.value)} placeholder={config.default_prompt} />
          <span className="field__hint">
            Write it once in English. {'{language_english_name}'}, {'{language_native_name}'} and {'{agent_name}'} are filled from the selected language and AGENT_NAME. Empty = built-in prompt.
          </span>
        </div>
        <div className="field">
          <label className="field__label">Temperature</label>
          <div className="range">
            <label className="range__toggle"><input type="checkbox" checked={draft.useTemp} onChange={(e) => set('useTemp', e.target.checked)} />override</label>
            <input type="range" min={0} max={2} step={0.05} value={draft.temp} disabled={!draft.useTemp} onChange={(e) => set('temp', Number(e.target.value))} />
            <span className="range__value">{draft.useTemp ? draft.temp.toFixed(2) : 'default'}</span>
          </div>
          <span className="field__hint">Lower is more consistent, higher more varied. Applies to LLM 1 and LLM 2; unchecked keeps the provider default.</span>
        </div>
      </>
    )),
    topic: sectionForm('topic', (
      <>
        <div className="flow-editor__head">
          <div className="flow-editor__text">
            <h2 className="flow-editor__title">Topic check — context control</h2>
            <p className="flow-editor__sub">Runs on every caller message, before LLM 1. NEW restarts the context at that message (the pills on the call page); FOLLOW_UP keeps it growing. On failure or timeout the turn counts as FOLLOW_UP.</p>
          </div>
          {resetPill(topicIsDefault, () => set('topicPrompt', config.topic_prompt_default))}
        </div>
        <div className="field">
          <textarea className="field__textarea" rows={14} value={draft.topicPrompt} onChange={(e) => set('topicPrompt', e.target.value)} />
          <span className="field__hint">
            Contract: it sees the current topic's conversation and the caller's latest message, and must reply with one line — NEW or FOLLOW_UP, a colon, then a short reason.
          </span>
        </div>
      </>
    )),
    router: sectionForm('knowledge', (
      <>
        <div className="flow-editor__head">
          <div className="flow-editor__text">
            <h2 className="flow-editor__title">LLM 1 — decide & search</h2>
            <p className="flow-editor__sub">The reply to each caller message: decides whether to call search_knowledge_base (its spoken text is dropped when it does) and handles turns that need no facts — greetings, thanks, asking for a missing detail. It must not answer factual questions itself.</p>
          </div>
          {resetPill(routerIsDefault, () => set('routerPrompt', config.router_prompt_default))}
        </div>
        <div className="field">
          <textarea className="field__textarea" rows={14} value={draft.routerPrompt} onChange={(e) => set('routerPrompt', e.target.value)} />
          <span className="field__hint">Appended after the persona. Saving stores the LLM 1 and LLM 2 rules together as one version.</span>
        </div>
      </>
    )),
    rewrite: sectionForm('rewrite', (
      <>
        <div className="flow-editor__head">
          <div className="flow-editor__text">
            <h2 className="flow-editor__title">Query rewrite — search language</h2>
            <p className="flow-editor__sub">Turns the caller's recent messages into the one English question the FAQ search runs on. Runs inside the tool call, between LLM 1 and the vector search; on failure the search falls back to translation.</p>
          </div>
          {resetPill(rewriteIsDefault, () => set('rewritePrompt', config.rewrite_prompt_default))}
        </div>
        <div className="field">
          <textarea className="field__textarea" rows={14} value={draft.rewritePrompt} onChange={(e) => set('rewritePrompt', e.target.value)} />
          <span className="field__hint">
            Contract: it receives only the caller's last 10 messages (oldest first) and must reply with one English search query per line, one for each question in the latest message.
          </span>
        </div>
      </>
    )),
    answer: sectionForm('knowledge', (
      <>
        <div className="flow-editor__head">
          <div className="flow-editor__text">
            <h2 className="flow-editor__title">LLM 2 — answer from results</h2>
            <p className="flow-editor__sub">The reply after the search results arrive: writes the spoken answer only from the returned sheets. All grounding rules live here — keep every qualifier, never add own knowledge, say "I don't have that information" when nothing relevant was found.</p>
          </div>
          {resetPill(answerIsDefault, () => set('answerPrompt', config.answer_prompt_default))}
        </div>
        <div className="field">
          <textarea className="field__textarea" rows={14} value={draft.answerPrompt} onChange={(e) => set('answerPrompt', e.target.value)} />
          <span className="field__hint">Appended after the persona, replacing the LLM 1 rules when the request carries search results. Saving stores the LLM 1 and LLM 2 rules together as one version.</span>
        </div>
      </>
    )),
  };

  /* ---- the composed text the model actually receives, per selection ---- */
  const previewCard = (() => {
    if (preview.error) return <div className="banner banner--error">{preview.error}</div>;
    const llm1 = (
      <>
        <div className="card__note" style={{ marginBottom: 6 }}>LLM 1 receives (persona + decide rules)</div>
        <pre className="preview-prompt">{preview.prompt}</pre>
      </>
    );
    const llm2 = preview.answer ? (
      <>
        <div className="card__note" style={{ margin: active === 'answer' ? '0 0 6px' : '12px 0 6px' }}>LLM 2 receives (persona + answer rules)</div>
        <pre className="preview-prompt">{preview.answer}</pre>
      </>
    ) : null;
    if (active === 'router') return llm1;
    if (active === 'answer') return llm2 ?? llm1;
    if (active === 'topic') return (
      <>
        <div className="card__note" style={{ marginBottom: 6 }}>The classifier receives (this prompt + the current topic + the latest message)</div>
        <pre className="preview-prompt">{draft.topicPrompt}</pre>
      </>
    );
    if (active === 'rewrite') return (
      <>
        <div className="card__note" style={{ marginBottom: 6 }}>The rewriter receives (this prompt + the caller's last 10 messages)</div>
        <pre className="preview-prompt">{draft.rewritePrompt}</pre>
      </>
    );
    return <>{llm1}{llm2}</>;
  })();

  return (
    <>
      <Topbar
        title="Agent configuration"
        actions={<span className="pill pill--violet">{config.id ? `active version #${config.id}` : 'environment defaults'}</span>}
      />
      <div className="page">
        <div className="segmented" role="tablist">
          {(['prompts', 'call', 'history'] as View[]).map((v) => (
            <button key={v} type="button" role="tab" aria-selected={view === v} className={`segmented__opt${view === v ? ' is-on' : ''}`} onClick={() => setView(v)}>
              {v === 'prompts' ? 'Prompts' : v === 'call' ? 'Call settings' : `History (${history.length})`}
            </button>
          ))}
        </div>

        {view === 'prompts' && (
          <>
            {/* The agent's steps on one caller message, in run order. Click a step to edit its prompt. */}
            <div className="flow" role="tablist" aria-label="Prompt steps">
              {visibleSteps.map((s, i) => (
                <div className="flow__item" key={s.id}>
                  {i > 0 && <span className="flow__arrow" aria-hidden>→</span>}
                  <button type="button" role="tab" aria-selected={active === s.id} className={`flow__step${active === s.id ? ' is-on' : ''}`} onClick={() => setStep(s.id)}>
                    <span className="flow__name">{s.name}{dirty[s.section] && <i className="flow__dot" title="unsaved changes" />}</span>
                    <span className="flow__role">{s.role}</span>
                    <span className={`pill ${s.custom ? 'pill--violet' : 'pill--idle'}`}>{s.custom ? 'customised' : 'built-in'}</span>
                  </button>
                </div>
              ))}
            </div>

            <div className="grid grid--main">
              <div className="stack">{editor[active]}</div>
              <div className="stack">
                <section className="card">
                  <div className="card__head">
                    <span className="card__label">What the model receives</span>
                    <span className="topbar__spacer" />
                    {selected && <span className="pill pill--violet">{selected.native_name} · {selected.code}</span>}
                  </div>
                  <div className="card__body">{previewCard}</div>
                </section>
              </div>
            </div>
          </>
        )}

        {view === 'call' && (
          <div className="grid grid--main">
            <div className="stack">
              {sectionForm('language', (
                <>
                  <div className="flow-editor__head">
                    <div className="flow-editor__text">
                      <h2 className="flow-editor__title">Language & opening line</h2>
                      <p className="flow-editor__sub">The conversation language drives speech recognition, synthesis and the prompt's language rules.</p>
                    </div>
                  </div>
                  <div className="field">
                    <label className="field__label" htmlFor="language">Conversation language</label>
                    <select id="language" className="field__select lang-select" value={draft.language} onChange={(e) => set('language', e.target.value)}>
                      {config.languages.map((l) => <option key={l.code} value={l.code}>{l.native_name} — {l.english_name} ({l.code})</option>)}
                    </select>
                    <span className="field__hint">Environment default: {langName(null)}.</span>
                  </div>
                  <div className="field">
                    <label className="field__label" htmlFor="greeting">Opening line <span className="muted">(optional, in {selected?.english_name ?? 'the selected language'})</span></label>
                    <input id="greeting" className="field__input lang-select" style={{ maxWidth: 'none' }} value={draft.greeting} onChange={(e) => set('greeting', e.target.value)} placeholder="e.g. வணக்கம். நான் உங்களுக்கு எப்படி உதவலாம்?" />
                    <span className="field__hint">Spoken instantly through TTS when the call connects, skipping the LLM. Leave empty to let the LLM greet.</span>
                  </div>
                </>
              ))}
              {sectionForm('idle', (
                <>
                  <div className="flow-editor__head">
                    <div className="flow-editor__text">
                      <h2 className="flow-editor__title">Idle handling</h2>
                      <p className="flow-editor__sub">Silence after the agent speaks → one "still there?" check → goodbye and hang up. These fixed lines are spoken directly, skipping the LLM.</p>
                    </div>
                  </div>
                  <div className="field">
                    <label className="field__label" htmlFor="idle-secs">
                      Silence before the check, and again before hang-up (seconds)
                      {isDefault.secs ? <span className="pill pill--idle" style={{ marginLeft: 8 }}>default</span> : <button type="button" className="btn btn--quiet" onClick={() => set('idleSecs', defaults.secs)}>reset to {defaults.secs}</button>}
                    </label>
                    <input id="idle-secs" className="field__input" type="number" min={0} max={120} step={1} style={{ maxWidth: 160 }} value={draft.idleSecs} onChange={(e) => set('idleSecs', e.target.value)} />
                    <span className="field__hint">Currently in effect. 0 turns idle handling off.</span>
                  </div>
                  <div className="field">
                    <label className="field__label" htmlFor="idle-check">
                      "Are you still there?" line <span className="muted">(in {selected?.english_name ?? 'the selected language'})</span>
                      {isDefault.check ? <span className="pill pill--idle" style={{ marginLeft: 8 }}>default</span> : <button type="button" className="btn btn--quiet" onClick={() => set('idleCheck', defaults.check)}>reset to default</button>}
                    </label>
                    <input id="idle-check" className="field__input lang-select" style={{ maxWidth: 'none' }} value={draft.idleCheck} onChange={(e) => set('idleCheck', e.target.value)} />
                  </div>
                  <div className="field">
                    <label className="field__label" htmlFor="idle-bye">
                      Goodbye line before hanging up
                      {isDefault.bye ? <span className="pill pill--idle" style={{ marginLeft: 8 }}>default</span> : <button type="button" className="btn btn--quiet" onClick={() => set('idleBye', defaults.bye)}>reset to default</button>}
                    </label>
                    <input id="idle-bye" className="field__input lang-select" style={{ maxWidth: 'none' }} value={draft.idleBye} onChange={(e) => set('idleBye', e.target.value)} />
                    <span className="field__hint">The exact lines the agent will speak; "default" means the built-in line for the selected language.</span>
                  </div>
                </>
              ))}
            </div>
            <div className="stack">
              <section className="card">
                <div className="card__head"><span className="card__label">Managed elsewhere</span></div>
                <div className="card__body">
                  <ul className="hint-list">
                    <li>Models per step (STT, LLM, TTS, embedding) — <b>Models</b> page.</li>
                    <li>Pronunciation dictionary — <b>Pronunciation</b> page.</li>
                    <li>FAQ documents and retrieval tests — <b>Knowledge</b> page.</li>
                  </ul>
                </div>
              </section>
            </div>
          </div>
        )}

        {view === 'history' && (
          <section className="card">
            <div className="card__head">
              <span className="card__label">Versions</span>
              <span className="topbar__spacer" />
              <span className="card__note">Every save is a new immutable version; Restore loads one back into the editors.</span>
            </div>
            {history.length === 0 ? (
              <div className="card__body"><p className="card__note">No versions saved yet.</p></div>
            ) : (
              <div className="list">
                {history.map((v) => (
                  <div className="list__row" key={v.id}>
                    <button type="button" className="list__open" onClick={() => setOpen(open === v.id ? null : v.id)}>
                      <div className="list__name">#{v.id} {v.id === config.id && <span className="pill pill--up">active</span>} {v.note && <span className="card__note">· {v.note}</span>}</div>
                      <div className="list__meta">{when(v.created_at)} · {langName(v.language)} · temp {v.temperature ?? 'default'} · {v.system_prompt ? `${v.system_prompt.length} chars` : 'built-in prompt'}{v.greeting ? ' · fixed greeting' : ''}{v.idle_prompt_secs !== null ? ` · idle ${v.idle_prompt_secs}s` : ''}{v.router_prompt || v.answer_prompt ? ' · custom knowledge rules' : ''}{v.topic_prompt ? ' · custom topic prompt' : ''}{v.rewrite_prompt ? ' · custom rewrite prompt' : ''}{v.pronunciations?.length ? ` · ${v.pronunciations.length} pronunciations` : ''}{[v.stt_model, v.llm_model, v.tts_model, v.tts_speaker, v.translate_model].some(Boolean) ? ` · models: ${[v.stt_model, v.llm_model, v.tts_model && v.tts_speaker ? `${v.tts_model}/${v.tts_speaker}` : v.tts_model || v.tts_speaker, v.translate_model].filter(Boolean).join(', ')}` : ''}</div>
                      {open === v.id && v.system_prompt && <pre className="preview-prompt" style={{ marginTop: 8 }}>{v.system_prompt}</pre>}
                    </button>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={() => restore(v)}>Restore</button>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}
      </div>
    </>
  );
}
