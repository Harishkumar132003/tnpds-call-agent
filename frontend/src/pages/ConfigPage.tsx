import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import type { AgentConfig, ConfigVersion } from '../types';

type Section = 'language' | 'idle' | 'prompt' | 'knowledge' | 'rewrite';
const SECTION_LABEL: Record<Section, string> = { language: 'Language & opening line', idle: 'Idle handling', prompt: 'Prompt & temperature', knowledge: 'Knowledge rules', rewrite: 'Query rewrite prompt' };

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
  knowledgePrompt: string;
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
    knowledgePrompt: c.knowledge_prompt ?? c.knowledge_prompt_default,
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
  const [notes, setNotes] = useState<Record<Section, string>>({ language: '', idle: '', prompt: '', knowledge: '', rewrite: '' });
  const [saving, setSaving] = useState<Section | null>(null);
  const [preview, setPreview] = useState<{ prompt: string; error: string | null }>({ prompt: '', error: null });
  const [open, setOpen] = useState<number | null>(null);

  const load = async () => {
    const [c, h] = await Promise.all([api.config(), api.configHistory()]);
    setConfig(c);
    setHistory(h);
    setDraft(draftFrom(c));
  };
  useEffect(() => { load().catch((e) => toast.show({ tone: 'error', title: 'Could not load config', body: String(e) })); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Live preview of what the agent will receive for the chosen language and prompt.
  useEffect(() => {
    if (!draft?.language) return;
    const id = window.setTimeout(() => {
      api.previewPrompt(draft.language, draft.prompt.trim() || null, draft.knowledgePrompt.trim() || null)
        .then((r) => setPreview({ prompt: r.prompt, error: null }))
        .catch((e) => setPreview((p) => ({ ...p, error: (e as Error).message })));
    }, 250);
    return () => window.clearTimeout(id);
  }, [draft?.language, draft?.prompt, draft?.knowledgePrompt]);

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
    knowledge: draft.knowledgePrompt.trim() !== saved.knowledgePrompt.trim(),
    rewrite: draft.rewritePrompt.trim() !== saved.rewritePrompt.trim(),
  };
  const knowledgeIsDefault = draft.knowledgePrompt.trim() === config.knowledge_prompt_default.trim();
  const rewriteIsDefault = draft.rewritePrompt.trim() === config.rewrite_prompt_default.trim();

  /** Build the full version body: the saved values everywhere, except the section being saved. */
  const bodyFor = (section: Section) => {
    const d = { ...saved, ...(section === 'language' ? { language: draft.language, greeting: draft.greeting } : {}),
      ...(section === 'idle' ? { idleSecs: draft.idleSecs, idleCheck: draft.idleCheck, idleBye: draft.idleBye } : {}),
      ...(section === 'prompt' ? { prompt: draft.prompt, useTemp: draft.useTemp, temp: draft.temp } : {}),
      ...(section === 'knowledge' ? { knowledgePrompt: draft.knowledgePrompt } : {}) };
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
      knowledge_prompt: !d.knowledgePrompt.trim() || d.knowledgePrompt.trim() === config.knowledge_prompt_default.trim() ? null : d.knowledgePrompt.trim(),
      // only this section changes the rewrite prompt; "" (empty or equal to the built-in) = built-in
      rewrite_prompt: section !== 'rewrite' ? null : !draft.rewritePrompt.trim() || rewriteIsDefault ? '' : draft.rewritePrompt.trim(),
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
      knowledgePrompt: v.knowledge_prompt ?? config.knowledge_prompt_default,
      rewritePrompt: v.rewrite_prompt ?? config.rewrite_prompt_default,
    });
    setNotes({ language: `restored from #${v.id}`, idle: `restored from #${v.id}`, prompt: `restored from #${v.id}`, knowledge: `restored from #${v.id}`, rewrite: `restored from #${v.id}` });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const card = (section: Section, head: ReactNode, body: ReactNode) => (
    <form className="card" onSubmit={save(section)}>
      <div className="card__head">{head}</div>
      <div className="card__body">
        {body}
        <SaveBar dirty={dirty[section]} saving={saving === section} note={notes[section]} setNote={(v) => setNotes((n) => ({ ...n, [section]: v }))} label={SECTION_LABEL[section]} />
      </div>
    </form>
  );

  return (
    <>
      <Topbar
        title="Agent configuration"
        actions={<span className="pill pill--violet">{config.id ? `active version #${config.id}` : 'environment defaults'}</span>}
      />
      <div className="page">
        <div className="grid grid--main">
          <div className="stack">
            {card('language', <span className="card__label">Language & opening line</span>, (
              <>
                <div className="field">
                  <label className="field__label" htmlFor="language">Conversation language</label>
                  <select id="language" className="field__select lang-select" value={draft.language} onChange={(e) => set('language', e.target.value)}>
                    {config.languages.map((l) => <option key={l.code} value={l.code}>{l.native_name} — {l.english_name} ({l.code})</option>)}
                  </select>
                  <span className="field__hint">Used for speech recognition, speech synthesis and the prompt's language rules. Environment default: {langName(null)}.</span>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="greeting">Opening line <span className="muted">(optional, in {selected?.english_name ?? 'the selected language'})</span></label>
                  <input id="greeting" className="field__input lang-select" style={{ maxWidth: 'none' }} value={draft.greeting} onChange={(e) => set('greeting', e.target.value)} placeholder="e.g. வணக்கம், நான் ஆஷா. நான் உங்களுக்கு எப்படி உதவலாம்?" />
                  <span className="field__hint">Spoken instantly through TTS when the call connects. Leave empty to let the LLM greet.</span>
                </div>
              </>
            ))}

            {card('idle', (
              <>
                <span className="card__label">Idle handling</span>
                <span className="topbar__spacer" />
                <span className="card__note">silence after the agent speaks → one check → goodbye and hang up</span>
              </>
            ), (
              <>
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
                  <span className="field__hint">These are the exact lines the agent will speak. Edit them in place; "default" means the built-in line for the selected language.</span>
                </div>
              </>
            ))}

            {card('prompt', (
              <>
                <span className="card__label">System prompt (English, with placeholders) & temperature</span>
                <span className="topbar__spacer" />
                <button type="button" className="btn btn--quiet" onClick={() => set('prompt', config.default_prompt)}>Load built-in</button>
                <button type="button" className="btn btn--quiet" onClick={() => set('prompt', '')}>Use built-in (clear)</button>
              </>
            ), (
              <>
                <div className="field">
                  <textarea className="field__textarea" rows={14} value={draft.prompt} onChange={(e) => set('prompt', e.target.value)} placeholder={config.default_prompt} />
                  <span className="field__hint">
                    Write the prompt once in English. {'{language_english_name}'} and {'{language_native_name}'} are filled from the selected language, {'{agent_name}'} from AGENT_NAME. Empty = built-in prompt.
                  </span>
                </div>
                <div className="field">
                  <label className="field__label">Temperature</label>
                  <div className="range">
                    <label className="range__toggle"><input type="checkbox" checked={draft.useTemp} onChange={(e) => set('useTemp', e.target.checked)} />override</label>
                    <input type="range" min={0} max={2} step={0.05} value={draft.temp} disabled={!draft.useTemp} onChange={(e) => set('temp', Number(e.target.value))} />
                    <span className="range__value">{draft.useTemp ? draft.temp.toFixed(2) : 'default'}</span>
                  </div>
                  <span className="field__hint">Lower is more consistent, higher more varied. Unchecked keeps the provider default.</span>
                </div>
                {preview.error && <div className="banner banner--error">{preview.error}</div>}
              </>
            ))}

            {card('knowledge', (
              <>
                <span className="card__label">Knowledge-base rules (appended to the prompt)</span>
                <span className="topbar__spacer" />
                {knowledgeIsDefault ? <span className="pill pill--idle">default</span> : <button type="button" className="btn btn--quiet" onClick={() => set('knowledgePrompt', config.knowledge_prompt_default)}>reset to default</button>}
                <span className={`pill ${config.knowledge_on ? 'pill--up' : 'pill--warn'}`}>{config.knowledge_on ? 'search tool on' : 'search tool off'}</span>
              </>
            ), (
              <div className="field">
                <textarea className="field__textarea" rows={9} value={draft.knowledgePrompt} onChange={(e) => set('knowledgePrompt', e.target.value)} />
                <span className="field__hint">
                  Tells the model when to call search_knowledge_base and how to answer from the results. Appended after the system prompt only when the search tool is available; the preview on the right shows the combined text.
                </span>
              </div>
            ))}

            {config.query_rewrite_on && card('rewrite', (
              <>
                <span className="card__label">Query rewrite prompt (knowledge search)</span>
                <span className="topbar__spacer" />
                {rewriteIsDefault ? <span className="pill pill--idle">default</span> : <button type="button" className="btn btn--quiet" onClick={() => set('rewritePrompt', config.rewrite_prompt_default)}>reset to default</button>}
              </>
            ), (
              <div className="field">
                <textarea className="field__textarea" rows={12} value={draft.rewritePrompt} onChange={(e) => set('rewritePrompt', e.target.value)} />
                <span className="field__hint">
                  Instructions for the rewrite step before each knowledge search. It receives only the caller's last 10 messages (oldest first) and must reply with one English search query per line, one for each question in the latest message. Shown because KNOWLEDGE_QUERY_REWRITE is on.
                </span>
              </div>
            ))}
          </div>

          <div className="stack">
            <section className="card">
              <div className="card__head">
                <span className="card__label">What the agent will receive</span>
                <span className="topbar__spacer" />
                {selected && <span className="pill pill--violet">{selected.native_name} · {selected.code}</span>}
              </div>
              <div className="card__body">
                {preview.error ? <div className="banner banner--error">{preview.error}</div> : <pre className="preview-prompt">{preview.prompt}</pre>}
              </div>
            </section>

            <section className="card">
              <div className="card__head"><span className="card__label">History</span></div>
              {history.length === 0 ? (
                <div className="card__body"><p className="card__note">No versions saved yet.</p></div>
              ) : (
                <div className="list">
                  {history.map((v) => (
                    <div className="list__row" key={v.id}>
                      <button type="button" className="list__open" onClick={() => setOpen(open === v.id ? null : v.id)}>
                        <div className="list__name">#{v.id} {v.id === config.id && <span className="pill pill--up">active</span>} {v.note && <span className="card__note">· {v.note}</span>}</div>
                        <div className="list__meta">{when(v.created_at)} · {langName(v.language)} · temp {v.temperature ?? 'default'} · {v.system_prompt ? `${v.system_prompt.length} chars` : 'built-in prompt'}{v.greeting ? ' · fixed greeting' : ''}{v.idle_prompt_secs !== null ? ` · idle ${v.idle_prompt_secs}s` : ''}{v.knowledge_prompt ? ' · custom knowledge rules' : ''}{v.rewrite_prompt ? ' · custom rewrite prompt' : ''}{v.pronunciations?.length ? ` · ${v.pronunciations.length} pronunciations` : ''}{[v.stt_model, v.llm_model, v.tts_model, v.tts_speaker, v.translate_model].some(Boolean) ? ` · models: ${[v.stt_model, v.llm_model, v.tts_model && v.tts_speaker ? `${v.tts_model}/${v.tts_speaker}` : v.tts_model || v.tts_speaker, v.translate_model].filter(Boolean).join(', ')}` : ''}</div>
                        {open === v.id && v.system_prompt && <pre className="preview-prompt" style={{ marginTop: 8 }}>{v.system_prompt}</pre>}
                      </button>
                      <button type="button" className="btn btn--ghost btn--sm" onClick={() => restore(v)}>Restore</button>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
