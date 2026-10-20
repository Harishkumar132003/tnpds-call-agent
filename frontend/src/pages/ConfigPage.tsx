import { useEffect, useState, type FormEvent } from 'react';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import type { AgentConfig, ConfigVersion } from '../types';

export function ConfigPage() {
  const toast = useToast();
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [history, setHistory] = useState<ConfigVersion[]>([]);
  const [language, setLanguage] = useState('');
  const [prompt, setPrompt] = useState('');
  const [greeting, setGreeting] = useState('');
  const [useTemp, setUseTemp] = useState(false);
  const [temp, setTemp] = useState(0.7);
  const [note, setNote] = useState('');
  const [preview, setPreview] = useState<{ prompt: string; error: string | null }>({ prompt: '', error: null });
  const [saving, setSaving] = useState(false);
  const [open, setOpen] = useState<number | null>(null);

  const load = async () => {
    const [c, h] = await Promise.all([api.config(), api.configHistory()]);
    setConfig(c);
    setHistory(h);
    setLanguage(c.language ?? c.default_language);
    setPrompt(c.system_prompt ?? '');
    setGreeting(c.greeting ?? '');
    setUseTemp(c.temperature !== null);
    setTemp(c.temperature ?? 0.7);
  };
  useEffect(() => { load().catch((e) => toast.show({ tone: 'error', title: 'Could not load config', body: String(e) })); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Live preview of what the agent will actually receive for the chosen language.
  useEffect(() => {
    if (!language) return;
    const id = window.setTimeout(() => {
      api.previewPrompt(language, prompt.trim() || null)
        .then((r) => setPreview({ prompt: r.prompt, error: null }))
        .catch((e) => setPreview((p) => ({ ...p, error: (e as Error).message })));
    }, 250);
    return () => window.clearTimeout(id);
  }, [language, prompt]);

  if (!config) return <div className="loading">Loading config…</div>;

  const savedLanguage = config.language ?? config.default_language;
  const dirty =
    (prompt.trim() || null) !== (config.system_prompt ?? null) ||
    (useTemp ? temp : null) !== config.temperature ||
    (greeting.trim() || null) !== (config.greeting ?? null) ||
    language !== savedLanguage;
  const selected = config.languages.find((l) => l.code === language);

  const save = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const id = await api.saveConfig({
        system_prompt: prompt.trim() || null,
        temperature: useTemp ? temp : null,
        language: language === config.default_language && !config.language ? null : language,
        greeting: greeting.trim() || null,
        note: note.trim() || null,
      });
      toast.show({ tone: 'success', title: `Saved version #${id}`, body: `New sessions speak ${selected?.english_name ?? language} immediately; calls in progress keep theirs.` });
      setNote('');
      await load();
    } catch (err) {
      toast.show({ tone: 'error', title: 'Not saved', body: (err as Error).message });
    } finally {
      setSaving(false);
    }
  };

  const restore = (v: ConfigVersion) => {
    setPrompt(v.system_prompt ?? '');
    setUseTemp(v.temperature !== null);
    setTemp(v.temperature ?? 0.7);
    setLanguage(v.language ?? config.default_language);
    setGreeting(v.greeting ?? '');
    setNote(`restored from #${v.id}`);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const langName = (code: string | null) => {
    const l = config.languages.find((x) => x.code === (code ?? config.default_language));
    return l ? `${l.english_name}` : code ?? 'default';
  };

  return (
    <>
      <Topbar
        title="Agent configuration"
        actions={<span className="pill pill--violet">{config.id ? `active version #${config.id}` : 'environment defaults'}</span>}
      />
      <div className="page">
        <div className="grid grid--main">
          <form className="stack" onSubmit={save}>
            <section className="card">
              <div className="card__head"><span className="card__label">Language</span></div>
              <div className="card__body">
                <div className="field">
                  <label className="field__label" htmlFor="language">Conversation language</label>
                  <select
                    id="language"
                    className="field__select lang-select"
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                  >
                    {config.languages.map((l) => (
                      <option key={l.code} value={l.code}>
                        {l.native_name} — {l.english_name} ({l.code})
                      </option>
                    ))}
                  </select>
                </div>
                <span className="field__hint">Used for speech recognition, speech synthesis and the prompt's language rules. Environment default: {langName(null)}.</span>
                <div className="field" style={{ marginTop: 16 }}>
                  <label className="field__label" htmlFor="greeting">Opening line <span className="muted">(optional, in {selected?.english_name ?? 'the selected language'})</span></label>
                  <input id="greeting" className="field__input lang-select" value={greeting} onChange={(e) => setGreeting(e.target.value)} placeholder="e.g. வணக்கம், நான் ஆஷா. நான் உங்களுக்கு எப்படி உதவலாம்?" />
                  <span className="field__hint">Spoken instantly through TTS the moment the call connects, instead of waiting one to two seconds for the LLM to compose a greeting. Leave empty to let the LLM greet.</span>
                </div>
              </div>
            </section>

            <section className="card">
              <div className="card__head">
                <span className="card__label">System prompt (English, with placeholders)</span>
                <span className="topbar__spacer" />
                <button type="button" className="btn btn--quiet" onClick={() => setPrompt(config.default_prompt)}>Load built-in</button>
                <button type="button" className="btn btn--quiet" onClick={() => setPrompt('')}>Use built-in (clear)</button>
              </div>
              <div className="card__body">
                <div className="field">
                  <textarea className="field__textarea" rows={14} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={config.default_prompt} />
                  <span className="field__hint">
                    Write the prompt once in English. {'{language_english_name}'} and {'{language_native_name}'} are filled from the selected language, {'{agent_name}'} from AGENT_NAME. Empty = built-in prompt.
                  </span>
                </div>
                <div className="field">
                  <label className="field__label">Temperature</label>
                  <div className="range">
                    <label className="range__toggle"><input type="checkbox" checked={useTemp} onChange={(e) => setUseTemp(e.target.checked)} />override</label>
                    <input type="range" min={0} max={2} step={0.05} value={temp} disabled={!useTemp} onChange={(e) => setTemp(Number(e.target.value))} />
                    <span className="range__value">{useTemp ? temp.toFixed(2) : 'default'}</span>
                  </div>
                  <span className="field__hint">Lower is more consistent, higher more varied. Unchecked keeps the provider default.</span>
                </div>
                <div className="field">
                  <label className="field__label">Change note</label>
                  <input className="field__input" value={note} onChange={(e) => setNote(e.target.value)} placeholder="why this change?" />
                </div>
                <div className="form-actions">
                  <button type="submit" className="btn btn--primary" disabled={saving || !dirty || Boolean(preview.error)}>{saving ? 'Saving…' : 'Save new version'}</button>
                  {dirty && <span className="card__note">unsaved changes</span>}
                </div>
              </div>
            </section>
          </form>

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
                        <div className="list__meta">{when(v.created_at)} · {langName(v.language)} · temp {v.temperature ?? 'default'} · {v.system_prompt ? `${v.system_prompt.length} chars` : 'built-in prompt'}{v.greeting ? ' · fixed greeting' : ''}</div>
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
