import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Icon } from '../components/Icon';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import type { AgentConfig, ModelCatalogue, ModelOption, VoiceOption } from '../types';

interface Choice {
  stt: string;
  llm: string;
  tts: string;
  voice: string;
  translate: string;
  embedding: string;
}

/** The effective choice: the saved override, else the environment default. */
const choiceFrom = (c: AgentConfig): Choice => ({
  stt: c.stt_model ?? c.model_defaults.stt,
  llm: c.llm_model ?? c.model_defaults.llm,
  tts: c.tts_model ?? c.model_defaults.tts,
  voice: c.tts_speaker ?? c.model_defaults.tts_speaker,
  translate: c.translate_model ?? c.model_defaults.translate,
  embedding: c.embedding_model ?? c.model_defaults.embedding,
});

/** One selectable model card. */
function OptionCard({ on, title, id, note, tag, onPick }: { on: boolean; title: string; id?: string; note: string; tag?: string; onPick: () => void }) {
  return (
    <button type="button" role="radio" aria-checked={on} className={`opt${on ? ' is-on' : ''}`} onClick={onPick}>
      <span className="opt__radio" aria-hidden="true" />
      <span className="opt__body">
        <span className="opt__head">
          <span className="opt__title">{title}</span>
          {tag && <span className="opt__tag">{tag}</span>}
        </span>
        {id && id !== title && <span className="opt__id">{id}</span>}
        <span className="opt__note">{note}</span>
      </span>
    </button>
  );
}

/** A step's heading, options and hint. */
function Step({ title, isDefault, extra, hint, children }: { title: string; isDefault: boolean; extra?: ReactNode; hint: ReactNode; children: ReactNode }) {
  return (
    <section className="mstep">
      <div className="mstep__head">
        <span className="mstep__title">{title}</span>
        {isDefault && <span className="tagx">default</span>}
        {extra}
      </div>
      {children}
      <span className="field__hint">{hint}</span>
    </section>
  );
}

/** Pick the Sarvam model for each pipeline step, limited to what supports the agent's language. */
export function ModelsPage() {
  const toast = useToast();
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [models, setModels] = useState<ModelCatalogue | null>(null);
  const [draft, setDraft] = useState<Choice | null>(null);
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const [docCount, setDocCount] = useState<number | null>(null);
  const [allVoices, setAllVoices] = useState(false);

  const load = async () => {
    const c = await api.config();
    setConfig(c);
    setDraft(choiceFrom(c));
    setModels(await api.models(c.language ?? c.default_language));
    api.documents().then((d) => setDocCount(d.length)).catch(() => setDocCount(null));
  };
  useEffect(() => { load().catch((e) => toast.show({ tone: 'error', title: 'Could not load models', body: String(e) })); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (!config || !draft) return <Topbar title="Models" />;

  const language = config.language ?? config.default_language;
  const langName = config.languages.find((l) => l.code === language)?.english_name ?? language;
  const saved = choiceFrom(config);
  const md = config.model_defaults;
  const dirty = (Object.keys(draft) as (keyof Choice)[]).some((k) => draft[k] !== saved[k]);
  const embeddingChanged = draft.embedding !== saved.embedding;
  const set = <K extends keyof Choice>(key: K, value: Choice[K]) => setDraft((d) => (d ? { ...d, [key]: value } : d));

  const voices = models?.voices[draft.tts] ?? [];
  const picked = voices.find((v) => v.id === draft.voice);
  const recommended = voices.filter((v) => v.recommended);
  const others = voices.filter((v) => !v.recommended);
  const showOthers = allVoices || (picked ? !picked.recommended : false);

  const pickTts = (id: string) => {
    set('tts', id);
    const list = models?.voices[id] ?? [];
    if (!list.some((v) => v.id === draft.voice)) set('voice', (list.find((v) => v.recommended) ?? list[0])?.id ?? draft.voice);
  };

  const save = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      // a new version with every other setting as stored; equal to the .env default is stored as null
      const r = await api.saveConfig({
        system_prompt: config.system_prompt,
        temperature: config.temperature,
        language: config.language,
        greeting: config.greeting,
        idle_prompt_secs: config.idle_prompt_secs,
        idle_check_line: config.idle_check_line,
        idle_goodbye_line: config.idle_goodbye_line,
        knowledge_prompt: config.knowledge_prompt,
        pronunciations: null,
        stt_model: draft.stt === md.stt ? null : draft.stt,
        llm_model: draft.llm === md.llm ? null : draft.llm,
        tts_model: draft.tts === md.tts ? null : draft.tts,
        tts_speaker: draft.voice === md.tts_speaker ? null : draft.voice,
        translate_model: draft.translate === md.translate ? null : draft.translate,
        embedding_model: draft.embedding === md.embedding ? null : draft.embedding,
        note: note.trim() || `Models: ${draft.stt}, ${draft.llm}, ${draft.tts}/${draft.voice}, ${draft.translate}, ${draft.embedding}`,
      });
      toast.show({ tone: 'success', title: `Saved version #${r.id}`, body: r.reindexed !== null && r.reindexed !== undefined ? `Re-embedded ${r.reindexed} documents with ${draft.embedding}${r.needs_reupload ? `; ${r.needs_reupload} must be uploaded again (see Knowledge)` : ''}. New calls use these models.` : 'New calls use these models. Calls already in progress keep theirs.' });
      setNote('');
      await load();
    } catch (err) {
      toast.show({ tone: 'error', title: 'Not saved', body: (err as Error).message });
    } finally {
      setSaving(false);
    }
  };

  const options = (key: Exclude<keyof Choice, 'voice'>, list: ModelOption[], onPick?: (id: string) => void) => {
    const value = draft[key];
    const known = list.some((m) => m.id === value);
    return (
      <div className="opts" role="radiogroup">
        {!known && <OptionCard on title={value} note="not in the catalogue any more; pick another" onPick={() => undefined} />}
        {list.map((m) => <OptionCard key={m.id} on={m.id === value} title={m.label} id={m.id} note={m.note} tag={m.recommended ? 'recommended' : undefined} onPick={() => (onPick ?? ((id) => set(key, id)))(m.id)} />)}
      </div>
    );
  };

  const voiceChip = (v: VoiceOption) => (
    <button type="button" role="radio" aria-checked={v.id === draft.voice} key={v.id} className={`voice${v.id === draft.voice ? ' is-on' : ''}`} onClick={() => set('voice', v.id)} title={`${v.gender}${v.tier < 3 ? `, Sarvam tier ${v.tier}` : ''}`}>
      <span className="voice__name">{v.style ? v.id.split('_')[0] : v.id}</span>
      <span className="voice__meta">{v.style ? v.style : `${v.gender === 'female' ? 'F' : 'M'}${v.tier < 3 ? ` · T${v.tier}` : ''}`}</span>
    </button>
  );

  return (
    <>
      <Topbar
        title="Models"
        actions={(
          <>
            <span className="pill pill--info">{langName}</span>
            <span className="pill pill--idle">active version #{config.id ?? '—'}</span>
          </>
        )}
      />
      <div className="page stack">
        <div className="grid grid--main">
          <form className="card" onSubmit={save} style={{ alignSelf: 'start' }}>
            <div className="card__head">
              <span className="card__label">Sarvam model for each step</span>
              <span className="topbar__spacer" />
              <span className="card__note">{models ? `only models that support ${langName}` : 'loading catalogue…'}</span>
            </div>
            <div className="card__body stack-steps">
              <Step title="Speech to text" isDefault={draft.stt === md.stt} hint="Transcribes the caller over Sarvam's streaming websocket.">
                {options('stt', models?.stt ?? [])}
              </Step>
              <Step title="Language model" isDefault={draft.llm === md.llm} hint="Writes the replies and decides when to search the knowledge base. Sarvam's own models only.">
                {options('llm', models?.llm ?? [])}
              </Step>
              <Step title="Text to speech" isDefault={draft.tts === md.tts} hint={`Speaks the replies. Only the listed models serve ${langName} over the websocket.`}>
                {options('tts', models?.tts ?? [], pickTts)}
              </Step>
              <Step
                title="Voice"
                isDefault={draft.voice === md.tts_speaker}
                extra={picked?.recommended ? <span className="tagx tagx--on">recommended for {langName}</span> : null}
                hint={draft.tts === 'bulbul:v4-flash'
                  ? <>v4 Flash voices are recorded for one language and one style (narration, ads, customer...). Listen to one on the <Link className="link" to="/pronunciation">Pronunciation</Link> page after saving.</>
                  : <>F / M is the voice's gender; T1 and T2 are Sarvam's production tiers by error rate (T1 best). Every voice speaks all 11 languages. Listen to one on the <Link className="link" to="/pronunciation">Pronunciation</Link> page after saving.</>}
              >
                {!voices.some((v) => v.id === draft.voice) && <div className="banner banner--warn">Saved voice "{draft.voice}" is not in the catalogue any more; pick another.</div>}
                {recommended.length > 0 ? (
                  <>
                    <div className="voices" role="radiogroup">
                      <span className="voices__group">Recommended by Sarvam for {langName}</span>
                      {recommended.map(voiceChip)}
                    </div>
                    <div className="voices" role="radiogroup" style={{ marginTop: 8 }}>
                      <button type="button" className="voices__group voices__toggle" onClick={() => setAllVoices((v) => !v)}>
                        <Icon name="chevron" size={12} className={showOthers ? 'is-open' : ''} /> All {others.length} other voices
                      </button>
                      {showOthers && others.map(voiceChip)}
                    </div>
                  </>
                ) : (
                  <div className="voices" role="radiogroup">
                    <span className="voices__group">{voices.length} {langName} voices, each with one persona style</span>
                    {others.map(voiceChip)}
                  </div>
                )}
              </Step>
              <Step title="Translation" isDefault={draft.translate === md.translate} hint="Turns the caller's question into English before searching the knowledge base.">
                {options('translate', models?.translate ?? [])}
              </Step>
              <Step
                title="Embedding (OpenAI)"
                isDefault={draft.embedding === md.embedding}
                extra={embeddingChanged ? <span className="tagx tagx--on">re-index on save</span> : null}
                hint={embeddingChanged
                  ? `Changing this re-embeds every indexed document${docCount !== null ? ` (${docCount})` : ''} when you save, which can take a few minutes. Documents uploaded before the portal kept their text cannot be re-embedded and will ask to be uploaded again.`
                  : 'Turns documents and questions into vectors for the search. Every document in the knowledge base is embedded with this model.'}
              >
                {options('embedding', models?.embedding ?? [])}
              </Step>
            </div>
            <div className="card__foot form-actions">
              <button type="submit" className="btn btn--primary" disabled={saving || !dirty}>{saving ? (embeddingChanged ? 'Re-indexing…' : 'Saving…') : embeddingChanged ? 'Save and re-index documents' : 'Save models'}</button>
              <input className="field__input" style={{ maxWidth: 320 }} value={note} onChange={(e) => setNote(e.target.value)} placeholder="change note (optional)" />
              {dirty && <span className="card__note">unsaved changes</span>}
            </div>
          </form>

          <div className="stack">
            <section className="card">
              <div className="card__head"><span className="card__label">In effect now</span></div>
              <div className="card__body">
                <div className="facts facts--tight">
                  <div className="fact"><div className="fact__label">Speech to text</div><div className="fact__value mono">{saved.stt}</div></div>
                  <div className="fact"><div className="fact__label">Language model</div><div className="fact__value mono">{saved.llm}</div></div>
                  <div className="fact"><div className="fact__label">Text to speech</div><div className="fact__value mono">{saved.tts} / {saved.voice}</div></div>
                  <div className="fact"><div className="fact__label">Translation</div><div className="fact__value mono">{saved.translate}</div></div>
                  <div className="fact"><div className="fact__label">Embedding</div><div className="fact__value mono">{saved.embedding}</div></div>
                </div>
              </div>
            </section>
            <section className="card">
              <div className="card__head"><span className="card__label">How it works</span></div>
              <div className="card__body">
                <ul className="hint-list">
                  <li>The lists depend on the conversation language set on the <Link className="link" to="/config">Configuration</Link> page: a model or voice that cannot handle {langName} is not offered.</li>
                  <li>A choice equal to the environment default is stored as "default", so it keeps following .env on the server.</li>
                  <li>Saving creates a new config version; everything else in it stays as it is. If a stored choice ever stops fitting the language, the agent falls back to the environment default and logs a warning.</li>
                </ul>
              </div>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
