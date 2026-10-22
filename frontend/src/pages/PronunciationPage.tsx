import { useEffect, useState, type FormEvent } from 'react';
import { Empty } from '../components/Empty';
import { Icon } from '../components/Icon';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import type { AgentConfig } from '../types';

const MAX_ENTRIES = 100;

interface Row {
  key: number;
  word: string;
  spoken: string;
}

let nextKey = 1;
const row = (word = '', spoken = ''): Row => ({ key: nextKey++, word, spoken });

/** "word = spoken" lines from the API → editable rows. */
function rowsFrom(text: string): Row[] {
  return text
    .split('\n')
    .filter((l) => l.includes('='))
    .map((l) => {
      const i = l.indexOf('=');
      return row(l.slice(0, i).trim(), l.slice(i + 1).trim());
    });
}

/** Rows → the "word = spoken" lines the API stores. Blank rows are dropped. */
const linesFrom = (rows: Row[]) =>
  rows
    .filter((r) => r.word.trim() || r.spoken.trim())
    .map((r) => `${r.word.trim()} = ${r.spoken.trim()}`)
    .join('\n');

export function PronunciationPage() {
  const toast = useToast();
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [saved, setSaved] = useState('');
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const [testText, setTestText] = useState('TNPDS வெப்சைட்ல போய் OTP போடுங்க.');
  const [testing, setTesting] = useState(false);
  const [testAudio, setTestAudio] = useState<string | null>(null);

  const load = async () => {
    const c = await api.config();
    setConfig(c);
    setSaved(c.pronunciations ?? '');
    const r = rowsFrom(c.pronunciations ?? '');
    setRows(r.length ? r : [row()]);
  };
  useEffect(() => { load().catch((e) => toast.show({ tone: 'error', title: 'Could not load pronunciation', body: String(e) })); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (!config) return <Topbar title="Pronunciation" />;

  const language = config.language ?? config.default_language;
  const langName = config.languages.find((l) => l.code === language)?.english_name ?? language;
  const lines = linesFrom(rows);
  const dirty = lines !== saved.trim();
  const filled = rows.filter((r) => r.word.trim() || r.spoken.trim());
  const incomplete = filled.filter((r) => !r.word.trim() || !r.spoken.trim());
  const words = filled.map((r) => r.word.trim());
  const duplicates = new Set(words.filter((w, i) => w && words.indexOf(w) !== i));
  const problem = incomplete.length
    ? 'Every entry needs both the word and how it should be spoken.'
    : filled.length > MAX_ENTRIES
      ? `Sarvam allows at most ${MAX_ENTRIES} entries.`
      : duplicates.size
        ? `Duplicate word: ${[...duplicates].join(', ')}`
        : null;

  const update = (key: number, patch: Partial<Row>) => setRows((rs) => rs.map((r) => (r.key === key ? { ...r, ...patch } : r)));
  const remove = (key: number) => setRows((rs) => (rs.length === 1 ? [row()] : rs.filter((r) => r.key !== key)));
  const add = () => setRows((rs) => [...rs, row()]);

  const save = async (e: FormEvent) => {
    e.preventDefault();
    if (problem) return;
    setSaving(true);
    try {
      // a new config version with every other setting as stored; only the dictionary changes
      const { id } = await api.saveConfig({
        system_prompt: config.system_prompt,
        temperature: config.temperature,
        language: config.language,
        greeting: config.greeting,
        idle_prompt_secs: config.idle_prompt_secs,
        idle_check_line: config.idle_check_line,
        idle_goodbye_line: config.idle_goodbye_line,
        knowledge_prompt: config.knowledge_prompt,
        pronunciations: lines,
        // models are managed on the Models page; keep the active version's choices
        stt_model: config.stt_model,
        llm_model: config.llm_model,
        tts_model: config.tts_model,
        tts_speaker: config.tts_speaker,
        translate_model: config.translate_model,
        embedding_model: config.embedding_model,
        note: note.trim() || `Pronunciation updated (${filled.length} entries)`,
      });
      toast.show({ tone: 'success', title: `Saved version #${id}`, body: filled.length ? `Sarvam dictionary rebuilt with ${filled.length} entries. New calls use it.` : 'Dictionary removed. New calls speak the default way.' });
      setNote('');
      await load();
    } catch (err) {
      toast.show({ tone: 'error', title: 'Not saved', body: (err as Error).message });
    } finally {
      setSaving(false);
    }
  };

  const play = async () => {
    setTesting(true);
    setTestAudio(null);
    try {
      const r = await api.pronunciationTest(testText, dirty ? lines : null);
      setTestAudio(`data:audio/wav;base64,${r.audio_base64}`);
    } catch (e) {
      toast.show({ tone: 'error', title: 'Could not synthesise', body: (e as Error).message });
    } finally {
      setTesting(false);
    }
  };

  return (
    <>
      <Topbar
        title="Pronunciation"
        actions={(
          <>
            <span className="pill pill--info">{langName}</span>
            <span className={`pill ${config.tts_dict_id ? 'pill--up' : 'pill--warn'}`}>{config.tts_dict_id ? <>dictionary <span className="mono">{config.tts_dict_id}</span></> : 'no dictionary'}</span>
          </>
        )}
      />
      <div className="page stack">
        <div className="grid grid--main">
          <form className="card" onSubmit={save} style={{ alignSelf: 'start' }}>
            <div className="card__head">
              <span className="card__label">Entries</span>
              <span className="topbar__spacer" />
              <span className="card__note">{filled.length} / {MAX_ENTRIES}</span>
            </div>
            <div className="card__body">
              <div className="pron">
                <div className="pron__row pron__row--head">
                  <span className="field__label">Written as</span>
                  <span />
                  <span className="field__label">Spoken as ({langName})</span>
                  <span />
                </div>
                {rows.map((r, i) => (
                  <div className="pron__row" key={r.key}>
                    <input
                      className="field__input lang-select"
                      value={r.word}
                      onChange={(e) => update(r.key, { word: e.target.value })}
                      placeholder={i === 0 ? 'TNPDS' : 'word'}
                      aria-label="Written form"
                      spellCheck={false}
                    />
                    <span className="pron__eq mono">=</span>
                    <input
                      className="field__input lang-select"
                      value={r.spoken}
                      onChange={(e) => update(r.key, { spoken: e.target.value })}
                      placeholder={i === 0 ? 'டி என் பி டி எஸ்' : 'how it should be spoken'}
                      aria-label="Spoken form"
                      spellCheck={false}
                    />
                    <button type="button" className="btn btn--icon" title="Remove entry" onClick={() => remove(r.key)}><Icon name="close" size={14} /></button>
                  </div>
                ))}
              </div>
              <div className="row" style={{ marginTop: 'var(--s3)' }}>
                <button type="button" className="btn btn--ghost btn--sm" onClick={add} disabled={rows.length >= MAX_ENTRIES}><Icon name="plus" size={14} /> Add entry</button>
                <span className="field__hint">Exact matches only, plain text. Case matters: TNPDS and tnpds are two entries.</span>
              </div>
            </div>
            <div className="card__foot form-actions">
              <button type="submit" className="btn btn--primary" disabled={saving || !dirty || !!problem}>{saving ? 'Saving…' : 'Save pronunciation'}</button>
              <input className="field__input" style={{ maxWidth: 320 }} value={note} onChange={(e) => setNote(e.target.value)} placeholder="change note (optional)" />
              {problem ? <span className="card__note" style={{ color: 'var(--red)' }}>{problem}</span> : dirty ? <span className="card__note">unsaved changes</span> : null}
            </div>
          </form>

          <div className="stack">
            <section className="card">
              <div className="card__head"><span className="card__label">Listen</span></div>
              <div className="card__body">
                <div className="field">
                  <label className="field__label" htmlFor="pron-test">Sample sentence</label>
                  <textarea id="pron-test" className="field__textarea lang-select" rows={3} value={testText} onChange={(e) => setTestText(e.target.value)} style={{ minHeight: 0, maxWidth: 'none' }} />
                  <span className="field__hint">{dirty ? 'Plays with the entries as typed above, through a temporary dictionary.' : 'Plays with the saved dictionary, in the agent\u2019s voice.'}</span>
                </div>
                <button type="button" className="btn btn--solid" disabled={testing || !testText.trim()} onClick={play}>{testing ? 'Synthesising…' : 'Play'}</button>
                {testAudio && <audio controls autoPlay src={testAudio} style={{ marginTop: 'var(--s3)', width: '100%' }} />}
              </div>
            </section>

            <section className="card">
              <div className="card__head"><span className="card__label">How it works</span></div>
              <div className="card__body">
                {filled.length === 0 && !dirty ? (
                  <Empty icon="sparkle" title="No entries yet" body="Add words the voice mispronounces: acronyms, site names, numbers to read digit by digit." />
                ) : null}
                <ul className="hint-list">
                  <li>Saving creates a Sarvam pronunciation dictionary for {langName} and deletes the previous one. Calls that start after the save use it.</li>
                  <li>The spoken form is plain text in the language, not a phonetic alphabet: write it the way the voice should say it.</li>
                  <li>The dictionary belongs to the conversation language. Changing the language on the configuration page needs a save here too.</li>
                </ul>
              </div>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}
