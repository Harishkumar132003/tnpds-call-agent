import { useEffect, useState, type FormEvent } from 'react';
import { Empty } from '../components/Empty';
import { Link } from 'react-router-dom';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { SEARCH_TYPE_LABEL } from '../types';
import type { EvalQuestion, SearchType } from '../types';

const PAGE = 50;

export function EvalsPage() {
  const toast = useToast();
  const [questions, setQuestions] = useState<EvalQuestion[]>([]);
  const [untranslated, setUntranslated] = useState(0);
  const [translating, setTranslating] = useState(false);
  const [total, setTotal] = useState(0);
  const [runnable, setRunnable] = useState(0);
  const [page, setPage] = useState(0);
  const [running, setRunning] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [filter, setFilter] = useState('');
  const [busy, setBusy] = useState(false);
  const [showRun, setShowRun] = useState(false);
  const [runName, setRunName] = useState('');
  const [concurrency, setConcurrency] = useState(1);
  const [searchType, setSearchType] = useState<SearchType>('semantic');
  const [collection, setCollection] = useState('');
  const [onlyIds, setOnlyIds] = useState('');
  const [draft, setDraft] = useState({ question_en: '', question_ta: '', expected_source: '', note: '' });
  const [editing, setEditing] = useState<EvalQuestion | null>(null);
  const [showAdd, setShowAdd] = useState(false);

  const load = async () => {
    const [q, b] = await Promise.all([api.evalQuestions(page * PAGE, PAGE, filter.trim()), api.evalBatches()]);
    setQuestions(q.questions);
    setTotal(q.total);
    setRunnable(q.runnable);
    setUntranslated(q.untranslated);
    setTranslating(q.translating);
    setRunning(b.running);
  };
  useEffect(() => {
    load().catch((e) => toast.show({ tone: 'error', title: 'Could not load evaluation', body: String(e) }));
  }, [page, filter]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    api.evalSources().then((s) => setSources(s.sources)).catch(() => undefined);
  }, []);

  // refresh while a batch runs or questions are still being converted to Tamil
  const live = running !== null || translating;
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => { load().catch(() => undefined); }, 5_000);
    return () => window.clearInterval(id);
  }, [live]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const questionIds = onlyIds.split(/[\s,]+/).filter(Boolean).map(Number);
      if (questionIds.some((n) => !Number.isInteger(n) || n <= 0)) throw new Error('Question ids must be whole numbers separated by commas.');
      const batch = await api.evalRun(runName.trim() || null, concurrency, searchType, { collection: collection.trim() || null, questionIds: questionIds.length ? questionIds : null });
      toast.show({ tone: 'success', title: 'Evaluation started', body: `${batch.total} questions, ${concurrency} at a time, ${SEARCH_TYPE_LABEL[searchType]} search.` });
      setShowRun(false);
      setRunName('');
      setOnlyIds('');
      await load();
    } catch (e) {
      toast.show({ tone: 'error', title: 'Could not start', body: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const add = async (e: FormEvent) => {
    e.preventDefault();
    if (!draft.question_en.trim() || !draft.expected_source) return;
    setBusy(true);
    try {
      await api.evalAddQuestion({ question_en: draft.question_en.trim(), question_ta: draft.question_ta.trim() || null, expected_source: draft.expected_source, note: draft.note.trim() || null });
      setDraft({ question_en: '', question_ta: '', expected_source: draft.expected_source, note: '' });
      setShowAdd(false);
      await load();
    } catch (err) {
      toast.show({ tone: 'error', title: 'Could not add', body: (err as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const saveEdit = async (e: FormEvent) => {
    e.preventDefault();
    if (!editing) return;
    setBusy(true);
    try {
      await api.evalUpdateQuestion(editing.id, { question_en: editing.question_en, question_ta: editing.question_ta ?? '', expected_source: editing.expected_source, note: editing.note ?? '' });
      setEditing(null);
      await load();
    } catch (err) {
      toast.show({ tone: 'error', title: 'Could not save', body: (err as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const toggle = async (q: EvalQuestion) => {
    try {
      await api.evalUpdateQuestion(q.id, { enabled: !q.enabled });
      setQuestions((qs) => qs.map((x) => (x.id === q.id ? { ...x, enabled: !q.enabled } : x)));
    } catch (err) {
      toast.show({ tone: 'error', title: 'Could not update', body: (err as Error).message });
    }
  };

  const remove = async (q: EvalQuestion) => {
    if (!window.confirm(`Delete question #${q.id}?`)) return;
    try {
      await api.evalDeleteQuestion(q.id);
      await load();
    } catch (err) {
      toast.show({ tone: 'error', title: 'Delete failed', body: (err as Error).message });
    }
  };

  const shown = questions;
  const enabledWithTamil = runnable;
  const pages = Math.max(1, Math.ceil(total / PAGE));

  return (
    <>
      <Topbar
        title="Evaluation"
        onSearch={(v) => { setFilter(v); setPage(0); }}
        searchPlaceholder="Question, sheet, note…"
        actions={(
          <>
            {running && <Link className="pill pill--info" to={`/eval-results/${running}`}>batch running · view</Link>}
            {translating && <span className="pill pill--warn">writing Tamil… {untranslated} left</span>}
            <button type="button" className="btn btn--ghost btn--sm" disabled={busy} onClick={() => { setEditing(null); setShowAdd(true); }}>Add question</button>
            <button type="button" className="btn btn--primary btn--sm" disabled={busy || running !== null || enabledWithTamil === 0} onClick={() => setShowRun((v) => !v)}>
              {running ? 'Running…' : `Run evaluation (${enabledWithTamil})`}
            </button>
          </>
        )}
      />
      <div className="page stack">
        {untranslated > 0 && !translating && (
          <div className="banner banner--warn">
            {untranslated} question{untranslated === 1 ? '' : 's'} have no Tamil text yet and will be skipped by a run.{' '}
            <button type="button" className="btn btn--ghost btn--sm" onClick={() => api.evalTranslateMissing().then(load).catch((e) => toast.show({ tone: 'error', title: 'Could not start', body: String(e) }))}>Write Tamil now</button>
          </div>
        )}

        {showRun && !running && (
          <div className="scrim" onClick={() => setShowRun(false)}>
            <form className="modal" onSubmit={run} onClick={(e) => e.stopPropagation()}>
              <div className="modal__head">
                <span className="card__label">Start an evaluation run</span>
                <span className="topbar__spacer" />
                <button type="button" className="btn btn--ghost btn--sm" onClick={() => setShowRun(false)}>Close</button>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="run-name">Batch name</label>
                <input id="run-name" className="field__input" autoFocus value={runName} onChange={(e) => setRunName(e.target.value)} placeholder="e.g. v2 collection, config 41" />
              </div>
              <div className="field">
                <label className="field__label" htmlFor="run-conc">Questions at a time</label>
                <select id="run-conc" className="field__select" value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))}>
                  <option value={1}>1 · like one caller (safest)</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                  <option value={4}>4 · likely to hit the Sarvam rate limit</option>
                </select>
                <span className="field__hint">The agent's Sarvam key is shared with live calls. A 429 or a timeout from Sarvam is retried up to three times with backoff.</span>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="run-search">Search</label>
                <select id="run-search" className="field__select" value={searchType} onChange={(e) => setSearchType(e.target.value as SearchType)}>
                  <option value="semantic">Semantic · embeddings, as calls do</option>
                  <option value="keyword">Keyword · BM25 word match</option>
                  <option value="hybrid">Hybrid · both, merged by rank (RRF)</option>
                </select>
                <span className="field__hint">
                  {searchType === 'semantic'
                    ? 'A sheet is sent to the answer step only when its similarity reaches the configured minimum score.'
                    : 'Keyword and hybrid scores are on their own scales, so every retrieved sheet is sent to the answer step, up to the top-k and the character cap.'}
                </span>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="run-collection">Qdrant collection</label>
                <input id="run-collection" className="field__input mono" value={collection} onChange={(e) => setCollection(e.target.value)} placeholder="blank · the configured collection" />
                <span className="field__hint">Search a test copy of the knowledge base instead of the live one. Calls are not affected.</span>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="run-ids">Only these question ids</label>
                <input id="run-ids" className="field__input mono" value={onlyIds} onChange={(e) => setOnlyIds(e.target.value)} placeholder="blank · every runnable question, e.g. 52, 61, 172" />
              </div>
              <div className="modal__foot">
                <span className="field__hint">{onlyIds.trim() ? 'Only the listed questions' : `${enabledWithTamil} enabled questions with Tamil text`} will run, each in a brand-new session. The decide step is skipped: every question is searched as said, then answered.</span>
                <button type="submit" className="btn btn--primary" disabled={busy}>{busy ? 'Starting…' : 'Run'}</button>
              </div>
            </form>
          </div>
        )}

        <section className="card">
            <div className="card__head">
              <span className="card__label">Questions</span>
              <span className="topbar__spacer" />
              <span className="card__note">{total} {filter.trim() ? 'matching' : 'total'} · {enabledWithTamil} runnable · page {page + 1} of {pages}</span>
            </div>
            {questions.length === 0 ? (
              <Empty icon="chat" title={filter.trim() ? 'No matching questions' : 'No questions yet'} body={filter.trim() ? 'Try another search.' : "Import the workbook's example questions or add one with the button above."} />
            ) : (
              <div className="table-wrap">
                <table className="table table--left">
                  <thead><tr><th>#</th><th>Tamil (asked)</th><th>English</th><th>Expected chunk</th><th>On</th><th></th></tr></thead>
                  <tbody>
                    {shown.map((q) => (
                      <tr key={q.id} style={q.enabled ? undefined : { opacity: 0.5 }}>
                        <td className="mono">{q.id}</td>
                        <td>
                          {q.question_ta ? <span className="cell-link indic">{q.question_ta}</span> : <span className="pill pill--warn">no Tamil yet</span>}
                          {q.note && <span className="cell-sub">{q.note}</span>}
                        </td>
                        <td><span className="cell-sub" style={{ display: 'block' }}>{q.question_en}</span><span className={`pill ${q.origin === 'workbook' ? 'pill--violet' : 'pill--info'}`} style={{ marginTop: 4 }}>{q.origin}</span></td>
                        <td className="mono">{q.expected_source}</td>
                        <td><input type="checkbox" checked={q.enabled} onChange={() => toggle(q)} /></td>
                        <td style={{ whiteSpace: 'nowrap' }}>
                          <button type="button" className="btn btn--ghost btn--sm" onClick={() => setEditing({ ...q })}>Edit</button>
                          <button type="button" className="btn btn--ghost btn--sm" onClick={() => remove(q)}>Delete</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {pages > 1 && (
              <div className="card__body" style={{ display: 'flex', alignItems: 'center', gap: 'var(--s3)' }}>
                <button type="button" className="btn btn--ghost btn--sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>← Previous</button>
                <span className="card__note">{page * PAGE + 1}–{Math.min(total, (page + 1) * PAGE)} of {total}</span>
                <button type="button" className="btn btn--ghost btn--sm" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>Next →</button>
              </div>
            )}
          </section>

        {(editing || showAdd) && (
          <div className="scrim" onClick={() => { setEditing(null); setShowAdd(false); }}>
            {editing ? (
              <form className="modal" onSubmit={saveEdit} onClick={(e) => e.stopPropagation()}>
                <div className="modal__head">
                  <span className="card__label">Edit question #{editing.id}</span>
                  <span className="topbar__spacer" />
                  <button type="button" className="btn btn--ghost btn--sm" onClick={() => setEditing(null)}>Close</button>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="e-en">English</label>
                  <textarea id="e-en" className="field__textarea" rows={2} value={editing.question_en} onChange={(e) => setEditing({ ...editing, question_en: e.target.value })} style={{ minHeight: 0 }} />
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="e-ta">Tamil (what the agent is asked)</label>
                  <textarea id="e-ta" className="field__textarea indic" rows={3} autoFocus value={editing.question_ta ?? ''} onChange={(e) => setEditing({ ...editing, question_ta: e.target.value })} style={{ minHeight: 0 }} />
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="e-src">Expected chunk</label>
                  <select id="e-src" className="field__select" value={editing.expected_source} onChange={(e) => setEditing({ ...editing, expected_source: e.target.value })}>
                    {!sources.includes(editing.expected_source) && <option value={editing.expected_source}>{editing.expected_source}</option>}
                    {sources.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="e-note">Note</label>
                  <input id="e-note" className="field__input" value={editing.note ?? ''} onChange={(e) => setEditing({ ...editing, note: e.target.value })} />
                </div>
                <div className="modal__foot">
                  <button type="button" className="btn btn--ghost" onClick={() => setEditing(null)}>Cancel</button>
                  <button type="submit" className="btn btn--primary" disabled={busy}>{busy ? 'Saving…' : 'Save'}</button>
                </div>
              </form>
            ) : (
              <form className="modal" onSubmit={add} onClick={(e) => e.stopPropagation()}>
                <div className="modal__head">
                  <span className="card__label">Add a question</span>
                  <span className="topbar__spacer" />
                  <button type="button" className="btn btn--ghost btn--sm" onClick={() => setShowAdd(false)}>Close</button>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="a-en">English question</label>
                  <textarea id="a-en" className="field__textarea" rows={2} autoFocus value={draft.question_en} onChange={(e) => setDraft({ ...draft, question_en: e.target.value })} placeholder="e.g. How do I block my smart card?" style={{ minHeight: 0 }} />
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="a-ta">Tamil <span className="muted">(optional: written by Sarvam when left empty)</span></label>
                  <textarea id="a-ta" className="field__textarea indic" rows={2} value={draft.question_ta} onChange={(e) => setDraft({ ...draft, question_ta: e.target.value })} style={{ minHeight: 0 }} />
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="a-src">Expected chunk</label>
                  <select id="a-src" className="field__select" value={draft.expected_source} onChange={(e) => setDraft({ ...draft, expected_source: e.target.value })}>
                    <option value="">Choose the sheet that should answer it…</option>
                    {sources.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="a-note">Note</label>
                  <input id="a-note" className="field__input" value={draft.note} onChange={(e) => setDraft({ ...draft, note: e.target.value })} />
                </div>
                <div className="modal__foot">
                  <button type="button" className="btn btn--ghost" onClick={() => setShowAdd(false)}>Cancel</button>
                  <button type="submit" className="btn btn--primary" disabled={busy || !draft.question_en.trim() || !draft.expected_source}>{busy ? 'Adding…' : 'Add question'}</button>
                </div>
              </form>
            )}
          </div>
        )}
      </div>
    </>
  );
}
