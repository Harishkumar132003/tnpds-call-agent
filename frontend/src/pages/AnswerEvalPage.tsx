import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { StatTile } from '../components/StatTile';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import type { AgentConfig, AnswerDataset, AnswerItem, AnswerMessage, AnswerResult, AnswerRun } from '../types';

const MAIN = 'main';

const pct = (n: number, of: number) => (of ? `${Math.round((100 * n) / of)}%` : '—');

const promptLabel = (run: AnswerRun) => (
  run.prompt_source === 'default' ? 'built-in prompt' : run.prompt_source === 'config' ? 'configuration prompt' : 'draft prompt'
);

const isExpected = (expected: string | null, source: string) =>
  Boolean(expected && (source === expected || source.startsWith(`${expected} `)));

function Conversation({ item }: { item: AnswerItem }) {
  const earlier = item.messages.slice(0, -1);
  const last = item.messages[item.messages.length - 1];
  const [open, setOpen] = useState(false);
  return (
    <div className="stack" style={{ gap: 4 }}>
      {earlier.length > 0 && (
        <button type="button" className="btn btn--ghost btn--sm" style={{ alignSelf: 'flex-start' }} onClick={() => setOpen((v) => !v)}>
          {open ? 'hide' : 'show'} {earlier.length} earlier message{earlier.length === 1 ? '' : 's'}
        </button>
      )}
      {open && earlier.map((m, i) => (
        <div key={i} className="cell-sub indic" style={{ whiteSpace: 'pre-wrap' }}>
          <span className="mono" style={{ textTransform: 'uppercase', fontSize: 10 }}>{m.role === 'user' ? 'caller' : 'agent'}</span>{' '}{m.text}
        </div>
      ))}
      <span className="cell-link indic" style={{ whiteSpace: 'pre-wrap' }}>{last?.text}</span>
      {item.question_en && <span className="cell-sub">{item.question_en}</span>}
      {item.rewritten_query && <span className="cell-sub" title="the English query the sheets were found with">⌕ {item.rewritten_query}</span>}
    </div>
  );
}

function Sheets({ item }: { item: AnswerItem }) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  if (!item.sheets.length) return <span className="muted">no sheet retrieved</span>;
  return (
    <div className="kb">
      {item.sheets.map((s, i) => {
        const id = s.source.split(' ')[0];
        const expected = isExpected(item.expected_source, s.source);
        return (
          <div key={`${s.document_id ?? id}-${i}`}>
            <div className="kb__row" style={expected ? { fontWeight: 600 } : undefined}>
              <span className="mono">#{i + 1}</span>{' '}
              <span title={s.source}>{id}</span>{' '}
              <span className="pill pill--idle">{s.score.toFixed(3)}</span>
              {s.used ? <span className="pill pill--info">sent</span> : <span className="pill pill--idle">not sent</span>}
              {expected && <span className="pill pill--violet">expected</span>}{' '}
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => setOpenIdx(openIdx === i ? null : i)}>{openIdx === i ? 'hide' : 'text'}</button>
            </div>
            {openIdx === i && <pre className="preview-prompt" style={{ whiteSpace: 'pre-wrap', margin: '4px 0 8px', maxHeight: 320, overflow: 'auto' }}>{s.text}</pre>}
          </div>
        );
      })}
    </div>
  );
}

/** A fixed set of conversations with frozen search results, answered with a draft answer prompt
 *  (LLM 2), so the answer rules can be tuned against the same inputs every time. */
export function AnswerEvalPage() {
  const toast = useToast();
  const [config, setConfig] = useState<AgentConfig | null>(null);
  const [items, setItems] = useState<AnswerItem[] | null>(null);
  const [datasets, setDatasets] = useState<AnswerDataset[]>([]);
  const [dataset, setDataset] = useState<string>(MAIN);
  const [allRuns, setAllRuns] = useState<AnswerRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [compare, setCompare] = useState<string>('');
  const [results, setResults] = useState<Record<string, AnswerResult[]>>({});
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [maxSheets, setMaxSheets] = useState<number | null>(null);
  const [minScore, setMinScore] = useState<number | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);
  const [filter, setFilter] = useState<'all' | 'single' | 'follow_up' | 'wrong' | 'unjudged'>('all');
  const [editing, setEditing] = useState<{ id: number | null; messages: AnswerMessage[]; question_en: string; expected_source: string; note: string; dataset: string } | null>(null);
  const [sources, setSources] = useState<string[]>([]);

  const fail = (title: string) => (e: unknown) => toast.show({ tone: 'error', title, body: (e as Error).message ?? String(e) });

  const runs = useMemo(() => allRuns.filter((r) => (r.dataset || MAIN) === dataset), [allRuns, dataset]);
  const loadDatasets = () => api.answerDatasets().then((d) => setDatasets(d.datasets));
  const loadItems = (name = dataset) => api.answerItems(name).then((d) => setItems(d.items));
  const loadRuns = () => api.answerRuns().then((d) => setAllRuns(d.runs));

  useEffect(() => {
    api.config().then((c) => { setConfig(c); setPrompt(c.answer_prompt ?? c.answer_prompt_default); }).catch(fail('Could not load the configuration'));
    loadDatasets().catch(fail('Could not load the datasets'));
    loadRuns().catch(fail('Could not load the runs'));
    api.evalSources().then((d) => setSources(d.sources)).catch(() => undefined);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // switching dataset: its items, and its latest run selected
  useEffect(() => {
    setItems(null);
    setCompare('');
    loadItems(dataset).catch(fail('Could not load the items'));
  }, [dataset]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setSelected((cur) => (cur && runs.some((r) => r.id === cur) ? cur : runs[0]?.id ?? null));
  }, [runs]);

  const ensureResults = (id: string | null) => {
    if (!id || results[id]) return;
    api.answerRun(id).then((d) => setResults((r) => ({ ...r, [id]: d.results }))).catch(fail('Could not load the run'));
  };
  useEffect(() => { ensureResults(selected); }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { ensureResults(compare || null); }, [compare]); // eslint-disable-line react-hooks/exhaustive-deps

  const generate = async () => {
    setGenerating(true);
    try {
      const d = await api.answerGenerate(60, 40, dataset);
      setItems(d.items);
      await loadDatasets();
      toast.show({ tone: 'success', title: 'Set generated', body: `${d.added.single} single questions from the evaluation bank and ${d.added.follow_up} follow-ups from real calls, each with the sheets a search returns today.` });
    } catch (e) { fail('Could not generate')(e); } finally { setGenerating(false); }
  };

  const run = async () => {
    setBusy(true);
    try {
      const d = await api.answerRunStart(prompt.trim() || null, maxSheets, minScore, dataset);
      setResults((r) => ({ ...r, [d.run.id]: d.results }));
      await loadRuns();
      await loadDatasets();
      setCompare(selected && selected !== d.run.id ? selected : '');
      setSelected(d.run.id);
      toast.show({ tone: 'success', title: 'Run finished', body: `${d.run.answered} of ${d.run.results} answered${d.run.errors ? `, ${d.run.errors} errors` : ''}. Tick the answers that are correct.` });
    } catch (e) { fail('Run failed')(e); } finally { setBusy(false); }
  };

  const judge = async (r: AnswerResult, ok: boolean | null, note?: string | null) => {
    const judgeNote = note === undefined ? r.judge_note : note;
    setResults((all) => ({ ...all, [r.run_id]: (all[r.run_id] ?? []).map((x) => (x.id === r.id ? { ...x, judged: ok !== null, judged_ok: Boolean(ok), judge_note: judgeNote } : x)) }));
    try {
      const d = await api.answerJudge(r.run_id, r.id, ok, judgeNote);
      setAllRuns((rs) => rs.map((x) => (x.id === d.run.id ? d.run : x)));
    } catch (err) { fail('Could not save the verdict')(err); }
  };

  const saveItem = async (e: FormEvent) => {
    e.preventDefault();
    if (!editing) return;
    const target = editing.dataset.trim() || MAIN;
    const body = { messages: editing.messages.filter((m) => m.text.trim()), question_en: editing.question_en.trim() || null, expected_source: editing.expected_source.trim() || null, note: editing.note.trim() || null, dataset: target };
    try {
      if (editing.id === null) await api.answerAddItem(body); else await api.answerUpdateItem(editing.id, body);
      setEditing(null);
      await loadDatasets();
      if (target !== dataset) setDataset(target); else await loadItems();
    } catch (err) { fail('Could not save')(err); }
  };

  const remove = async (item: AnswerItem) => {
    if (!window.confirm('Remove this item from the set?')) return;
    try { await api.answerDeleteItem(item.id); await loadItems(); await loadDatasets(); } catch (err) { fail('Could not remove')(err); }
  };

  const refresh = async (item: AnswerItem) => {
    try {
      await api.answerRefreshItem(item.id);
      await loadItems();
      toast.show({ tone: 'success', title: 'Sheets refreshed', body: 'Fetched again from the current collection. Earlier runs keep the answers they gave for the old sheets.' });
    } catch (err) { fail('Could not refresh')(err); }
  };

  const selectedRun = runs.find((r) => r.id === selected) ?? null;
  const compareRun = runs.find((r) => r.id === compare) ?? null;
  const byItem = (id: string | null) => {
    const map = new Map<number | null, AnswerResult>();
    for (const r of (id && results[id]) || []) map.set(r.item_id, r);
    return map;
  };
  const cur = useMemo(() => byItem(selected), [selected, results]); // eslint-disable-line react-hooks/exhaustive-deps
  const prev = useMemo(() => byItem(compare || null), [compare, results]); // eslint-disable-line react-hooks/exhaustive-deps

  const trimmed = prompt.trim();
  const promptState = !config ? '' : trimmed === config.answer_prompt_default.trim() ? 'built-in default' : trimmed === (config.answer_prompt ?? '').trim() ? 'same as the configuration' : 'draft, not saved';

  const shown = (items ?? []).filter((it) => {
    if (filter === 'single' || filter === 'follow_up') return it.kind === filter;
    const r = cur.get(it.id);
    if (filter === 'wrong') return Boolean(r?.judged && !r.judged_ok);
    if (filter === 'unjudged') return Boolean(r && !r.error && !r.judged);
    return true;
  });
  const counts = { single: (items ?? []).filter((i) => i.kind === 'single').length, follow_up: (items ?? []).filter((i) => i.kind === 'follow_up').length };

  return (
    <>
      <Topbar
        title="Answer prompt"
        actions={(
          <>
            <label className="field__hint" htmlFor="ae-dataset">dataset{' '}
              <select id="ae-dataset" className="field__select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                {!datasets.some((d) => d.dataset === dataset) && <option value={dataset}>{dataset}</option>}
                {datasets.map((d) => <option key={d.dataset} value={d.dataset}>{d.dataset} ({d.items})</option>)}
              </select>
            </label>
            {items && <span className="pill pill--idle">{items.length} items · {counts.single} single · {counts.follow_up} follow-up</span>}
            <button type="button" className="btn btn--ghost btn--sm" disabled={generating} onClick={generate}>{generating ? 'Generating…' : items?.length ? 'Add 60 + 40 more' : 'Generate 100 items'}</button>
            <button type="button" className="btn btn--ghost btn--sm" onClick={() => setEditing({ id: null, messages: [{ role: 'user', text: '' }], question_en: '', expected_source: '', note: '', dataset })}>Add item</button>
          </>
        )}
      />
      <div className="page stack">
        <section className="card">
          <div className="card__head">
            <span className="card__label">Answer rules to try (LLM 2)</span>
            {promptState && <span className={`pill ${promptState.startsWith('draft') ? 'pill--violet' : 'pill--idle'}`}>{promptState}</span>}
            <span className="topbar__spacer" />
            {config && <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(config.answer_prompt ?? config.answer_prompt_default)}>Use the configuration's</button>}
            {config && <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(config.answer_prompt_default)}>Use the built-in</button>}
          </div>
          <div className="card__body stack">
            <textarea className="field__textarea" rows={14} value={prompt} onChange={(e) => setPrompt(e.target.value)} spellCheck={false} />
            <div className="row" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <span className="field__hint" style={{ flex: 1, minWidth: 260 }}>Each item is answered by the active model from its frozen sheets, with the persona from the configuration and these rules, exactly as a call composes the answer step. Nothing is saved to the configuration: when a prompt works, paste it on the Configuration page.</span>
              <div className="field" style={{ minWidth: 200 }}>
                <label className="field__label" htmlFor="ae-max">Sheets given to the model</label>
                <select id="ae-max" className="field__select" value={maxSheets ?? ''} onChange={(e) => setMaxSheets(e.target.value ? Number(e.target.value) : null)}>
                  <option value="">all sent sheets (as calls do)</option>
                  <option value="3">first 3 only</option>
                  <option value="2">first 2 only</option>
                  <option value="1">first 1 only</option>
                </select>
              </div>
              <div className="field" style={{ minWidth: 200 }}>
                <label className="field__label" htmlFor="ae-min">Score floor (hybrid RRF)</label>
                <select id="ae-min" className="field__select" value={minScore ?? ''} onChange={(e) => setMinScore(e.target.value ? Number(e.target.value) : null)}>
                  <option value="">none</option>
                  <option value="0.3">0.300 and above</option>
                  <option value="0.4">0.400 and above</option>
                  <option value="0.5">0.500 and above</option>
                </select>
              </div>
              <button type="button" className="btn btn--primary" disabled={busy || !items?.length} onClick={run}>{busy ? 'Running…' : `Run ${items?.length ?? 0} items of ${dataset}`}</button>
            </div>
          </div>
        </section>

        {selectedRun && (
          <div className="stats">
            <StatTile label="Answered" icon="chat" value={pct(selectedRun.answered, selectedRun.results)} note={`${selectedRun.answered} of ${selectedRun.results}${selectedRun.errors ? ` · ${selectedRun.errors} errors` : ''}`} />
            <StatTile label="Marked correct" icon="tick" value={pct(selectedRun.judged_ok, selectedRun.judged)} note={`${selectedRun.judged_ok} of ${selectedRun.judged} judged by you`} pill={selectedRun.judged ? { tone: selectedRun.judged_ok / selectedRun.judged >= 0.8 ? 'up' : 'warn', text: `${selectedRun.results - selectedRun.judged} to judge` } : undefined} />
            <StatTile label="Answer time" icon="refresh" value={selectedRun.avg_ms ? `${(selectedRun.avg_ms / 1000).toFixed(1)} s` : '—'} note="average per item" />
            <StatTile label="Model" icon="sparkle" value={selectedRun.model ?? '—'} note={`config #${selectedRun.config_id ?? '—'} · ${promptLabel(selectedRun)}${selectedRun.max_sheets ? ` · first ${selectedRun.max_sheets} sheets only` : ''}${selectedRun.min_score ? ` · score ≥ ${selectedRun.min_score.toFixed(3)}` : ''}`} />
          </div>
        )}

        <section className="card">
          <div className="card__head">
            <span className="card__label">Items</span>
            {selectedRun && <span className="card__note">showing run {when(selectedRun.started_at)} · {promptLabel(selectedRun)}</span>}
            {selectedRun && <button type="button" className="btn btn--ghost btn--sm" onClick={() => setShowPrompt((v) => !v)}>{showPrompt ? 'Hide prompt' : 'Show prompt'}</button>}
            <span className="topbar__spacer" />
            <label className="field__hint" htmlFor="ae-filter">show{' '}
              <select id="ae-filter" className="field__select" value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
                <option value="all">all</option>
                <option value="single">single questions</option>
                <option value="follow_up">follow-ups</option>
                <option value="wrong">marked wrong</option>
                <option value="unjudged">not judged yet</option>
              </select>
            </label>
            {runs.length > 1 && (
              <label className="field__hint" htmlFor="ae-compare">compare with{' '}
                <select id="ae-compare" className="field__select" value={compare} onChange={(e) => setCompare(e.target.value)}>
                  <option value="">—</option>
                  {runs.filter((r) => r.id !== selected).map((r) => <option key={r.id} value={r.id}>{when(r.started_at)} · {promptLabel(r)}</option>)}
                </select>
              </label>
            )}
          </div>
          {showPrompt && selectedRun && (
            <div className="card__body stack" style={{ borderBottom: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <span className="card__label">Rules used by this run</span>
                <span className={`pill ${selectedRun.prompt_source === 'draft' ? 'pill--violet' : 'pill--idle'}`}>{promptLabel(selectedRun)}</span>
                {selectedRun.prompt.trim() === prompt.trim()
                  ? <span className="pill pill--up">same as the editor</span>
                  : <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(selectedRun.prompt)}>Load into the editor</button>}
                {compareRun && (compareRun.prompt.trim() === selectedRun.prompt.trim()
                  ? <span className="pill pill--idle">compared run used the same rules</span>
                  : <span className="pill pill--warn">compared run used different rules</span>)}
              </div>
              <pre className="preview-prompt" style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{selectedRun.prompt}</pre>
            </div>
          )}
          <div className="table-wrap">
            <table className="table table--left">
              <thead><tr><th>#</th><th>Conversation</th><th>Sheets given to the model</th><th>Answer</th><th title="Correct = only facts from the sent sheets, answers the question, conditions kept">Verdict</th><th></th></tr></thead>
              <tbody>
                {shown.map((it) => {
                  const r = cur.get(it.id);
                  const p = prev.get(it.id);
                  return (
                    <tr key={it.id}>
                      <td className="mono" style={{ whiteSpace: 'nowrap' }}>{it.id}<div><span className={`pill ${it.kind === 'follow_up' ? 'pill--violet' : 'pill--idle'}`}>{it.kind === 'follow_up' ? 'follow-up' : 'single'}</span></div>{it.expected_source && <div className="cell-sub">expects {it.expected_source}</div>}</td>
                      <td style={{ maxWidth: 300 }}><Conversation item={it} /></td>
                      <td style={{ minWidth: 240, maxWidth: 360 }}><Sheets item={it} /></td>
                      <td style={{ maxWidth: 420 }}>
                        {r?.answer && <span className="indic" style={{ whiteSpace: 'pre-wrap' }}>{r.answer}</span>}
                        {r?.error && <span className="cell-sub" style={{ color: 'var(--down)' }}>{r.error}</span>}
                        {!r && <span className="muted">—</span>}
                        {r && !r.error && <div className="cell-sub">{r.ms ? `${(r.ms / 1000).toFixed(1)} s` : ''}{r.completion_tokens ? ` · ${r.completion_tokens} tokens` : ''}</div>}
                        {compareRun && p && <div className="cell-sub indic" style={{ whiteSpace: 'pre-wrap', borderTop: '1px dashed var(--border)', marginTop: 6, paddingTop: 4 }} title="the run compared with">↳ {p.answer ?? (p.error ? `error: ${p.error}` : '—')}{p.judged ? ` (${p.judged_ok ? 'correct' : 'wrong'})` : ''}</div>}
                      </td>
                      <td style={{ whiteSpace: 'nowrap', minWidth: 150 }}>
                        {r && !r.error ? (
                          <div className="stack" style={{ gap: 4 }}>
                            <div>
                              <button type="button" className={`btn btn--sm ${r.judged && r.judged_ok ? 'btn--primary' : 'btn--ghost'}`} onClick={() => judge(r, r.judged && r.judged_ok ? null : true)} title="The answer uses only the sent sheets and answers the question">✓ correct</button>{' '}
                              <button type="button" className={`btn btn--sm ${r.judged && !r.judged_ok ? 'btn--primary' : 'btn--ghost'}`} onClick={() => judge(r, r.judged && !r.judged_ok ? null : false)} title="Wrong, invented, merged across sheets, or missing a condition">✗ wrong</button>
                            </div>
                            <input className="field__input" style={{ fontSize: 12 }} placeholder="why (optional)" defaultValue={r.judge_note ?? ''} onBlur={(e) => { if ((e.target.value || '') !== (r.judge_note ?? '')) judge(r, r.judged ? r.judged_ok : null, e.target.value || null); }} />
                          </div>
                        ) : <span className="muted">—</span>}
                      </td>
                      <td style={{ whiteSpace: 'nowrap' }}>
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => setEditing({ id: it.id, messages: it.messages.map((m) => ({ ...m })), question_en: it.question_en ?? '', expected_source: it.expected_source ?? '', note: it.note ?? '', dataset: it.dataset || MAIN })}>Edit</button>{' '}
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => refresh(it)} title="Fetch the sheets again from the current collection">Refresh sheets</button>{' '}
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => remove(it)}>Remove</button>
                      </td>
                    </tr>
                  );
                })}
                {items && items.length === 0 && <tr><td colSpan={6} className="muted">No items in dataset "{dataset}" yet: generate a set (60 single questions from the evaluation bank + 40 follow-ups from real calls), or add one.</td></tr>}
                {items && items.length > 0 && shown.length === 0 && <tr><td colSpan={6} className="muted">Nothing matches this filter.</td></tr>}
                {items === null && <tr><td colSpan={6}><span className="spinner" /></td></tr>}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <div className="card__head">
            <span className="card__label">Runs</span>
            <span className="card__note">dataset {dataset}</span>
            <span className="topbar__spacer" />
            <span className="card__note">click a run to show its answers above · "Load prompt" puts that run's rules in the editor</span>
          </div>
          {runs.length === 0 ? (
            <div className="card__body muted">No runs for this dataset yet.</div>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Started</th><th>Prompt</th><th>Sheets</th><th>Items</th><th>Answered</th><th>Judged</th><th>Correct</th><th>Errors</th><th>Answer time</th><th>Model</th><th>Config</th><th></th></tr></thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id} className={`row-link${r.id === selected ? ' is-selected' : ''}`} onClick={() => setSelected(r.id)}>
                      <td><span className="cell-link">{when(r.started_at)}</span>{r.status !== 'done' && <span className="cell-sub">{r.status}{r.error ? `: ${r.error}` : ''}</span>}</td>
                      <td><span className={`pill ${r.prompt_source === 'draft' ? 'pill--violet' : 'pill--idle'}`} title={r.prompt}>{promptLabel(r)}</span></td>
                      <td>{[r.max_sheets ? `first ${r.max_sheets}` : null, r.min_score ? `≥ ${r.min_score.toFixed(3)}` : null].filter(Boolean).join(' · ') || 'all'}</td>
                      <td className="mono">{r.results}/{r.total}</td>
                      <td>{pct(r.answered, r.results)}</td>
                      <td className="mono">{r.judged}/{r.results}</td>
                      <td>{r.judged ? pct(r.judged_ok, r.judged) : '—'}</td>
                      <td>{r.errors || '—'}</td>
                      <td>{r.avg_ms ? `${(r.avg_ms / 1000).toFixed(1)} s` : '—'}</td>
                      <td className="mono">{r.model ?? '—'}</td>
                      <td className="mono">{r.config_id ? `#${r.config_id}` : '—'}</td>
                      <td onClick={(e) => e.stopPropagation()} style={{ whiteSpace: 'nowrap' }}>
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPrompt(r.prompt)}>Load prompt</button>{' '}
                        <button type="button" className="btn btn--ghost btn--sm" onClick={() => { if (window.confirm('Delete this run?')) api.answerDeleteRun(r.id).then(() => { if (selected === r.id) setSelected(null); if (compare === r.id) setCompare(''); return loadRuns(); }).catch(fail('Could not delete')); }}>Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {editing && (
          <div className="scrim" onClick={() => setEditing(null)}>
            <form className="modal" onSubmit={saveItem} onClick={(e) => e.stopPropagation()} style={{ maxWidth: 720 }}>
              <div className="modal__head">
                <span className="card__label">{editing.id === null ? 'Add an item' : `Edit item ${editing.id}`}</span>
                <span className="topbar__spacer" />
                <button type="button" className="btn btn--ghost btn--sm" onClick={() => setEditing(null)}>Close</button>
              </div>
              <div className="field">
                <span className="field__label">Conversation <span className="muted">(the last message must be the caller's; the sheets are fetched for it)</span></span>
                <div className="stack" style={{ gap: 6 }}>
                  {editing.messages.map((m, i) => (
                    <div key={i} style={{ display: 'flex', gap: 6 }}>
                      <select className="field__select" style={{ width: 110 }} value={m.role} onChange={(e) => setEditing({ ...editing, messages: editing.messages.map((x, j) => (j === i ? { ...x, role: e.target.value as AnswerMessage['role'] } : x)) })}>
                        <option value="user">caller</option>
                        <option value="assistant">agent</option>
                      </select>
                      <textarea className="field__textarea indic" rows={2} style={{ flex: 1 }} value={m.text} onChange={(e) => setEditing({ ...editing, messages: editing.messages.map((x, j) => (j === i ? { ...x, text: e.target.value } : x)) })} />
                      <button type="button" className="btn btn--ghost btn--sm" disabled={editing.messages.length === 1} onClick={() => setEditing({ ...editing, messages: editing.messages.filter((_, j) => j !== i) })}>×</button>
                    </div>
                  ))}
                  <button type="button" className="btn btn--ghost btn--sm" style={{ alignSelf: 'flex-start' }} onClick={() => setEditing({ ...editing, messages: [...editing.messages, { role: editing.messages[editing.messages.length - 1]?.role === 'user' ? 'assistant' : 'user', text: '' }] })}>+ message</button>
                </div>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="ae-en">English <span className="muted">(optional, for your reference)</span></label>
                <input id="ae-en" className="field__input" value={editing.question_en} onChange={(e) => setEditing({ ...editing, question_en: e.target.value })} />
              </div>
              <div className="field">
                <label className="field__label" htmlFor="ae-src">Expected sheet <span className="muted">(optional)</span></label>
                <input id="ae-src" className="field__input mono" list="ae-sources" value={editing.expected_source} onChange={(e) => setEditing({ ...editing, expected_source: e.target.value })} placeholder="e.g. pb87-2" />
                <datalist id="ae-sources">{sources.map((s) => <option key={s} value={s} />)}</datalist>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="ae-note">Note</label>
                <input id="ae-note" className="field__input" value={editing.note} onChange={(e) => setEditing({ ...editing, note: e.target.value })} />
              </div>
              <div className="field">
                <label className="field__label" htmlFor="ae-set">Dataset <span className="muted">(a new name makes a new dataset)</span></label>
                <input id="ae-set" className="field__input" list="ae-datasets" value={editing.dataset} onChange={(e) => setEditing({ ...editing, dataset: e.target.value })} />
                <datalist id="ae-datasets">{datasets.map((d) => <option key={d.dataset} value={d.dataset} />)}</datalist>
              </div>
              <div className="modal__foot">
                <span className="field__hint">Changing the conversation fetches the sheets again from the current collection.</span>
                <button type="button" className="btn btn--ghost" onClick={() => setEditing(null)}>Cancel</button>
                <button type="submit" className="btn btn--primary">{editing.id === null ? 'Add item' : 'Save'}</button>
              </div>
            </form>
          </div>
        )}
      </div>
    </>
  );
}
