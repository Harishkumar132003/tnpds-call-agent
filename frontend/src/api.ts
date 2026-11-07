import type {
  EvalBatch,
  EvalQuestion,
  EvalResult,
  ModelCatalogue,
  TestCallInfo,
  AgentConfig,
  CallDetail,
  CallSummary,
  ConfigVersion,
  KnowledgeDocument,
  KnowledgeSearch,
  RewriteQuestion,
  RewriteResult,
  RewriteRun,
  SearchType,
  KnowledgeStatus,
  Stats,
  Turn,
} from './types';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function send<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const data = (await res.json().catch(() => ({}))) as T & { detail?: string };
  if (!res.ok) throw new Error(data.detail ?? `${res.status} ${res.statusText}`);
  return data;
}

export const api = {
  pronunciationTest: (text: string, pronunciations: string | null) =>
    send<{ audio_base64: string; dict_id: string | null }>('/api/pronunciation/test', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, pronunciations }),
    }),
  knowledgeStatus: () => get<KnowledgeStatus>('/api/knowledge/status'),
  documents: () => get<KnowledgeDocument[]>('/api/documents'),
  uploadDocument: (file: File) => {
    const body = new FormData();
    body.append('file', file);
    return send<KnowledgeDocument>('/api/documents', { method: 'POST', body });
  },
  deleteDocument: (id: string) => send<{ deleted: boolean }>(`/api/documents/${id}`, { method: 'DELETE' }),
  knowledgeSearch: (question: string, limit = 5, documentId?: string) =>
    send<KnowledgeSearch>('/api/knowledge/search', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question, limit, document_id: documentId ?? null }),
    }),
  stats: () => get<Stats>('/api/stats'),
  testCall: () => get<TestCallInfo>('/api/test-call'),
  calls: (limit = 200) => get<CallSummary[]>(`/api/calls?limit=${limit}`),
  call: (id: string) => get<{ call: CallDetail; turns: Turn[] }>(`/api/calls/${id}`),
  config: () => get<AgentConfig>('/api/config'),
  configHistory: () => get<ConfigVersion[]>('/api/config/history'),
  models: (language: string) => get<ModelCatalogue>(`/api/models?language=${encodeURIComponent(language)}`),
  // POST: the draft prompt goes in the body, not the URL (a long Tamil prompt hit 414)
  previewPrompt: (language: string, systemPrompt: string | null, routerPrompt?: string | null, answerPrompt?: string | null) =>
    send<{ language: string; prompt: string; answer_prompt: string | null }>('/api/config/preview', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ language, system_prompt: systemPrompt || null, router_prompt: routerPrompt || null, answer_prompt: answerPrompt || null }),
    }),
  saveConfig: async (body: {
    system_prompt: string | null;
    temperature: number | null;
    language: string | null;
    greeting: string | null;
    idle_prompt_secs: number | null;
    idle_check_line: string | null;
    idle_goodbye_line: string | null;
    router_prompt: string | null;
    answer_prompt: string | null;
    /** null keeps the saved topic prompt; "" resets it to the built-in text */
    topic_prompt?: string | null;
    /** null keeps the saved rewrite prompt; "" resets it to the built-in text */
    rewrite_prompt?: string | null;
    pronunciations?: string | null;
    stt_model?: string | null;
    llm_model?: string | null;
    tts_model?: string | null;
    tts_speaker?: string | null;
    translate_model?: string | null;
    embedding_model?: string | null;
    note: string | null;
  }) => {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = (await res.json()) as { id?: number; reindexed?: number | null; needs_reupload?: number | null; detail?: string };
    if (!res.ok) throw new Error(data.detail ?? `${res.status}`);
    return data as { id: number; reindexed: number | null; needs_reupload: number | null };
  },
  // ---- evaluation ----
  evalQuestions: (offset = 0, limit = 50, q = '') =>
    get<{ questions: EvalQuestion[]; total: number; offset: number; limit: number; untranslated: number; runnable: number; translating: boolean }>(
      `/api/evals/questions?offset=${offset}&limit=${limit}${q ? `&q=${encodeURIComponent(q)}` : ''}`,
    ),
  evalAddQuestion: (body: { question_en: string; question_ta: string | null; expected_source: string; note: string | null }) =>
    send<EvalQuestion>('/api/evals/questions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
  evalUpdateQuestion: (id: number, body: Partial<Pick<EvalQuestion, 'question_en' | 'question_ta' | 'expected_source' | 'note' | 'enabled'>>) =>
    send<EvalQuestion>(`/api/evals/questions/${id}`, { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
  evalDeleteQuestion: (id: number) => send<{ deleted: boolean }>(`/api/evals/questions/${id}`, { method: 'DELETE' }),
  evalTranslateQuestion: (id: number) => send<EvalQuestion>(`/api/evals/questions/${id}/translate`, { method: 'POST' }),
  evalRetranslate: (origin: string | null) =>
    send<{ cleared: number; started: boolean }>('/api/evals/questions/retranslate', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ origin }) }),
  evalTranslateMissing: () => send<{ started: boolean }>('/api/evals/questions/translate-missing', { method: 'POST' }),
  evalImport: (path: string) =>
    send<{ read: number; added: number; skipped: number }>('/api/evals/questions/import', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path }) }),
  evalSources: () => get<{ sources: string[]; names: string[] }>('/api/evals/sources'),
  evalBatches: () => get<{ batches: EvalBatch[]; running: string | null }>('/api/evals/batches'),
  evalBatch: (id: string) => get<{ batch: EvalBatch; results: EvalResult[] }>(`/api/evals/batches/${id}`),
  evalRun: (note: string | null, concurrency = 1, searchType: SearchType = 'semantic', opts: { collection?: string | null; questionIds?: number[] | null } = {}) =>
    send<EvalBatch>('/api/evals/batches', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ note, concurrency, search_type: searchType, collection: opts.collection ?? null, question_ids: opts.questionIds ?? null }) }),
  evalCancel: (id: string) => send<{ cancelled: boolean }>(`/api/evals/batches/${id}/cancel`, { method: 'POST' }),
  evalDeleteBatch: (id: string) => send<{ deleted: boolean }>(`/api/evals/batches/${id}`, { method: 'DELETE' }),
  rewriteQuestions: () => get<{ questions: RewriteQuestion[]; seeded: number }>('/api/rewrite-evals/questions'),
  rewriteAddQuestion: (q: { question_ta: string; question_en: string | null; expected_source: string }) =>
    send<RewriteQuestion>('/api/rewrite-evals/questions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(q) }),
  rewriteUpdateQuestion: (id: number, q: { question_ta: string; question_en: string | null; expected_source: string }) =>
    send<RewriteQuestion>(`/api/rewrite-evals/questions/${id}`, { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(q) }),
  rewriteDeleteQuestion: (id: number) => send<{ deleted: boolean }>(`/api/rewrite-evals/questions/${id}`, { method: 'DELETE' }),
  rewriteAddRandom: (count: number) =>
    send<{ added: number }>('/api/rewrite-evals/questions/random', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ count }) }),
  rewriteRuns: () => get<{ runs: RewriteRun[]; running: boolean; call_timeout_ms: number }>('/api/rewrite-evals/runs'),
  rewriteRun: (id: string) => get<{ run: RewriteRun; results: RewriteResult[] }>(`/api/rewrite-evals/runs/${id}`),
  rewriteRunStart: (prompt: string | null, searchType: SearchType) =>
    send<{ run: RewriteRun; results: RewriteResult[] }>('/api/rewrite-evals/runs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt, search_type: searchType }) }),
  rewriteJudge: (runId: string, resultId: number, ok: boolean) =>
    send<{ run: RewriteRun }>(`/api/rewrite-evals/runs/${runId}/results/${resultId}`, { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ok }) }),
  rewriteDeleteRun: (id: string) => send<{ deleted: boolean }>(`/api/rewrite-evals/runs/${id}`, { method: 'DELETE' }),
};
