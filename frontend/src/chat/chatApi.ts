/** The text chat's own API client and types.
 *
 * Kept apart from `src/api.ts` on purpose: the chat is a separate backend
 * package (`backend/chat_agent`), and nothing in the voice-agent pages imports
 * from here.
 */

export interface ChatStatus {
  available: boolean;
  config_id: number | null;
  language: string | null;
  model: string | null;
  temperature: number | null;
  greeting: string | null;
  knowledge_on: boolean;
  topic_context_on: boolean;
  min_score: number | null;
  top_k: number | null;
  query_rewrite_on: boolean;
}

export interface ChatHit {
  source: string;
  document_id: string;
  chunk_index: number;
  score: number;
  text: string;
  used: boolean;
}

export interface ChatSearch {
  question: string;
  translated_query: string | null;
  rewritten_query: string | null;
  translate_ms: number;
  rewrite_ms: number;
  embed_ms: number;
  search_ms: number;
  hits: ChatHit[];
  error: string | null;
}

/** One step the turn ran; the fields past `step`/`ms` differ per step. */
export interface ChatStep {
  step: string;
  ms: number;
  decision?: string | null;
  reason?: string | null;
  restarted?: boolean;
  messages_sent?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  searched?: boolean;
  finish?: string | null;
  found?: boolean;
  sheets_sent?: number;
  sheets_retrieved?: number;
  sources?: (string | null)[];
  questions?: unknown[];
  note?: string;
  error?: string;
}

export interface ChatTurn {
  index: number;
  user_text: string;
  reply: string;
  searched: boolean;
  filler: string | null;
  topic: { decision?: string | null; reason?: string | null } | null;
  steps: ChatStep[];
  searches: ChatSearch[];
  dropped_text: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  ms: number;
  error: string | null;
  at: string;
}

export interface ChatSessionInfo {
  id: string;
  language: string;
  language_name: string;
  greeting: string | null;
  config_id: number | null;
  model: string;
  temperature: number | null;
  knowledge_on: boolean;
  topic_context_on: boolean;
  min_score: number | null;
  top_k: number | null;
  started_at: string;
  turns: ChatTurn[];
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const data = (await res.json().catch(() => ({}))) as T & { detail?: string };
  if (!res.ok) throw new Error(data.detail ?? `${res.status} ${res.statusText}`);
  return data;
}

export const chatApi = {
  status: () => call<ChatStatus>('/api/chat/status'),
  start: () => call<ChatSessionInfo>('/api/chat/sessions', { method: 'POST' }),
  send: (sessionId: string, text: string) =>
    call<ChatTurn>(`/api/chat/sessions/${sessionId}/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text }),
    }),
  end: (sessionId: string) =>
    call<{ ended: boolean }>(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' }),
};
