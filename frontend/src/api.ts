import type {
  AgentConfig,
  CallDetail,
  CallSummary,
  ConfigVersion,
  KnowledgeDocument,
  KnowledgeSearch,
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
  calls: (limit = 200) => get<CallSummary[]>(`/api/calls?limit=${limit}`),
  call: (id: string) => get<{ call: CallDetail; turns: Turn[] }>(`/api/calls/${id}`),
  config: () => get<AgentConfig>('/api/config'),
  configHistory: () => get<ConfigVersion[]>('/api/config/history'),
  previewPrompt: (language: string, systemPrompt: string | null) => {
    const q = new URLSearchParams({ language });
    if (systemPrompt) q.set('system_prompt', systemPrompt);
    return get<{ language: string; prompt: string }>(`/api/config/preview?${q}`);
  },
  saveConfig: async (body: {
    system_prompt: string | null;
    temperature: number | null;
    language: string | null;
    greeting: string | null;
    note: string | null;
  }) => {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = (await res.json()) as { id?: number; detail?: string };
    if (!res.ok) throw new Error(data.detail ?? `${res.status}`);
    return data.id as number;
  },
};
