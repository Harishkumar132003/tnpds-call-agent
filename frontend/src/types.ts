export interface CallSummary {
  id: string;
  transport: string;
  direction: string | null;
  from_number: string | null;
  to_number: string | null;
  language: string;
  started_at: string;
  ended_at: string | null;
  end_reason: string | null;
  turn_count: number;
  config_id: number | null;
  duration_secs: number;
  avg_latency_ms: number | null;
  max_latency_ms: number | null;
  had_interruption: boolean | null;
}

export interface CallDetail {
  id: string;
  session_id: string | null;
  transport: string;
  direction: string | null;
  from_number: string | null;
  to_number: string | null;
  call_id: string | null;
  stream_id: string | null;
  language: string;
  config_id: number | null;
  started_at: string;
  ended_at: string | null;
  end_reason: string | null;
  turn_count: number;
}

export interface Turn {
  turn_index: number;
  user_text: string | null;
  assistant_text: string | null;
  user_spoke_at: string | null;
  bot_spoke_at: string | null;
  completed_at: string | null;
  response_latency_ms: number | null;
  turn_release_ms: number | null;
  stt_ttfb_ms: number | null;
  llm_ttfb_ms: number | null;
  tts_ttfb_ms: number | null;
  interrupted: boolean;
  /** Pipecat's non-overlapping parts of the latency interval, in order. */
  contributions: { key: string; label: string; owner: string; ms: number }[];
  /** Knowledge-base searches made while answering this turn. */
  searches: TurnSearch[];
}

export interface TurnSearch {
  question: string;
  translated_query: string | null;
  translate_ms: number;
  embed_ms: number;
  search_ms: number;
  error: string | null;
  at: string;
  hits: { source: string; document_id?: string; chunk_index: number; score: number; text: string; used: boolean }[];
}

export interface Stats {
  calls_total: number;
  calls_today: number;
  live_calls: number;
  turns_total: number;
  avg_latency_ms: number | null;
  p95_latency_ms: number | null;
  interrupted_turns: number;
  by_transport: { transport: string; calls: number; avg_latency_ms: number | null }[];
  latency_buckets: { label: string; turns: number }[];
}

export interface LanguageOption {
  code: string;
  english_name: string;
  native_name: string;
}

export interface AgentConfig {
  id: number | null;
  system_prompt: string | null;
  temperature: number | null;
  /** Saved language override, or null for the environment default. */
  language: string | null;
  /** Fixed opening line spoken instantly via TTS; null lets the LLM greet. */
  greeting: string | null;
  default_language: string;
  /** The built-in English prompt template with {language_*} placeholders. */
  default_prompt: string;
  agent_name: string;
  languages: LanguageOption[];
}

export interface ConfigVersion {
  id: number;
  system_prompt: string | null;
  temperature: number | null;
  language: string | null;
  greeting: string | null;
  note: string | null;
  created_at: string;
}

export interface KnowledgeStatus {
  qdrant_ok: boolean;
  qdrant_url: string;
  collection: string;
  points: number | null;
  embedding_model: string;
  embedding_key_set: boolean;
  supported_extensions: string[];
}

export interface KnowledgeDocument {
  id: string;
  name: string;
  media_type: string | null;
  size_bytes: number | null;
  chars: number | null;
  chunk_count: number | null;
  embedding_model: string | null;
  status: string;
  error: string | null;
  created_at: string;
}

export interface KnowledgeHit {
  score: number;
  text: string;
  document_id: string;
  document_name: string;
  chunk_index: number;
}

export interface KnowledgeSearch {
  hits: KnowledgeHit[];
  embed_ms: number;
  search_ms: number;
  translate_ms: number;
  /** English form of the question that was embedded, when it was translated. */
  translated_query: string | null;
}
