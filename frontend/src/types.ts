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
  /** LLM_TOPIC_CONTEXT: was the caller's turn a new question or a follow-up? */
  topic?: { decision: 'NEW' | 'FOLLOW_UP'; reason: string | null; text: string | null } | null;
}

export interface TurnSearch {
  question: string;
  translated_query: string | null;
  translate_ms: number;
  embed_ms: number;
  search_ms: number;
  error: string | null;
  at: string;
  /** KNOWLEDGE_QUERY_REWRITE: the context-aware English query that was searched. */
  rewritten_query?: string | null;
  rewrite_ms?: number;
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
  /** Idle handling overrides; null = environment / per-language default. */
  idle_prompt_secs: number | null;
  idle_check_line: string | null;
  idle_goodbye_line: string | null;
  /** Per-step LLM rules when the search tool is on; null = built-in text.
   *  LLM 1 decides and searches; LLM 2 answers from the search results. */
  router_prompt: string | null;
  router_prompt_default: string;
  answer_prompt: string | null;
  answer_prompt_default: string;
  /** LLM_TOPIC_CONTEXT: the NEW/FOLLOW_UP topic classifier's prompt and whether the flag is on. */
  topic_prompt: string | null;
  topic_prompt_default: string;
  topic_context_on: boolean;
  /** KNOWLEDGE_QUERY_REWRITE: saved rewrite instructions (null = built-in) and whether the flag is on. */
  rewrite_prompt: string | null;
  rewrite_prompt_default: string;
  query_rewrite_on: boolean;
  knowledge_on: boolean;
  /** Pronunciation dictionary as "word = spoken form" lines, and its Sarvam id. */
  pronunciations: string;
  tts_dict_id: string | null;
  /** Per-step Sarvam model overrides; null = environment default (see model_defaults). */
  stt_model: string | null;
  llm_model: string | null;
  tts_model: string | null;
  tts_speaker: string | null;
  translate_model: string | null;
  embedding_model: string | null;
  model_defaults: { stt: string; llm: string; tts: string; tts_speaker: string; translate: string; embedding: string };
  idle_defaults: { prompt_secs: number; check_lines: Record<string, string>; goodbye_lines: Record<string, string> };
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
  idle_prompt_secs: number | null;
  idle_check_line: string | null;
  idle_goodbye_line: string | null;
  router_prompt: string | null;
  answer_prompt: string | null;
  topic_prompt?: string | null;
  rewrite_prompt?: string | null;
  pronunciations: { word: string; spoken: string }[] | null;
  tts_dict_id: string | null;
  stt_model: string | null;
  llm_model: string | null;
  tts_model: string | null;
  tts_speaker: string | null;
  translate_model: string | null;
  embedding_model: string | null;
  note: string | null;
  created_at: string;
}

export interface ModelOption {
  id: string;
  label: string;
  note: string;
  recommended: boolean;
}

export interface VoiceOption {
  id: string;
  gender: string;
  /** Sarvam's production tiers by error rate: 1 best, 2 good, 3 unranked. */
  tier: number;
  /** Persona style of a bulbul:v4-flash voice (narration, customer...); null for v3. */
  style: string | null;
  /** Named by Sarvam's docs for the catalogue's language. */
  recommended: boolean;
}

/** Sarvam models per pipeline step that support one language. */
export interface ModelCatalogue {
  language: string;
  stt: ModelOption[];
  llm: ModelOption[];
  tts: ModelOption[];
  translate: ModelOption[];
  embedding: ModelOption[];
  voices: Record<string, VoiceOption[]>;
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
  rewritten_query?: string | null;
  rewrite_ms?: number;
}

export interface TestCallInfo {
  bot_url: string;
  reachable: boolean;
  /** null when the bot is unreachable; false when BOT_TRANSPORT disables WebRTC */
  webrtc: boolean | null;
}

// ---- evaluation ----

export interface EvalQuestion {
  id: number;
  question_en: string;
  question_ta: string | null;
  expected_source: string;
  note: string | null;
  enabled: boolean;
  origin: string;
  created_at: string;
  updated_at: string;
}

/** How an evaluation batch searched the knowledge base. */
export type SearchType = 'semantic' | 'keyword' | 'hybrid';

export const SEARCH_TYPE_LABEL: Record<SearchType, string> = {
  semantic: 'semantic',
  keyword: 'keyword (BM25)',
  hybrid: 'hybrid (RRF)',
};

export interface EvalBatch {
  id: string;
  started_at: string;
  finished_at: string | null;
  status: string;
  total: number;
  done: number;
  config_id: number | null;
  collection: string | null;
  note: string | null;
  error: string | null;
  concurrency: number;
  force_search: boolean;
  search_type: SearchType;
  results: number;
  retrieved: number;
  sent: number;
  rank1: number;
  errors: number;
  avg_ms: number;
}

/** Rewrite-prompt evaluation: a fixed question set rewritten with a draft prompt. */
export interface RewriteQuestion {
  id: number;
  question_ta: string;
  question_en: string | null;
  expected_source: string;
  source_question_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface RewriteRun {
  id: string;
  started_at: string;
  finished_at: string | null;
  status: string;
  prompt: string;
  prompt_source: 'draft' | 'config' | 'default';
  config_id: number | null;
  model: string | null;
  collection: string | null;
  search_type: SearchType;
  total: number;
  error: string | null;
  results: number;
  retrieved: number;
  sent: number;
  rank1: number;
  errors: number;
  /** How many rewritten queries the reviewer ticked as correct. */
  judged_ok: number;
  avg_rewrite_ms: number;
}

export interface RewriteResult {
  id: number;
  run_id: string;
  question_id: number | null;
  question_ta: string;
  expected_source: string;
  rewritten_query: string | null;
  hits: EvalHit[];
  expected_rank: number | null;
  expected_sent: boolean;
  rewrite_ms: number | null;
  ms: number | null;
  error: string | null;
  at: string;
  judged_ok: boolean;
}

export interface EvalHit {
  source: string | null;
  document_id: string | null;
  chunk_index: number | null;
  score: number | null;
  used: boolean;
}

export interface EvalResult {
  id: number;
  question_id: number | null;
  question_ta: string | null;
  question_en: string | null;
  expected_source: string | null;
  rewritten_query: string | null;
  hits: EvalHit[] | null;
  expected_rank: number | null;
  expected_sent: boolean;
  answer: string | null;
  error: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  ms: number | null;
  at: string;
}
