/** Shared number/date formatting so every page reads the same. */

export const ms = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : `${Math.round(v).toLocaleString()} ms`;

export const secs = (v: number | null | undefined) => {
  if (v === null || v === undefined) return '—';
  if (v < 60) return `${v}s`;
  return `${Math.floor(v / 60)}m ${v % 60}s`;
};

export const when = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—';

export const clock = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleTimeString(undefined, { hour12: false }) : '';

/** Latency tone: under 1.5 s is fine, under 3 s is slow, above that is a problem. */
export type Tone = 'up' | 'warn' | 'down' | 'idle';
export const latencyTone = (v: number | null | undefined): Tone =>
  v === null || v === undefined ? 'idle' : v < 1500 ? 'up' : v < 3000 ? 'warn' : 'down';
