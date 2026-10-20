import { latencyTone, ms } from '../format';

/** A latency figure in a status pill: green under 1.5 s, amber under 3 s, red above. */
export function LatencyPill({ value }: { value: number | null | undefined }) {
  const tone = latencyTone(value);
  return <span className={`pill pill--${tone}`}>{ms(value)}</span>;
}
