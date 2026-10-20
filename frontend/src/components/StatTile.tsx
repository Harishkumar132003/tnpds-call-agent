import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

interface Props {
  label: string;
  icon: IconName;
  value: string | number;
  /** The qualifier under the number: a pill plus a plain phrase. */
  pill?: { tone: 'up' | 'down' | 'warn' | 'info' | 'idle' | 'violet'; text: string };
  note?: ReactNode;
}

/**
 * A stat tile, not a chart: a single magnitude with one qualifier. The number
 * carries the weight; the pill states what it means rather than decorating it.
 */
export function StatTile({ label, icon, value, pill, note }: Props) {
  return (
    <section className="card">
      <div className="card__head card__head--bare">
        <span className="card__label">
          <Icon name={icon} size={13} />
          {label}
        </span>
        <span className="card__tools">
          <Icon name="dots" size={14} />
        </span>
      </div>
      <div className="card__body">
        <div className="stat__value">{value}</div>
        <div className="stat__foot">
          {pill && <span className={`pill pill--${pill.tone}`}>{pill.text}</span>}
          {note}
        </div>
      </div>
    </section>
  );
}
