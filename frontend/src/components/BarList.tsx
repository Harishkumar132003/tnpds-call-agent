interface Item {
  name: string;
  value: number;
  meta?: string;
  soft?: boolean;
}

interface Props {
  items: Item[];
  /** Unit word for the direct label, e.g. "relations". */
  unit?: string;
}

/**
 * Single-hue magnitude bars. One series, so no legend is needed -- the card
 * title names it -- and every bar is directly labelled rather than relying on
 * an axis. Bars share one scale, anchored at zero.
 */
export function BarList({ items, unit }: Props) {
  const max = Math.max(1, ...items.map((i) => i.value));

  return (
    <div className="bars">
      {items.map((item) => (
        <div key={item.name}>
          <div className="bar__top">
            <span className="bar__name mono">{item.name}</span>
            <span className="bar__meta">
              {item.value.toLocaleString()}
              {unit ? ` ${unit}` : ''}
              {item.meta ? ` · ${item.meta}` : ''}
            </span>
          </div>
          <div className="bar__track">
            <div
              className={item.soft ? 'bar__fill bar__fill--soft' : 'bar__fill'}
              style={{ width: `${Math.max(2, (item.value / max) * 100)}%` }}
              title={`${item.name}: ${item.value.toLocaleString()}${unit ? ` ${unit}` : ''}`}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
