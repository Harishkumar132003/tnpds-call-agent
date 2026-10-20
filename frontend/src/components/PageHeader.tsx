import type { ReactNode } from 'react';

interface Props {
  title: string;
  lede?: ReactNode;
  actions?: ReactNode;
}

/** The full-bleed page header, matching the dashboard topbar. */
export function PageHeader({ title, lede, actions }: Props) {
  return (
    <header className="topbar">
      <div>
        <h1 className="topbar__title">{title}</h1>
        {lede && <p className="topbar__lede">{lede}</p>}
      </div>
      <span className="topbar__spacer" />
      {actions}
    </header>
  );
}
