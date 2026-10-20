import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Icon } from './Icon';

interface Props {
  title: string;
  /** Where this page came from. Rendered before the title, as a real link so
   *  it can be opened in a new tab and does not depend on history depth --
   *  arriving here from a bookmark still has somewhere to go back to. */
  back?: { to: string; label: string };
  actions?: ReactNode;
  onSearch?: (value: string) => void;
  searchPlaceholder?: string;
}

/** Title left, tools right, mirroring the reference dashboard's header. */
export function Topbar({
  title,
  back,
  actions,
  onSearch,
  searchPlaceholder = 'Search…',
}: Props) {
  return (
    <header className="topbar">
      {back && (
        <Link className="btn btn--ghost btn--sm" to={back.to} title={back.label}>
          <Icon name="back" size={14} />
          {back.label}
        </Link>
      )}
      <h1 className="topbar__title">{title}</h1>
      <span className="topbar__spacer" />

      {onSearch && (
        <label className="search">
          <Icon name="search" size={14} className="search__icon" />
          <input
            className="search__input"
            placeholder={searchPlaceholder}
            onChange={(e) => onSearch(e.target.value)}
          />
        </label>
      )}

      {actions}
    </header>
  );
}
