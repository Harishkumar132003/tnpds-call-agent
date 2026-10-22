import { NavLink } from 'react-router-dom';
import { Icon } from './Icon';

interface Props {
  liveCalls: number;
  backendUp: boolean;
  language: string;
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  isActive ? 'nav__item is-active' : 'nav__item';

/** The collapsible rail from the Cube product, with this console's sections. */
export function Sidebar({ liveCalls, backendUp, language }: Props) {
  return (
    <aside className="rail">
      <div className="ws">
        <span className="ws__glyph">
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M8 1.5 14 5v6l-6 3.5L2 11V5l6-3.5Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
            <path d="M8 8.2 14 5M8 8.2v6.3M8 8.2 2 5" stroke="currentColor" strokeWidth="1.4" />
          </svg>
        </span>
        <span className="ws__text">
          <span className="ws__name">Voice agent</span>
          <span className="ws__sub">Agent · {language}</span>
        </span>
      </div>

      <nav className="nav">
        <div className="nav__group">Main menu</div>
        <NavLink to="/" end className={navClass}>
          <Icon name="grid" className="nav__icon" />
          <span className="nav__label">Dashboard</span>
        </NavLink>
        <NavLink to="/calls" className={navClass}>
          <Icon name="chat" className="nav__icon" />
          <span className="nav__label">Calls</span>
          <span className="nav__spacer" />
          {liveCalls > 0 && <span className="nav__tag">{liveCalls} live</span>}
        </NavLink>

        <div className="nav__group">Agent</div>
        <NavLink to="/config" className={navClass}>
          <Icon name="pencil" className="nav__icon" />
          <span className="nav__label">Configuration</span>
        </NavLink>
        <NavLink to="/models" className={navClass}>
          <Icon name="layers" className="nav__icon" />
          <span className="nav__label">Models</span>
        </NavLink>
        <NavLink to="/knowledge" className={navClass}>
          <Icon name="file" className="nav__icon" />
          <span className="nav__label">Knowledge</span>
        </NavLink>
        <NavLink to="/pronunciation" className={navClass}>
          <Icon name="sparkle" className="nav__icon" />
          <span className="nav__label">Pronunciation</span>
        </NavLink>
        <NavLink to="/test-call" className={navClass}>
          <Icon name="send" className="nav__icon" />
          <span className="nav__label">Test call</span>
        </NavLink>
      </nav>

      <div className="who">
        <span className="who__avatar">A</span>
        <span className="who__text">
          <span className="who__name">Agent</span>
          <span className="who__sub">{backendUp ? 'tracking connected' : 'backend unreachable'}</span>
        </span>
      </div>
    </aside>
  );
}
