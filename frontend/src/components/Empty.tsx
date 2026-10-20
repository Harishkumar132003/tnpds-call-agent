import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

interface Props {
  icon?: IconName;
  title: string;
  body: ReactNode;
  actions?: ReactNode;
}

export function Empty({ icon = 'layers', title, body, actions }: Props) {
  return (
    <div className="empty">
      <div className="empty__glyph">
        <Icon name={icon} size={18} />
      </div>
      <div className="empty__title">{title}</div>
      <p className="empty__body">{body}</p>
      {actions && <div className="empty__actions">{actions}</div>}
    </div>
  );
}
