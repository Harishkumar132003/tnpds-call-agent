import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { Icon, type IconName } from './Icon';

type Tone = 'error' | 'success' | 'info';

export interface Toast {
  id: number;
  tone: Tone;
  title: string;
  /** The detail under the title: what is wrong, and where. */
  body?: string;
  /** One optional action, e.g. jumping to the offending line. */
  action?: { label: string; run: () => void };
}

interface ToastState {
  show: (toast: Omit<Toast, 'id'>) => void;
  dismiss: (id: number) => void;
}

const Ctx = createContext<ToastState | null>(null);

const ICON: Record<Tone, IconName> = {
  error: 'alert',
  success: 'tick',
  info: 'bell',
};

/* An error stays until dismissed. It usually names a line to go and fix, and
 * a message that disappears before it has been read is worse than none. */
const LIFETIME: Record<Tone, number> = {
  error: 0,
  success: 3200,
  info: 5000,
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  const show = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = next.current++;
    /* One at a time per tone: saving repeatedly with the same mistake should
     * replace the message, not stack five copies of it. */
    setToasts((current) => [...current.filter((t) => t.tone !== toast.tone), { ...toast, id }]);
  }, []);

  const value = useMemo(() => ({ show, dismiss }), [show, dismiss]);

  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <ToastRow key={toast.id} toast={toast} onDismiss={() => dismiss(toast.id)} />
        ))}
      </div>
    </Ctx.Provider>
  );
}

function ToastRow({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  useEffect(() => {
    const life = LIFETIME[toast.tone];
    if (!life) return;
    const timer = setTimeout(onDismiss, life);
    return () => clearTimeout(timer);
  }, [toast, onDismiss]);

  return (
    <div className={`toast toast--${toast.tone}`}>
      <span className="toast__glyph">
        <Icon name={ICON[toast.tone]} size={14} />
      </span>
      <div className="toast__text">
        <div className="toast__title">{toast.title}</div>
        {toast.body && <div className="toast__body">{toast.body}</div>}
        {toast.action && (
          <button
            type="button"
            className="toast__action"
            onClick={() => {
              toast.action?.run();
              onDismiss();
            }}
          >
            {toast.action.label}
          </button>
        )}
      </div>
      <button
        type="button"
        className="toast__close"
        aria-label="Dismiss"
        onClick={onDismiss}
      >
        ×
      </button>
    </div>
  );
}

export function useToast(): ToastState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useToast must be used inside a ToastProvider');
  return ctx;
}
