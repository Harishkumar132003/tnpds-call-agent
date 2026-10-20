import type { ReactElement } from 'react';

/** 16px stroke icons, drawn inline so there is no icon-font dependency. */

export type IconName =
  | 'database'
  | 'layers'
  | 'sparkle'
  | 'check'
  | 'settings'
  | 'chevron'
  | 'back'
  | 'close'
  | 'plus'
  | 'folder'
  | 'plug'
  | 'alert'
  | 'search'
  | 'calendar'
  | 'filter'
  | 'bell'
  | 'download'
  | 'dots'
  | 'grid'
  | 'shield'
  | 'expand'
  | 'refresh'
  | 'scan'
  | 'tick'
  | 'pencil'
  | 'copy'
  | 'file'
  | 'chat'
  | 'upload'
  | 'table'
  | 'send';

const PATHS: Record<IconName, ReactElement> = {
  database: (
    <>
      <ellipse cx="8" cy="3.75" rx="5.25" ry="2.25" />
      <path d="M2.75 3.75v8.5c0 1.24 2.35 2.25 5.25 2.25s5.25-1.01 5.25-2.25v-8.5" />
      <path d="M2.75 8c0 1.24 2.35 2.25 5.25 2.25S13.25 9.24 13.25 8" />
    </>
  ),
  layers: (
    <>
      <path d="M8 1.75 14.25 5 8 8.25 1.75 5 8 1.75Z" />
      <path d="m2.5 8 5.5 2.9L13.5 8" />
      <path d="m2.5 11 5.5 2.9 5.5-2.9" />
    </>
  ),
  sparkle: (
    <>
      <path d="M8 1.75 9.6 6.4 14.25 8 9.6 9.6 8 14.25 6.4 9.6 1.75 8 6.4 6.4 8 1.75Z" />
    </>
  ),
  check: (
    <>
      <circle cx="8" cy="8" r="6.25" />
      <path d="m5.5 8.2 1.8 1.8 3.3-3.6" />
    </>
  ),
  settings: (
    <>
      <circle cx="8" cy="8" r="2.25" />
      <path d="M12.9 9.8a1.2 1.2 0 0 0 .24 1.32l.05.04a1.45 1.45 0 1 1-2.05 2.05l-.04-.05a1.2 1.2 0 0 0-1.32-.24 1.2 1.2 0 0 0-.73 1.1v.13a1.45 1.45 0 1 1-2.9 0v-.07a1.2 1.2 0 0 0-.79-1.1 1.2 1.2 0 0 0-1.32.24l-.04.05a1.45 1.45 0 1 1-2.05-2.05l.05-.04a1.2 1.2 0 0 0 .24-1.32 1.2 1.2 0 0 0-1.1-.73H1.9a1.45 1.45 0 1 1 0-2.9h.07a1.2 1.2 0 0 0 1.1-.79 1.2 1.2 0 0 0-.24-1.32l-.05-.04A1.45 1.45 0 1 1 4.83 2.1l.04.05a1.2 1.2 0 0 0 1.32.24h.06a1.2 1.2 0 0 0 .73-1.1V1.2a1.45 1.45 0 1 1 2.9 0v.07a1.2 1.2 0 0 0 .73 1.1 1.2 1.2 0 0 0 1.32-.24l.04-.05a1.45 1.45 0 1 1 2.05 2.05l-.05.04a1.2 1.2 0 0 0-.24 1.32v.06a1.2 1.2 0 0 0 1.1.73h.13a1.45 1.45 0 1 1 0 2.9h-.07a1.2 1.2 0 0 0-1.1.73Z" />
    </>
  ),
  chevron: <path d="m4.5 6.25 3.5 3.5 3.5-3.5" />,
  back: <path d="M12.75 8H3.25m0 0 4-4m-4 4 4 4" />,
  close: <path d="M12 4 4 12M4 4l8 8" />,
  plus: <path d="M8 3.25v9.5M3.25 8h9.5" />,
  folder: (
    <path d="M14.25 12.25a1.5 1.5 0 0 1-1.5 1.5H3.25a1.5 1.5 0 0 1-1.5-1.5v-8.5a1.5 1.5 0 0 1 1.5-1.5h3l1.5 2.25h4.5a1.5 1.5 0 0 1 1.5 1.5Z" />
  ),
  plug: (
    <>
      <path d="M6 1.75v4M10 1.75v4" />
      <path d="M4.25 5.75h7.5v2.5a3.75 3.75 0 0 1-7.5 0Z" />
      <path d="M8 12v2.25" />
    </>
  ),
  alert: (
    <>
      <path d="M8 2.75 14.25 13.5H1.75L8 2.75Z" />
      <path d="M8 6.75v3M8 11.75h.01" />
    </>
  ),
  search: (
    <>
      <circle cx="7.25" cy="7.25" r="4.5" />
      <path d="m10.5 10.5 3 3" />
    </>
  ),
  calendar: (
    <>
      <rect x="2.25" y="3.25" width="11.5" height="10.5" rx="1.5" />
      <path d="M2.25 6.25h11.5M5.5 1.75v2.5M10.5 1.75v2.5" />
    </>
  ),
  filter: <path d="M2.25 3.75h11.5l-4.5 5v4.5l-2.5-1.5v-3l-4.5-5Z" />,
  bell: (
    <>
      <path d="M4 6.75a4 4 0 1 1 8 0c0 3 1.25 4 1.25 4H2.75s1.25-1 1.25-4Z" />
      <path d="M6.5 13a1.75 1.75 0 0 0 3 0" />
    </>
  ),
  download: (
    <>
      <path d="M8 2.25v7.5M5 7l3 3 3-3" />
      <path d="M2.75 12.25v.5a1 1 0 0 0 1 1h8.5a1 1 0 0 0 1-1v-.5" />
    </>
  ),
  upload: (
    <>
      <path d="M8 9.75v-7.5M5 5l3-3 3 3" />
      <path d="M2.75 12.25v.5a1 1 0 0 0 1 1h8.5a1 1 0 0 0 1-1v-.5" />
    </>
  ),
  send: <path d="M8 13V3M8 3l-4 4M8 3l4 4" />,
  table: (
    <>
      <rect x="2.25" y="3.25" width="11.5" height="9.5" rx="1.5" />
      <path d="M2.25 7h11.5M6.25 3.25v9.5" />
    </>
  ),
  dots: (
    <>
      <circle cx="3.5" cy="8" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="8" cy="8" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="12.5" cy="8" r="0.9" fill="currentColor" stroke="none" />
    </>
  ),
  grid: (
    <>
      <rect x="2.25" y="2.25" width="5" height="5" rx="1" />
      <rect x="8.75" y="2.25" width="5" height="5" rx="1" />
      <rect x="2.25" y="8.75" width="5" height="5" rx="1" />
      <rect x="8.75" y="8.75" width="5" height="5" rx="1" />
    </>
  ),
  shield: <path d="M8 1.75 13.25 4v4c0 3.2-2.35 5.4-5.25 6.25C5.1 13.4 2.75 11.2 2.75 8V4L8 1.75Z" />,
  expand: <path d="M9.75 2.25h4v4M6.25 13.75h-4v-4M13.75 2.25l-5 5M2.25 13.75l5-5" />,
  refresh: (
    <>
      <path d="M13.25 8a5.25 5.25 0 1 1-1.62-3.79" />
      <path d="M13.25 2.5v3h-3" />
    </>
  ),
  scan: (
    <>
      <path d="M2.75 5.75V4a1.25 1.25 0 0 1 1.25-1.25h1.75" />
      <path d="M10.25 2.75H12A1.25 1.25 0 0 1 13.25 4v1.75" />
      <path d="M13.25 10.25V12A1.25 1.25 0 0 1 12 13.25h-1.75" />
      <path d="M5.75 13.25H4A1.25 1.25 0 0 1 2.75 12v-1.75" />
      <path d="M3.5 8h9" />
    </>
  ),
  /* A bare tick, for use inside an already-circular mark. */
  tick: <path d="m3.75 8.4 2.9 2.9 5.6-6.1" />,
  pencil: (
    <>
      <path d="M11.2 2.55a1.55 1.55 0 0 1 2.25 2.13l-7.4 7.87-3 .9.85-3.05 7.3-7.85Z" />
      <path d="m10.1 3.7 2.2 2.1" />
    </>
  ),
  copy: (
    <>
      <rect x="5.75" y="5.75" width="7.5" height="7.5" rx="1.5" />
      <path d="M10.25 5.75v-1.5a1.5 1.5 0 0 0-1.5-1.5h-4.5a1.5 1.5 0 0 0-1.5 1.5v4.5a1.5 1.5 0 0 0 1.5 1.5h1.5" />
    </>
  ),
  chat: (
    <>
      <path d="M13.25 9.5a1.5 1.5 0 0 1-1.5 1.5H6l-3 2.5V4.25a1.5 1.5 0 0 1 1.5-1.5h7.25a1.5 1.5 0 0 1 1.5 1.5v5.25Z" />
    </>
  ),
  file: (
    <>
      <path d="M9 1.75H4.75a1.5 1.5 0 0 0-1.5 1.5v9.5a1.5 1.5 0 0 0 1.5 1.5h6.5a1.5 1.5 0 0 0 1.5-1.5V5.5L9 1.75Z" />
      <path d="M8.75 2v3.25h3.75" />
    </>
  ),
};

interface Props {
  name: IconName;
  size?: number;
  className?: string;
}

export function Icon({ name, size = 16, className }: Props) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PATHS[name]}
    </svg>
  );
}
