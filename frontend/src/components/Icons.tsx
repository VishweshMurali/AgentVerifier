// Small line icons for the test flow's cards. Plain SVG, drawn in currentColor.
import type { ReactNode } from 'react';

const wrap = (body: ReactNode) => (
  <svg viewBox="0 0 32 32" width="34" height="34" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {body}
  </svg>
);

const ICONS: Record<string, ReactNode> = {
  banking77: wrap(
    <>
      <rect x="4" y="8" width="24" height="16" rx="3" />
      <path d="M4 13h24M9 19h5" />
    </>,
  ),
  clinc150: wrap(
    <>
      <path d="M6 8h20v12H14l-6 5v-5H6z" />
      <path d="M11 14h10" />
    </>,
  ),
  emotion: wrap(
    <>
      <circle cx="16" cy="16" r="11" />
      <path d="M11 19c1.5 2 3 3 5 3s3.5-1 5-3M12 13h.01M20 13h.01" />
    </>,
  ),
  massive_en: wrap(
    <>
      <rect x="12" y="4" width="8" height="15" rx="4" />
      <path d="M7 15a9 9 0 0 0 18 0M16 24v4" />
    </>,
  ),
  agent: wrap(
    <>
      <rect x="7" y="10" width="18" height="14" rx="6" />
      <path d="M16 10V5M12.5 17h.01M19.5 17h.01" />
      <circle cx="16" cy="4.5" r="1.5" />
    </>,
  ),
  prompt: wrap(<path d="M6 9h20v12H15l-5 4v-4H6zM11 15h10" />),
  majority: wrap(<path d="M6 25V15M13 25V7M20 25v-6M27 25v-3M4 25h25" />),
  naive_bayes: wrap(
    <>
      <circle cx="11" cy="16" r="6" />
      <circle cx="21" cy="16" r="6" />
    </>,
  ),
  linear: wrap(<path d="M5 26 27 6M8 12l.01.01M13 20l.01.01M20 9l.01.01M24 19l.01.01" />),
  linear_svm: wrap(<path d="M5 24 26 7M9 27 30 10M2 21 23 4" />),
  random_forest: wrap(<path d="M9 6 4 16h10zM23 6l-5 10h10zM9 16v8M23 16v8M16 12l-4 8h8zM16 20v6" />),
  boosting: wrap(<path d="M5 24h5V18h5v-5h5V8h6" />),
};

export function Icon({ name }: { name: string }) {
  return <span className="icon">{ICONS[name] ?? ICONS.majority}</span>;
}
