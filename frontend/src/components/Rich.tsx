import type { Rich } from '../story/lines.ts';

export function RichText({ value }: { value: Rich }) {
  return (
    <>
      {value.map((seg, i) =>
        typeof seg === 'string' ? (
          <span key={i}>{seg}</span>
        ) : (
          <b key={i} className={seg.tone ? `tone-${seg.tone}` : undefined}>
            {seg.strong}
          </b>
        ),
      )}
    </>
  );
}
