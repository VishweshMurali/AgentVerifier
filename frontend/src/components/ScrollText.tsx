// Words built by the reader's scrolling (owner-approved sample, samples/words.html):
//   ScrollTiles: rounded blocks fold away and the letters rise out of them (headline, the answer)
//   ScrollWrite: a sentence written letter by letter with a soft cyan tail (the answer's reason)
//   ScrollRise:  words rise out of a mask with a blur (screen headlines)
// Progress comes from where the text sits on the screen, glided slowly, so scrolling back un-builds
// it. One shared animation loop; screen readers get the whole text, never single letters.
import { useEffect, useRef, type CSSProperties, type ReactNode } from 'react';
import { prefersReducedMotion } from '../motion/flight.ts';

type Tick = (now: number) => void;
const ticks = new Set<Tick>();
let raf = 0;
const loop = (now: number) => {
  ticks.forEach((t) => t(now));
  raf = ticks.size ? requestAnimationFrame(loop) : 0;
};
function useTick(tick: Tick) {
  const ref = useRef(tick);
  ref.current = tick;
  useEffect(() => {
    const t: Tick = (now) => ref.current(now);
    ticks.add(t);
    if (!raf) raf = requestAnimationFrame(loop);
    return () => void ticks.delete(t);
  }, []);
}

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const ease = (t: number) => t * t * (3 - 2 * t);
/** Progress of the screen the text sits on: 0 as its top enters at the bottom of the window, 1 as
 *  it settles (top a little above the window's top). Text low on a screen is therefore never left
 *  half-built at rest, and nothing is finished before the reader arrives. */
const placeP = (el: HTMLElement, _from = 1, _to = 0) => {
  const vh = window.innerHeight;
  const sec = (el.closest('section') as HTMLElement | null) ?? el;
  return clamp((vh - sec.getBoundingClientRect().top) / (vh * 0.94), 0, 1);
};
const GLIDE = 0.035;
const SHAPES = ['42% 42% 8% 8% / 50% 50% 8% 8%', '8% 8% 42% 42% / 8% 8% 50% 50%', '10%', '50%', '8% 42% 42% 8% / 8% 50% 50% 8%'];

export function ScrollTiles({ text, className = '', intro = false, as: Tag = 'h2' }: { text: string; className?: string; intro?: boolean; as?: 'h1' | 'h2' }) {
  const box = useRef<HTMLHeadingElement>(null);
  const p = useRef(0);
  const t0 = useRef(performance.now());
  const still = prefersReducedMotion();
  useTick((now) => {
    const el = box.current;
    if (!el) return;
    // the first screen also builds on its own, slowly, so the page is never blank on arrival
    // the first screen's headline is already in place on arrival, so it builds on a slow clock instead
    const target = still ? 1 : intro ? clamp((now - t0.current - 400) / 3600, 0, 1) : placeP(el, 1.0, 0.25);
    p.current += (target - p.current) * GLIDE;
    const chars = el.querySelectorAll<HTMLElement>('.st-ch');
    const n = chars.length, TAIL = 5, head = p.current * (n + TAIL);
    chars.forEach((ch, i) => {
      const e = ease(clamp((head - i) / TAIL, 0, 1));
      const tile = ch.firstElementChild as HTMLElement, letter = ch.lastElementChild as HTMLElement;
      tile.style.transform = `scaleY(${(1 - e).toFixed(3)})`;
      tile.style.opacity = (1 - e).toFixed(3);
      letter.style.opacity = e.toFixed(3);
      letter.style.transform = `translateY(${((1 - e) * 0.35).toFixed(3)}em)`;
    });
  });
  let i = 0;
  const n = text.length;
  return (
    <Tag ref={box} className={`st-tiles ${className}`} aria-label={text}>
      {text.split(' ').map((word, wi) => {
        const out = (
          <span key={wi} className="st-word" aria-hidden="true">
            {[...word].map((c) => {
              const k = i++;
              return (
                <span key={k} className="st-ch">
                  <i style={{ borderRadius: SHAPES[(k * 7 + 3) % SHAPES.length] }} />
                  <span style={{ '--i': k, '--n': n } as CSSProperties}>{c}</span>
                </span>
              );
            })}
          </span>
        );
        i++;
        return out;
      })}
    </Tag>
  );
}

export type Part = string | { b: string };

export function ScrollWrite({ parts, className = '' }: { parts: Part[]; className?: string }) {
  const box = useRef<HTMLParagraphElement>(null);
  const p = useRef(0);
  const still = prefersReducedMotion();
  useTick(() => {
    const el = box.current;
    if (!el) return;
    p.current += ((still ? 1 : placeP(el, 0.95, 0.35)) - p.current) * GLIDE;
    const chars = el.querySelectorAll<HTMLElement>('.sw-ch');
    const n = chars.length, TAIL = 30, head = p.current * (n + TAIL);
    chars.forEach((s, i) => {
      const e = ease(clamp((head - i) / TAIL, 0, 1));
      s.style.opacity = (0.05 + 0.95 * e).toFixed(3);
      s.style.transform = `translateY(${((1 - e) * 0.5).toFixed(3)}em) rotate(${((1 - e) * 10).toFixed(2)}deg)`;
      s.style.color = e < 0.98 ? `rgba(34,227,240,${(0.3 + 0.7 * e).toFixed(2)})` : '';
    });
  });
  const plain = parts.map((x) => (typeof x === 'string' ? x : x.b)).join('');
  let key = 0;
  const letters = (text: string): ReactNode[] =>
    text.split(/(\s+)/).map((w) =>
      /^\s+$/.test(w) ? (
        <span key={key++} className="sw-ch">
          {w}
        </span>
      ) : (
        <span key={key++} className="sw-w">
          {[...w].map((c) => (
            <span key={key++} className="sw-ch">
              {c}
            </span>
          ))}
        </span>
      ),
    );
  return (
    <p ref={box} className={`sw ${className}`} aria-label={plain}>
      <span aria-hidden="true">{parts.map((x, k) => (typeof x === 'string' ? <span key={`p${k}`}>{letters(x)}</span> : <b key={`p${k}`}>{letters(x.b)}</b>))}</span>
    </p>
  );
}

export function ScrollRise({ text, className = '' }: { text: string; className?: string }) {
  const box = useRef<HTMLHeadingElement>(null);
  const p = useRef(0);
  const still = prefersReducedMotion();
  useTick(() => {
    const el = box.current;
    if (!el) return;
    p.current += ((still ? 1 : placeP(el, 1.0, 0.3)) - p.current) * GLIDE;
    const words = el.querySelectorAll<HTMLElement>('.sr-w');
    const n = words.length, TAIL = 1.4, head = p.current * (n + TAIL);
    words.forEach((w, i) => {
      const e = ease(clamp((head - i) / TAIL, 0, 1));
      w.style.opacity = e.toFixed(3);
      w.style.transform = `translateY(${((1 - e) * 110).toFixed(1)}%) rotate(${((1 - e) * 6).toFixed(2)}deg)`;
      w.style.filter = e > 0.99 ? '' : `blur(${((1 - e) * 8).toFixed(1)}px)`;
    });
  });
  return (
    <h2 ref={box} className={`sr ${className}`} aria-label={text}>
      {text.split(' ').map((w, i) => (
        <span key={i} aria-hidden="true">
          {i > 0 && ' '}
          <span className="sr-m">
            <span className="sr-w">{w}</span>
          </span>
        </span>
      ))}
    </h2>
  );
}

/**
 * Cards and panels built by scrolling: each direct child (or each element matching `items`) rises,
 * sharpens and fades in one after another as the group travels up the screen, and un-builds when
 * the reader scrolls back. Uses the separate translate/scale properties, so hover effects keep
 * their own transform.
 */
export function Reveal({
  children,
  className = '',
  items,
  from = 1.0,
  to = 0.45,
  intro = false,
}: {
  children: ReactNode;
  className?: string;
  items?: string;
  from?: number;
  to?: number;
  intro?: boolean;
}) {
  const box = useRef<HTMLDivElement>(null);
  const p = useRef(0);
  const t0 = useRef(performance.now());
  const still = prefersReducedMotion();
  useTick((now) => {
    const el = box.current;
    if (!el) return;
    let target = still ? 1 : placeP(el, from, to);
    // on the first screen the group is already in place on arrival: it waits for the headline,
    // then builds on a slow clock; after that, scrolling drives it as usual
    if (intro && !still) target = Math.min(target, clamp((now - t0.current - 2400) / 2600, 0, 1));
    p.current += (target - p.current) * GLIDE;
    const kids = items ? el.querySelectorAll<HTMLElement>(items) : (el.children as unknown as NodeListOf<HTMLElement>);
    const n = kids.length, TAIL = 1.6, head = p.current * (n + TAIL);
    for (let i = 0; i < n; i++) {
      const k = kids[i];
      const e = ease(clamp((head - i) / TAIL, 0, 1));
      k.style.opacity = e.toFixed(3);
      k.style.translate = `0 ${((1 - e) * 46).toFixed(1)}px`;
      k.style.scale = (0.94 + 0.06 * e).toFixed(3);
      k.style.filter = e > 0.99 ? '' : `blur(${((1 - e) * 7).toFixed(1)}px)`;
      k.style.setProperty('--e', e.toFixed(3));
    }
  });
  return (
    <div ref={box} className={className}>
      {children}
    </div>
  );
}
