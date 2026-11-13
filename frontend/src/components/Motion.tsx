// Small motion pieces for the story: word-by-word headlines, numbers that count up, the canvas
// background with its readout. All driven by JavaScript so they work on every browser.
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { prefersReducedMotion } from '../motion/flight.ts';
import { startRibbon, type Beats, type Flyer, type Lane } from '../motion/ribbon.ts';

/** A headline split into words that rise out of a mask one after another (see .wm and .w in styles.css). */
export function Words({ text, now = false }: { text: string; now?: boolean }) {
  const words = text.split(/\s+/).filter(Boolean);
  return (
    <>
      {words.map((w, i) => (
        <span key={i}>
          {i > 0 && ' '}
          <span className="wm">
            <span className={`w${now ? ' now' : ''}`} style={{ '--i': i } as CSSProperties}>
              {w}
            </span>
          </span>
        </span>
      ))}
    </>
  );
}

const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

/** A small label that decodes left to right from random characters each time `run` turns true. */
export function Decode({ text, run, className = '', instantWhenStill = false }: { text: string; run: boolean; className?: string; instantWhenStill?: boolean }) {
  const still = prefersReducedMotion();
  const [shown, setShown] = useState(text);
  useEffect(() => {
    // opt-in (mockup): under reduced motion the label is simply shown, never half-decoded
    if (!run || (still && instantWhenStill)) {
      setShown(text);
      return;
    }
    let raf = 0;
    const t0 = performance.now() + 150;
    // reduced motion slows the decode instead of skipping it
    const ms = still ? 2000 : 800;
    const step = (now: number) => {
      const p = Math.min(1, Math.max(0, (now - t0) / ms));
      const keep = Math.floor(p * text.length);
      setShown(
        [...text]
          .map((c, i) => (i < keep || c === ' ' || c === '·' ? c : GLYPHS[Math.floor(Math.random() * GLYPHS.length)]))
          .join(''),
      );
      if (p < 1) raf = requestAnimationFrame(step);
      else setShown(text);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [run, text, still, instantWhenStill]);
  return (
    <span className={className} aria-label={text}>
      <span aria-hidden="true">{shown}</span>
    </span>
  );
}

/** A number that counts up from zero each time `run` turns true. Reduced motion counts slower. */
export function CountUp({ value, decimals = 1, prefix = '', suffix = '', run }: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  run: boolean;
}) {
  const still = prefersReducedMotion();
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!run) return;
    let raf = 0;
    const t0 = performance.now() + 250;
    const ms = still ? 2600 : 1300;
    const step = (now: number) => {
      const p = Math.min(1, Math.max(0, (now - t0) / ms));
      setShown(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [run, value, still]);
  return (
    <>
      {prefix}
      {shown.toFixed(decimals)}
      {suffix}
    </>
  );
}

interface BackgroundProps {
  lanes: Lane[];
  revealDirection: number;
  nCases: number;
  sourceLabel: string;
  sceneCount: number;
  night: boolean;
  flyer: () => Flyer | null;
  /** Scene positions of the ribbon's beats; the 9-scene story's when left out. */
  beats?: Beats;
}

/** The dotted-ribbon canvas, the reveal's corner brackets and the bottom-left readout. */
export function Background({ lanes, revealDirection, nCases, sourceLabel, sceneCount, night, flyer, beats }: BackgroundProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const hudCase = useRef<HTMLElement>(null);
  const hudScene = useRef<HTMLElement>(null);
  const nightRef = useRef(night);
  nightRef.current = night;
  const flyerRef = useRef(flyer);
  flyerRef.current = flyer;

  useEffect(() => {
    if (!canvas.current) return;
    return startRibbon(canvas.current, {
      lanes,
      revealDirection,
      nCases,
      beats,
      still: prefersReducedMotion(),
      isNight: () => nightRef.current,
      flyer: () => flyerRef.current(),
      hudCase: hudCase.current,
      hudScene: hudScene.current,
    });
  }, [lanes, revealDirection, nCases, beats]);

  return (
    <>
      <div className="bg" aria-hidden="true">
        <div className="bg-night" />
        <canvas ref={canvas} />
      </div>
      <div className="frame" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </div>
      <div className="hud" aria-hidden="true">
        <span>{sourceLabel}</span>
        <span>
          Case <b ref={hudCase}>0000</b> / {nCases.toLocaleString('en-US')}
        </span>
        <span>
          Scene <b ref={hudScene}>01</b> / {String(sceneCount).padStart(2, '0')}
        </span>
      </div>
    </>
  );
}
