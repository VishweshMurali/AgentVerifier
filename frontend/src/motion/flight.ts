// Robot flight, ported from design/frontend/story-reference.dc.html (startRobot).
// A spring follows a smooth path through the scene stops, so the guide glides, trails behind the
// scroll, leans into turns and settles with a little give. It parks right on odd scenes and left
// on even ones. It publishes its screen position so the background can part around it.
import type { Flyer } from './ribbon.ts';
import { scenePosition } from './ribbon.ts';

export interface FlightParts {
  robot: HTMLElement;
  fly: HTMLElement;
  ground: HTMLElement;
  bank: HTMLElement | SVGElement;
  gaze: HTMLElement | SVGElement;
  say: HTMLElement;
}

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const mix = (a: number, b: number, f: number) => a + (b - a) * f;
const ease = (x: number) => {
  const s = clamp(x, 0, 1);
  return s * s * s * (s * (s * 6 - 15) + 10);
};

/** Phones and narrow windows: the robot docks small in the corner and stops flying. */
export const DOCK_WIDTH = 900;

export function startFlight(p: FlightParts, still: boolean, publish: (f: Flyer | null) => void): () => void {
  const secs = Array.from(document.querySelectorAll<HTMLElement>('section.scene'));
  let sx = 0, sy = 0, vx = 0, vy = 0, lean = 0, gaze = 0, bub = 0, t = 0, last = 0, raf = 0;
  let ready = false, docked = false;
  const all = [p.robot, p.fly, p.ground, p.bank, p.gaze];

  const frame = (now: number) => {
    const dt = Math.min(0.033, (now - last) / 1000 || 0.016);
    last = now;
    t += dt;
    raf = requestAnimationFrame(frame);
    const W = window.innerWidth, H = window.innerHeight;
    if (W <= DOCK_WIDTH) {
      if (!docked) {
        all.forEach((el) => (el.style.transform = ''));
        p.ground.style.opacity = '';
        p.say.style.opacity = '';
        p.say.style.visibility = '';
        docked = true;
        ready = false;
        publish(null);
      }
      return;
    }
    docked = false;
    const pos = scenePosition(secs);
    const i = Math.floor(pos);
    const f = pos - i;
    const span = i < secs.length - 1;
    const xR = W - 380, xL = 100, yPark = H - 360;
    const stop = (n: number) => (n % 2 === 1 ? xL : xR);
    const e = ease((f - 0.15) / 0.7);
    // idle drift keeps going with reduced motion, only slower (SPEC: slows instead of stopping)
    const ti = still ? t * 0.35 : t;
    const tx =
      (span ? mix(stop(i), stop(i + 1), e) : stop(i)) +
      (span ? Math.sin(2 * Math.PI * e) * 70 * (i % 2 ? 1 : -1) : 0) +
      (36 * Math.sin(ti * 0.7) + 14 * Math.sin(ti * 1.9));
    const lift = H * (0.42 + 0.2 * Math.sin((i + 1) * 2.3));
    const ty = Math.max(70, yPark - (span ? Math.sin(Math.PI * e) * lift : 0) + (26 * Math.sin(ti * 0.9 + 1) + 10 * Math.sin(ti * 2.3)));
    // power-on: the guide launches up from below the screen and settles into its first stop
    if (!ready) {
      sx = tx;
      sy = still ? ty : ty + H * 0.55;
      vx = 0;
      vy = still ? 0 : -900;
      ready = true;
    }
    const k = still ? 400 : 26;
    const c = 2 * (still ? 1 : 0.55) * Math.sqrt(k);
    const h = dt / 2;
    for (let n = 0; n < 2; n++) {
      vx += (k * (tx - sx) - c * vx) * h;
      sx += vx * h;
      vy += (k * (ty - sy) - c * vy) * h;
      sy += vy * h;
    }
    const sm = 1 - Math.exp(-dt * 8);
    lean += (clamp(vx * 0.022, -28, 28) - lean) * sm;
    gaze += (clamp(vx * 0.012, -8, 8) - gaze) * sm;
    const alt = clamp((yPark - sy) / (H * 0.7), 0, 1);
    p.robot.style.transform = `translate3d(${sx.toFixed(1)}px,0,0)`;
    p.fly.style.transform = `translate3d(0,${sy.toFixed(1)}px,0)`;
    p.bank.style.transform = `rotate(${lean.toFixed(2)}deg) scale(${(1 - 0.15 * alt).toFixed(3)})`;
    p.gaze.style.transform = `translateX(${gaze.toFixed(2)}px)`;
    p.ground.style.transform = `translateY(${H - 40}px) scale(${(1 - 0.55 * alt).toFixed(3)})`;
    p.ground.style.opacity = (1 - 0.75 * alt).toFixed(3);
    publish({ x: sx + 130, y: sy + 140, vx, vy });
    // the speech bubble shows while the guide is settled at a scene, and never hides a focused control
    const parked = Math.abs(pos - Math.round(pos)) < 0.2 && Math.hypot(vx, vy) < 260;
    const focused = p.say.contains(document.activeElement);
    bub += ((parked || focused ? 1 : 0) - bub) * (1 - Math.exp(-dt * 9));
    if (focused) bub = Math.max(bub, 0.5);
    p.say.style.opacity = bub.toFixed(3);
    p.say.style.visibility = bub < 0.03 ? 'hidden' : 'visible';
  };
  last = performance.now();
  raf = requestAnimationFrame(frame);
  return () => {
    cancelAnimationFrame(raf);
    publish(null);
  };
}

/** Eases mouse-wheel scrolling so the flight and the reveals glide instead of stepping. */
export function startSmoothWheel(): () => void {
  const el = document.documentElement;
  let cur = 0, target = 0, lastSet = 0, raf = 0, last = 0;
  const now = () => window.scrollY || el.scrollTop || 0;
  const max = () => Math.max(0, el.scrollHeight - window.innerHeight);
  const sync = () => {
    cur = target = lastSet = now();
  };
  const tick = (ts: number) => {
    const dt = Math.min(0.05, (ts - last) / 1000 || 0.016);
    last = ts;
    cur += (target - cur) * (1 - Math.exp(-dt * 9));
    if (Math.abs(target - cur) < 0.5) cur = target;
    lastSet = cur;
    window.scrollTo(0, cur);
    raf = cur === target ? 0 : requestAnimationFrame(tick);
  };
  const go = () => {
    if (!raf) {
      last = performance.now();
      raf = requestAnimationFrame(tick);
    }
  };
  const onWheel = (e: WheelEvent) => {
    if (e.ctrlKey || max() < 40 || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    const tgt = e.target as Element | null;
    if (tgt?.closest?.('input,textarea,select,.checks,.scroll-y,.all-kinds,.drawer,.chat')) return;
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 32;
    else if (e.deltaMode === 2) dy *= window.innerHeight;
    e.preventDefault();
    if (!raf) sync();
    target = clamp(target + dy, 0, max());
    go();
  };
  // scrolls we did not cause (scrollbar drag, keys, touch) take over from the easing
  const onScroll = () => {
    if (Math.abs(now() - lastSet) > 3) {
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
      sync();
    }
  };
  // scene links glide too
  const onClick = (e: MouseEvent) => {
    const a = (e.target as Element | null)?.closest?.('a[href^="#"]');
    if (!a || max() < 40) return;
    const sec = document.getElementById((a.getAttribute('href') ?? '').slice(1));
    if (!sec) return;
    e.preventDefault();
    if (!raf) sync();
    target = clamp(sec.getBoundingClientRect().top + now(), 0, max());
    go();
    // do what the browser does for a jump link: update the address and move focus to the scene
    if (location.hash !== `#${sec.id}`) history.pushState(null, '', `#${sec.id}`);
    sec.focus({ preventScroll: true });
  };
  sync();
  window.addEventListener('wheel', onWheel, { passive: false });
  window.addEventListener('scroll', onScroll, { passive: true });
  document.addEventListener('click', onClick);
  return () => {
    if (raf) cancelAnimationFrame(raf);
    window.removeEventListener('wheel', onWheel);
    window.removeEventListener('scroll', onScroll);
    document.removeEventListener('click', onClick);
  };
}

export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}
