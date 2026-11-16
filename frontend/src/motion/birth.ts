// The sphere's life cycle for the opening screens (owner brief 2026-11-15), drawn with the page's
// own graph look: the same glowing nodes and colours, so the message graph is BORN from a sphere.
//   1 welcome  a soft ice-blue ball of particles: a dense bright core, dots thinning outward, no outline
//   2 agent    it charges: grows, the core brightens, jets of sparks shoot out of the core
//   2 -> 3     overcharge: it squeezes tight, shakes, sparks are sucked back in, the core goes white-hot
//   3 -> 4     a blast: flash, shockwave, rays, thousands of sparks fly out, and the graph's dots swirl
//              (with streaks) to their places while colour blooms in as each lands
// Everything is a function of the scroll position `q` (the page's scene position: 0 = welcome
// aligned, 1 = the agent screen, 2 = the kinds screen), so stopping holds the moment and scrolling
// back rebuilds the sphere. Spark jets and the shake run on time and never move a graph dot.
import { flag } from '../flags.ts';
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const mix = (a: number, b: number, f: number) => a + (b - a) * f;
const sstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
/** gentler: starts and lands softly, with a slight lean toward fast-then-slow */
const glide = (t: number) => 0.5 * t * t * (3 - 2 * t) + 0.5 * (1 - Math.pow(1 - t, 2));
const hash = (n: number) => {
  const s = Math.sin(n * 91.7 + 17.3) * 43758.5453;
  return s - Math.floor(s);
};

/** Every number the opening can be tuned by. The ?tune panel moves them live; `burstAt`, `burstSpan`
 *  and `chargeEnd` are scene positions (0 = welcome, 1 = the agent screen, 2 = the kinds screen). */
export const TUNE_DEFAULTS = {
  sphereSize: 0.71, // radius, in the graph's own units
  particles: 5000, // extra fine particles that make the ball (and fly out in the blast)
  density: 2.75, // higher = dots packed closer to the core
  dotSize: 0.35, // graph dot size inside the sphere (1 = the graph's normal node)
  coreSize: 0.24, // the bright core, as a share of the radius
  coreGlow: 0.2, // how saturated the core is
  chargeEnd: 2.2, // where charging is complete
  chargeGrow: 0.7, // how much bigger the charged sphere is
  neurons: 32, // most spark jets firing at full charge
  neuronSpeed: 3, // firing speed
  overcharge: 0.55, // how violent the build-up before the blast is (0 = none)
  squeeze: 0.6, // how far the ball squeezes in just before it blasts
  burstAt: 1.46, // where the blast starts
  burstSpan: 1.6, // how long, in scroll, until every dot has landed in the graph
  burstReach: 0.2, // how far the dots and sparks fly out before they land
  flash: 0.8, // brightness of the flash
  rays: 0.05, // the streaking rays of the blast
  swirl: 2.5, // how much the dots spiral on their way to the graph
  trail: 0, // length of the streaks behind the travelling dots
};
/** People who ask their system for less motion get the same show slowed down (SPEC: reduced motion slows
 *  the motion, it does not stop it): no shake, cursor stirring or scroll twist, a slower spin and slower jets. */
export const REDUCED = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches && !flag('tune') && !flag('motion'); // ?motion forces the full show for review
/** 0.15..1: how much of the particle show this device can afford. The page lowers it when frames run long
 *  and raises it again when there is room, so a slow phone gets fewer particles, never a stutter. */
export const PERF = { q: 1 };
export type Tune = typeof TUNE_DEFAULTS;
export const TUNE: Tune = { ...TUNE_DEFAULTS };
export const ICE = [186, 230, 253];
export const ICE_DAY = [120, 190, 255];

export interface BirthState {
  /** 0..1 while the sphere charges (welcome to the agent screen) */
  charge: number;
  /** 0..1 overcharge build-up just before the blast */
  over: number;
  /** 0..1 blast-and-form progress (agent screen to the kinds screen) */
  e: number;
  /** the core: 1 while the sphere is whole */
  alive: number;
  /** the flash's brightness */
  flash: number;
  /** 0..1 how fast the page is being scrolled right now: the galaxy twists and the jets quicken */
  energy: number;
}

export function birthState(q: number): BirthState {
  const b = TUNE.burstAt;
  return {
    charge: sstep(0.1, Math.max(0.2, TUNE.chargeEnd), q),
    over: sstep(b - 0.5, b - 0.02, q) * TUNE.overcharge,
    e: clamp((q - b) / TUNE.burstSpan, 0, 1),
    alive: 1 - sstep(b - 0.01, b + 0.12, q),
    flash: sstep(b - 0.03, b + 0.01, q) * (1 - sstep(b + 0.01, b + 0.2, q)) * TUNE.flash,
    energy: 0,
  };
}

export interface BirthPoints {
  /** where point i is at this moment, in graph units (home = its place in the graph) */
  place(i: number, hx: number, hy: number, hz: number, B: BirthState, out: Float32Array, k: number): void;
  /** 0..1: how far point i has travelled from the sphere to its place */
  arrived(i: number, B: BirthState): number;
}

export function makeBirthPoints(n: number): BirthPoints {
  const u = new Float32Array(n * 3), r = new Float32Array(n), jit = new Float32Array(n), bd = new Float32Array(n), dir = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const th = hash(i * 3 + 1) * Math.PI * 2, z = hash(i * 3 + 2) * 2 - 1, s = Math.sqrt(1 - z * z);
    u[i * 3] = Math.cos(th) * s;
    u[i * 3 + 1] = Math.sin(th) * s;
    u[i * 3 + 2] = z;
    r[i] = hash(i * 3 + 3); // raised to `density` when placed: dense at the core, thinning outward
    jit[i] = hash(i * 3 + 4) * 0.22;
    bd[i] = 0.35 + hash(i * 3 + 5) * 1.5;
    dir[i] = hash(i * 3 + 6) < 0.5 ? -1 : 1;
  }
  const arrived = (i: number, B: BirthState) => glide(clamp((B.e - jit[i]) / (1 - jit[i]), 0, 1));
  return {
    arrived,
    place(i, hx, hy, hz, B, out, k) {
      const grow = (1 + TUNE.chargeGrow * B.charge) * (1 - TUNE.squeeze * B.over) * TUNE.sphereSize * Math.pow(r[i], TUNE.density);
      const ei = arrived(i, B), lift = Math.sin(Math.PI * ei) * bd[i] * TUNE.burstReach; // out, then in to its place
      let x = mix(u[i * 3] * grow, hx, ei) + u[i * 3] * lift;
      const y = mix(u[i * 3 + 1] * grow, hy, ei) + u[i * 3 + 1] * lift;
      let z = mix(u[i * 3 + 2] * grow, hz, ei) + u[i * 3 + 2] * lift;
      // spiral: turn about the vertical axis while travelling, none at the start or the end
      const a = TUNE.swirl * dir[i] * 4 * ei * (1 - ei) * (B.e > 0 ? 1 : 0), ca = Math.cos(a), sa = Math.sin(a);
      const rx = x * ca + z * sa;
      z = -x * sa + z * ca;
      x = rx;
      out[k] = x;
      out[k + 1] = y;
      out[k + 2] = z;
    },
  };
}

/** The colour a point wears while it travels: ice-blue in the sphere, its own colour on arrival. */
export function travelColour(rgb: number[], ei: number, day = false): string {
  const f = sstep(0.05, 0.75, ei), ice = day ? ICE_DAY : ICE; // on a pale page the dots start as a saturated blue
  return `${Math.round(mix(ice[0], rgb[0], f))},${Math.round(mix(ice[1], rgb[1], f))},${Math.round(mix(ice[2], rgb[2], f))}`;
}

type Tone = (c: string) => string;
const SKY = '56,189,248', CYAN = '34,227,240';

/** The ball's radius on screen right now: grows as it charges, squeezes in as it overcharges. */
const radius = (R: number, B: BirthState) => R * (1 + TUNE.chargeGrow * B.charge) * (1 - TUNE.squeeze * B.over);
/** 0..1 pulse that beats faster and faster as the overcharge builds. */
const beat = (t: number, B: BirthState) => 0.5 + 0.5 * Math.sin(t * (2.2 + 16 * B.over * B.over));

/** Soft halo and a big saturated core that goes white-hot as it overcharges: drawn before the dots. */
export function drawBody(ctx: CanvasRenderingContext2D, x: number, y: number, R: number, B: BirthState, t: number, toneIn: Tone, day: boolean): void {
  if (B.alive < 0.01) return;
  const tone: Tone = day ? (c) => c : toneIn; // on a pale page the body keeps its bright blues
  const rad = radius(R, B), pulse = beat(t, B), a = B.alive;
  const hr = rad * (1.9 + 0.5 * B.charge + 0.5 * B.over * pulse);
  const halo = ctx.createRadialGradient(x, y, rad * 0.2, x, y, hr);
  halo.addColorStop(0, `rgba(${tone(SKY)},${(0.1 + 0.08 * B.charge + 0.12 * B.over) * a})`);
  halo.addColorStop(1, `rgba(${tone(SKY)},0)`);
  ctx.globalAlpha = 1;
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, hr, 0, Math.PI * 2);
  ctx.fill();
  // the core: a large saturated blue disc with a bright centre, hotter and flickering as it overcharges
  const cr = rad * (TUNE.coreSize + 0.14 * B.charge + 0.16 * B.over * (0.8 + 0.4 * pulse)), core = ctx.createRadialGradient(x, y, 0, x, y, cr * 1.7);
  core.addColorStop(0, `rgba(255,255,255,${0.95 * a})`);
  core.addColorStop(0.16 + 0.2 * B.over, `rgba(230,248,255,${0.8 * a})`);
  core.addColorStop(0.45, `rgba(${tone('40,160,245')},${(TUNE.coreGlow + 0.25 * B.charge) * a})`);
  core.addColorStop(0.78, `rgba(${tone('30,120,225')},${0.28 * a})`);
  core.addColorStop(1, `rgba(${tone(SKY)},0)`);
  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(x, y, cr * 1.7, 0, Math.PI * 2);
  ctx.fill();
}

const FOG_MAX = 5000;
const bucketCount = new Int32Array(32), bucketX = new Float32Array(32 * FOG_MAX), bucketY = new Float32Array(32 * FOG_MAX), bucketS = new Float32Array(32 * FOG_MAX);
const fogU = new Float32Array(FOG_MAX * 3), fogR = new Float32Array(FOG_MAX), fogS = new Float32Array(FOG_MAX), fogP = new Float32Array(FOG_MAX);
for (let i = 0; i < FOG_MAX; i++) {
  const th = hash(i * 5 + 101) * Math.PI * 2, z = hash(i * 5 + 102) * 2 - 1, s = Math.sqrt(1 - z * z);
  fogU[i * 3] = Math.cos(th) * s;
  fogU[i * 3 + 1] = Math.sin(th) * s;
  fogU[i * 3 + 2] = z;
  fogR[i] = hash(i * 5 + 103);
  fogS[i] = 0.5 + hash(i * 5 + 104) * 1.5; // how fast and far it flies in the blast
  fogP[i] = hash(i * 5 + 105);
}

/** The ball itself: thousands of fine crisp particles, dense at the core, slowly turning; in the
 *  blast they fly outward trailing streaks and fade. No outline, no skin. */
export function drawFog(ctx: CanvasRenderingContext2D, x: number, y: number, R: number, q: number, B: BirthState, t: number, toneIn: Tone, day: boolean, mx = -1e4, my = -1e4): void {
  const n = Math.min(FOG_MAX, Math.round(TUNE.particles * PERF.q));
  const b = TUNE.burstAt, sp = TUNE.burstSpan, eb = clamp((q - b) / (sp * 0.6), 0, 1);
  const fade = 1 - sstep(0.3, 1, eb);
  if (n <= 0 || fade < 0.01) return;
  const tone: Tone = day ? (c) => c : toneIn;
  const rad = radius(R, B), spin = t * (REDUCED ? 0.05 : 0.2), reach = R * 1.1, calm = REDUCED ? 0 : 1 - sstep(0, 0.25, eb); // the cursor and the scroll only stir the ball while it is whole
  const ee = glide(eb), pe = glide(clamp(eb - 0.05 * TUNE.trail, 0, 1));
  const shake = REDUCED ? 0 : B.over * rad * 0.035 * (eb > 0 ? 0 : 1);
  const col = day ? '40,120,225' : '186,230,253', hot = day ? '20,70,170' : '255,255,255';
  const cls4 = [col, hot];
  const trailOn = eb > 0 && eb < 1 && TUNE.trail > 0;
  const NB = 8, NC = 2, cnt = bucketCount.fill(0), alive = B.alive > 0.5 || eb > 0 ? 1 : B.alive;
  if (trailOn) {
    ctx.lineCap = 'round';
    ctx.strokeStyle = `rgba(${tone(CYAN)},${0.35 * fade})`;
    ctx.lineWidth = 0.9;
    ctx.beginPath();
  }
  for (let i = 0; i < n; i++) {
    const ux = fogU[i * 3], uy = fogU[i * 3 + 1], uz = fogU[i * 3 + 2];
    // turn about the vertical axis; scrolling twists the outer particles further, like the arms of a galaxy
    const ang = spin + B.energy * 2.4 * (1 - fogR[i]) * calm, ca = Math.cos(ang), sa = Math.sin(ang);
    const X = ux * ca + uz * sa, Z = -ux * sa + uz * ca;
    const persp = 1 / (1 - 0.2 * Z); // nearer particles sit a little further from the centre and look larger
    // slow turbulence: the ball moves like plasma, never like a still picture
    const flow = (0.035 + 0.05 * B.charge) * Math.sin(t * 0.9 + uy * 4 + fogP[i] * 6.28) * calm;
    const rr = rad * persp * Math.pow(fogR[i], TUNE.density * 0.85) * (1 + 0.03 * Math.sin(t * 2 + fogP[i] * 40) + flow + 0.12 * B.energy * Math.sin(t * 5 + fogP[i] * 30) * calm);
    const out = ee * R * (0.8 + fogS[i] * 2.6) * TUNE.burstReach;
    let px = x + X * (rr + out) + shake * Math.sin(t * 31 + fogP[i] * 90);
    let py = y + uy * (rr + out) + shake * Math.cos(t * 27 + fogP[i] * 70);
    // the cursor parts the particles like a hand through smoke, then they settle back
    const dx = px - mx, dy = py - my, d2 = dx * dx + dy * dy;
    if (calm > 0 && d2 < reach * reach * 0.6) {
      const d = Math.sqrt(d2) + 0.001, k = 1 - d / (reach * 0.77), push = k * k * R * 0.35 * calm;
      px += (dx / d) * push;
      py += (dy / d) * push;
    }
    if (trailOn) {
      const pout = pe * R * (0.8 + fogS[i] * 2.6) * TUNE.burstReach;
      ctx.moveTo(x + X * (rr + pout), y + uy * (rr + pout));
      ctx.lineTo(px, py);
    }
    const near = 1 - fogR[i], front = 0.5 + 0.5 * Z;
    const al = (0.25 + 0.5 * near * near + 0.25 * front) * fade * (0.7 + 0.5 * B.charge) * alive * (1 + 0.8 * B.over * near);
    const cls = near > 0.8 && B.over > 0.3 ? 1 : 0;
    const bk = Math.min(NB - 1, Math.floor(Math.min(1, al) * NB)) + cls * NB;
    const c = cnt[bk]++;
    bucketX[bk * FOG_MAX + c] = px;
    bucketY[bk * FOG_MAX + c] = py;
    bucketS[bk * FOG_MAX + c] = (0.9 + 0.9 * front + (near > 0.85 ? 0.8 : 0)) * (0.85 + 0.3 * persp - 0.3);
  }
  if (trailOn) ctx.stroke();
  for (let bk = 0; bk < NB * NC; bk++) {
    if (!cnt[bk]) continue;
    ctx.globalAlpha = ((bk % NB) + 0.5) / NB;
    ctx.fillStyle = `rgb(${tone(cls4[Math.floor(bk / NB)] ?? col)})`;
    for (let c = 0; c < cnt[bk]; c++) {
      const sz = bucketS[bk * FOG_MAX + c];
      ctx.fillRect(bucketX[bk * FOG_MAX + c] - sz / 2, bucketY[bk * FOG_MAX + c] - sz / 2, sz, sz);
    }
  }
  ctx.globalAlpha = 1;
}

const JETS = Array.from({ length: 44 }, (_, k) => ({
  a: hash(k * 7 + 1) * Math.PI * 2,
  per: 0.9 + hash(k * 7 + 2) * 1.4,
  off: hash(k * 7 + 3),
  spread: 0.05 + hash(k * 7 + 4) * 0.22,
  reach: 0.9 + hash(k * 7 + 5) * 0.9,
  bits: Array.from({ length: 16 }, (_, j) => ({ lag: j * 0.022, side: (hash(k * 13 + j) - 0.5) * 2, spd: 0.85 + hash(k * 17 + j) * 0.3 })),
}));

/** Spark jets: bursts of bright particles shoot out of the core, and as the overcharge builds more
 *  of them are sucked back in. Each jet is a spray of dots (no lines), brighter at its head. */
export function drawSparks(ctx: CanvasRenderingContext2D, x: number, y: number, R: number, B: BirthState, t: number, toneIn: Tone, day: boolean): void {
  if (B.charge < 0.02 || B.alive < 0.01 || TUNE.neurons <= 0) return;
  const tone: Tone = day ? (c) => c : toneIn, rad = radius(R, B);
  const live = Math.min(JETS.length, Math.floor(TUNE.neurons * (0.1 + 0.9 * B.charge) + 6 * B.over)), speed = TUNE.neuronSpeed * (REDUCED ? 0.4 : 1 + 2.4 * B.over + 1.2 * B.energy);
  const white = day ? '20,32,60' : '255,255,255';
  ctx.globalAlpha = 1;
  for (let k = 0; k < live; k++) {
    const s = JETS[k], ph = ((t * speed) / s.per + s.off) % 1;
    const inward = k % 3 === 2 && B.over > 0.15; // some jets gather instead of fire while overcharging
    const ca = Math.cos(s.a), sa = Math.sin(s.a);
    for (let j = 0; j < s.bits.length; j++) {
      const bit = s.bits[j], p = clamp((ph - bit.lag) / 0.55, 0, 1);
      if (p <= 0 || p >= 1) continue;
      const d = inward ? rad * (1.7 - 1.5 * easeOut(p)) : rad * (0.1 + 1.5 * s.reach * easeOut(p) * bit.spd);
      const lat = bit.side * s.spread * rad * (inward ? 1 - p : p);
      const px = x + ca * d - sa * lat, py = y + sa * d + ca * lat;
      const lit = Math.sin(Math.PI * p) * B.alive, head = j < 3;
      ctx.fillStyle = `rgba(${head ? white : tone(CYAN)},${(head ? 0.95 : 0.7) * lit})`;
      const sz = (head ? 2.6 : 1.6) * (1 - 0.5 * p);
      ctx.fillRect(px - sz / 2, py - sz / 2, sz, sz);
      if (head) {
        ctx.fillStyle = `rgba(${tone(CYAN)},${0.25 * lit})`;
        ctx.beginPath();
        ctx.arc(px, py, 6 * (1 - 0.4 * p), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

const RAYS = Array.from({ length: 34 }, (_, k) => ({ a: (k / 34) * Math.PI * 2 + hash(k + 60) * 0.12, len: 1.4 + hash(k + 70) * 3.6, w: 1 + hash(k + 80) * 2.2 }));

/** The blast: a flash that blooms (no clipping), a soft expanding shockwave band, and streaking rays. */
export function drawBurst(ctx: CanvasRenderingContext2D, x: number, y: number, R: number, q: number, B: BirthState, toneIn: Tone, day: boolean): void {
  const tone: Tone = day ? (c) => c : toneIn; // a pale page keeps the bright blues: darkened they turn to grey smoke
  const rad = R * (1 + TUNE.chargeGrow), b = TUNE.burstAt, sp = TUNE.burstSpan;
  const sh = clamp((q - b) / (sp * 0.6), 0, 1), white = day ? '30,110,230' : '255,255,255'; // on a pale page the flash is a saturated blue, never a dark smudge
  ctx.globalAlpha = 1;
  if (B.flash > 0.01) {
    const fr = rad * (2.4 + 2.4 * sh);
    const g = ctx.createRadialGradient(x, y, 0, x, y, fr);
    g.addColorStop(0, `rgba(${white},${(day ? 0.6 : 0.95) * Math.min(1, B.flash)})`);
    g.addColorStop(0.25, `rgba(${tone('190,235,255')},${0.6 * B.flash})`);
    g.addColorStop(0.6, `rgba(${tone(CYAN)},${0.22 * B.flash})`);
    g.addColorStop(1, `rgba(${tone(SKY)},0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, fr, 0, Math.PI * 2);
    ctx.fill();
  }
  if (sh > 0 && sh < 1) {
    const e = glide(sh), fade = Math.pow(1 - sh, 1.4);
    // shockwave: a soft thick band of light, not a thin outline
    const r0 = rad * (0.5 + 5.5 * e), g = ctx.createRadialGradient(x, y, Math.max(0, r0 * 0.6), x, y, r0 * 1.08);
    g.addColorStop(0, `rgba(${tone(SKY)},0)`);
    g.addColorStop(0.8, `rgba(${tone('120,215,255')},${(day ? 0.18 : 0.3) * fade})`);
    g.addColorStop(0.93, `rgba(${tone('200,240,255')},${(day ? 0.25 : 0.45) * fade})`);
    g.addColorStop(1, `rgba(${tone(SKY)},0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r0 * 1.08, 0, Math.PI * 2);
    ctx.fill();
    // rays: streaks racing outward, thin at the tip
    if (TUNE.rays > 0) {
      ctx.lineCap = 'round';
      for (const r of RAYS) {
        const tip = rad * (0.3 + r.len * e), tail = rad * (0.3 + r.len * Math.max(0, e - 0.35)), c = Math.cos(r.a), s = Math.sin(r.a);
        const lg = ctx.createLinearGradient(x + c * tail, y + s * tail, x + c * tip, y + s * tip);
        lg.addColorStop(0, `rgba(${tone(CYAN)},0)`);
        lg.addColorStop(1, `rgba(${white},${0.85 * fade * Math.min(1, TUNE.rays)})`);
        ctx.strokeStyle = lg;
        ctx.lineWidth = r.w * (1 - 0.5 * sh);
        ctx.beginPath();
        ctx.moveTo(x + c * tail, y + s * tail);
        ctx.lineTo(x + c * tip, y + s * tip);
        ctx.stroke();
      }
    }
  }
}
