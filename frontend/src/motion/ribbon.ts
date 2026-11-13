// Dotted-ribbon background, ported from design/frontend/story-reference.dc.html (startWaves).
// One diagonal ribbon of dots twists through 3D and is drawn flat; it is cut lengthwise into one
// lane per arm. The lanes travel together, split for the contenders, sit at their real score
// heights on the scores scene, the agent lane moves at the reveal (down when it trails, up when it
// leads), and all rejoin at the recap. Everything is plain canvas 2D driven by requestAnimationFrame,
// so it runs on every browser. Only the visuals live here; every number comes from the caller.

export interface Lane {
  name: string;
  rgb: [number, number, number];
  /** Accuracy in percent, used when the ribbon turns into a chart on the scores scene. */
  score: number;
}

/** Where the lanes split apart, turn into a score chart, show the reveal and rejoin. */
export interface Beats {
  split: number;
  chart: number;
  reveal: number;
  rejoin: number;
}

export const STORY_BEATS: Beats = { split: 1, chart: 3, reveal: 5, rejoin: 8 };

export interface Flyer {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface RibbonOptions {
  lanes: Lane[];
  /** +1 the agent lane (the last one) drops at the reveal, -1 it rises, 0 it stays. */
  revealDirection: number;
  /** Cases in the run, for the readout counter. */
  nCases: number;
  /** Scene positions (0-based) of the story beats; defaults are the 9-scene story's. */
  beats?: Beats;
  still: boolean;
  isNight: () => boolean;
  /** The robot's screen position and velocity, or null when it is docked. */
  flyer: () => Flyer | null;
  hudCase?: HTMLElement | null;
  hudScene?: HTMLElement | null;
}

type RGB = [number, number, number];
const K: RGB = [37, 48, 49];
// one entry per scene: five layer colours, wave height, wavelength, speed, and how high the waves sit
const SC: { c: RGB[]; amp: number; wl: number; sp: number; lift: number }[] = [
  { c: [K, [20, 88, 104], K, [40, 130, 140], [120, 190, 195]], amp: 1.0, wl: 1.0, sp: 1.0, lift: 0 },
  { c: [K, [28, 72, 122], K, [58, 110, 170], [130, 170, 215]], amp: 1.15, wl: 0.85, sp: 1.1, lift: -0.03 },
  { c: [K, [30, 98, 74], K, [60, 140, 100], [140, 200, 160]], amp: 0.8, wl: 1.25, sp: 0.8, lift: 0.03 },
  { c: [K, [56, 60, 140], K, [90, 100, 190], [150, 160, 225]], amp: 1.3, wl: 0.8, sp: 1.2, lift: -0.04 },
  { c: [K, [52, 110, 78], K, [40, 130, 120], [130, 200, 180]], amp: 0.7, wl: 1.35, sp: 0.7, lift: 0.05 },
  { c: [K, [40, 56, 70], K, [88, 110, 130], [160, 180, 195]], amp: 1.5, wl: 0.7, sp: 1.4, lift: -0.06 },
  { c: [K, [24, 100, 130], K, [60, 150, 170], [140, 205, 215]], amp: 1.0, wl: 1.0, sp: 0.9, lift: 0 },
  { c: [K, [70, 60, 120], K, [110, 100, 170], [170, 165, 210]], amp: 1.1, wl: 0.9, sp: 1.0, lift: -0.02 },
  { c: [K, [22, 104, 110], K, [70, 160, 160], [150, 210, 205]], amp: 0.9, wl: 1.15, sp: 0.8, lift: 0.03 },
];
const NC = 10;
const ALPHA = [0.3, 0.46, 0.64];
const NIGHT: RGB = [205, 238, 240];
const TEXT_SELECTOR = '.scene .eyebrow, .scene h1, .scene h2, .scene .lead, .scene .stat, .scene .card, .scene .land, .scene .checks, .say';

const mix = (a: number, b: number, f: number) => a + (b - a) * f;
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const sstep = (a: number, b: number, x: number) => {
  const s = clamp((x - a) / (b - a), 0, 1);
  return s * s * (3 - 2 * s);
};
const hsh = (n: number) => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/** Scene position as a number: 0 = first scene aligned, 1.5 = halfway from the second to the third. */
export function scenePosition(secs: readonly HTMLElement[]): number {
  const y = window.scrollY || document.documentElement.scrollTop || 0;
  const tops = secs.map((s) => s.getBoundingClientRect().top + y);
  let i = 0;
  for (let k = 0; k < tops.length; k++) if (y + 2 >= tops[k]) i = k;
  const span = tops[i + 1] !== undefined ? tops[i + 1] - tops[i] : 0;
  return i + (span ? clamp((y - tops[i]) / span, 0, 1) : 0);
}

interface Influence {
  x: number;
  y: number;
  r: number;
  p: number;
  dx: number;
  dy: number;
  sw: number;
  gl: number;
}

export function startRibbon(cv: HTMLCanvasElement, o: RibbonOptions): () => void {
  const ctx = cv.getContext('2d');
  if (!ctx) return () => {};
  const still = o.still;
  const NL = Math.max(1, o.lanes.length);
  const buckets: number[][] = [];
  for (let b = 0; b < NC * NL * 3; b++) buckets.push([]);
  const secs = Array.from(document.querySelectorAll<HTMLElement>('section.scene'));
  const lab = o.lanes.map(() => ({ x: 0, y: 0 }));
  const R = { y: 0.24, run: 0.34, sw: 0.17, w: 0.3, l: 1.15, s: 1.0, al: 1.0, n: 240, m: 18 };
  let W = 0, H = 0, dpr = 1, raf = 0, last = 0, t = 0, pos = 0, target = 0, kick = 0, kickGoal = 0, idx = 0;
  let ie = still ? 1 : 0, t0i = 0, nightT = 0, tb = 99;
  let mx = -999, my = -999, mtx = -999, mty = -999, minf = 0, mouseIn = false;
  let lastCase = -1, lastScene = -1, ch = 0;
  let rects: number[][] = [];
  let tEls: HTMLElement[] = [];
  let tElsAt = -1e9;
  const rg = document.createRange();

  const size = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = cv.clientWidth || window.innerWidth;
    H = cv.clientHeight || window.innerHeight;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // phones get about 5,000 dots instead of about 13,000
    const phone = W <= 900;
    R.n = phone ? 150 : 240;
    R.m = phone ? 11 : 18;
  };

  const look = (p: number) => {
    const i = Math.max(0, Math.min(SC.length - 2, Math.floor(p)));
    let g = clamp((Math.min(1, p - i) - 0.15) / 0.7, 0, 1);
    g = g * g * (3 - 2 * g);
    const a = SC[i], b = SC[i + 1];
    return {
      c: a.c.map((ca, n): RGB => [
        Math.round(mix(ca[0], b.c[n][0], g)),
        Math.round(mix(ca[1], b.c[n][1], g)),
        Math.round(mix(ca[2], b.c[n][2], g)),
      ]),
      amp: mix(a.amp, b.amp, g),
      wl: mix(a.wl, b.wl, g),
      sp: mix(a.sp, b.sp, g),
      lift: mix(a.lift, b.lift, g),
    };
  };
  type Look = ReturnType<typeof look>;

  const B = o.beats ?? STORY_BEATS;
  const story = (p: number) => ({
    split: sstep(B.split - 0.45, B.split, p) * (1 - sstep(B.rejoin - 0.8, B.rejoin, p)),
    drop: sstep(B.reveal - 0.5, B.reveal, p) * (1 - sstep(B.rejoin - 1, B.rejoin, p)) * o.revealDirection,
    chart: sstep(B.chart - 0.6, B.chart, p) * (1 - sstep(B.chart, B.chart + 0.7, p)),
  });

  const ribbon = (P: Look, sp: number, dr: number, infl: Influence[]) => {
    const F = H * 2.4;
    const li = Math.round(((0.78 + 0.14) / 1.28) * (R.n - 1));
    for (let k = 0; k < NL; k++) {
      const kk = NL === 1 ? 0 : (2 * k) / (NL - 1) - 1;
      const agentLane = k === NL - 1 && NL > 1;
      const ph = t * R.s + pos * 0.9 + k * 0.9 * sp;
      const lo = -1 + (2 * k) / NL, hi = -1 + (2 * (k + 1)) / NL, vc = (lo + hi) / 2;
      const fan = 1 + 0.8 * sp;
      const sepY = H * 0.15 * kk * sp + (agentLane ? H * 0.26 * dr : 0) - (k === 0 ? H * 0.05 * Math.abs(dr) : 0);
      const alk = agentLane ? 1 - 0.45 * Math.abs(dr) : 1;
      const tgtY = H * (0.88 - ((clamp(o.lanes[k]?.score ?? 60, 50, 100) - 60) / 40) * 0.4);
      for (let iu = 0; iu < R.n; iu++) {
        const u = -0.14 + (1.28 * iu) / (R.n - 1);
        const s = (u * Math.PI * 2) / (R.l * P.wl);
        const wave = H * R.sw * P.amp * (1 + 0.3 * kick) * (Math.sin(s + ph) * 0.62 + Math.sin(s * 1.7 - ph * 1.2) * 0.3);
        const cy = mix(H * (R.y + P.lift * 0.6 + (R.run * (u + 0.14)) / 1.28) + sepY + wave, tgtY + wave * 0.3, ch);
        const w = H * R.w * (0.6 + 0.4 * Math.sin(s * 0.8 + ph * 0.7 + 1.3));
        const th = s * 1.3 + ph * 0.8;
        const ct = Math.cos(th), st = Math.sin(th);
        const zc = H * 0.18 * Math.cos(s * 0.6 + ph * 0.5);
        const x0 = u * W;
        const open = Math.abs(ct);
        if (iu === li) {
          const zl = zc + vc * w * st, fl = F / (F - zl);
          lab[k].x = W / 2 + (x0 - W / 2) * fl * 1.18;
          lab[k].y = H / 2 + (cy + vc * w * ct - H / 2) * fl * 1.18;
        }
        for (let iv = 0; iv < R.m; iv++) {
          const v = vc + (lo + (iv / (R.m - 1)) * (hi - lo) - vc) * fan;
          const z = zc + v * w * st;
          const f = F / (F - z);
          let sx = W / 2 + (x0 - W / 2) * f * 1.18;
          let sy = H / 2 + (cy + v * w * ct - H / 2) * f * 1.18;
          let rad = (1.3 + 0.7 * f) * (0.85 + 0.2 * (1 - open)) * (1 + Math.max(0, 1 - f) * 2.2);
          let a = (0.3 + 0.32 * (1 - open)) * R.al * alk;
          a *= 0.55 + 0.45 * clamp(0.6 + z / (H * 0.5), 0, 1);
          // intro: every dot starts somewhere else and flies into place, each on its own delay
          if (ie < 1) {
            const hh = hsh(iu * 53 + iv * 7 + k * 1000);
            const lp = clamp(ie * 1.5 - hh * 0.5, 0, 1);
            const e3 = lp * lp * (3 - 2 * lp);
            if (e3 < 1) {
              const q = 1 - e3;
              sx += (hsh(hh * 91 + 1) - 0.5) * W * 0.9 * q * q;
              sy += (hsh(hh * 57 + 2) - 0.5) * H * 1.1 * q * q;
              a *= 0.25 + 0.75 * e3;
            }
          }
          // rebuild swarm: on a scene change every dot bursts out and swirls, then re-forms left to right
          if (tb < 2.3) {
            const hb = hsh(iu * 31 + iv * 11 + k * 7.7);
            let sb: number;
            if (tb < 0.3) {
              const q0 = 1 - tb / 0.3;
              sb = 1 - q0 * q0 * q0;
            } else {
              const rb = clamp((tb - 0.3 - ((u + 0.14) / 1.28) * 0.5 - hb * 0.25) / 1.1, 0, 1);
              sb = 1 - rb * rb * (3 - 2 * rb);
            }
            if (sb > 0.001) {
              const vx = sx - W / 2, vy = sy - H / 2, len = Math.sqrt(vx * vx + vy * vy) || 1;
              const push = sb * (120 + hb * 380);
              sx += (vx / len) * push + (-vy / len) * sb * 140 + (hsh(hb * 91) - 0.5) * 260 * sb;
              sy += (vy / len) * push * 0.8 + (vx / len) * sb * 140 + (hsh(hb * 57) - 0.5) * 260 * sb;
              a *= 1 - 0.4 * sb;
              rad *= 1 + 0.5 * sb;
            }
          }
          // cursor and robot: dots are pushed aside, dragged along and swirled behind
          for (let n = 0; n < infl.length; n++) {
            const q = infl[n];
            const dx = sx - q.x, dy = sy - q.y, d2 = dx * dx + dy * dy;
            if (d2 < q.r * q.r) {
              const d = Math.sqrt(d2) || 1, fl2 = 1 - d / q.r, g = fl2 * fl2;
              sx += (dx / d) * q.p * g + q.dx * g - (dy / d) * q.sw * g;
              sy += (dy / d) * q.p * g + q.dy * g + (dx / d) * q.sw * g;
              if (q.gl) {
                rad *= 1 + 0.8 * g;
                a += 0.3 * g;
              }
            }
          }
          if (minf > 0.01) {
            sx += ((mx - W / 2) / W) * z * 0.12 * minf;
            sy += ((my - H / 2) / H) * z * 0.12 * minf;
          }
          sx += (hsh(iu * 3 + iv * 5 + k * 17) - 0.5) * 3 * (1 - open);
          // thin the dots out near text, so each block of copy sits in a calm clearing
          if (rects.length) {
            let tf = 1;
            for (let n = 0; n < rects.length; n++) {
              const q = rects[n];
              const dx = Math.max(q[0] - sx, 0, sx - q[2]), dy = Math.max(q[1] - sy, 0, sy - q[3]);
              const d = Math.sqrt(dx * dx + dy * dy);
              if (d < 70) tf = Math.min(tf, 0.06 + 0.94 * (d / 70));
            }
            if (tf < 1 && hsh(iu * 7.3 + iv * 13.1 + k * 3.7) > tf) continue;
          }
          if (sx < -4 || sx > W + 4 || sy < -4 || sy > H + 4) continue;
          const ai = a < 0.32 ? 0 : a < 0.5 ? 1 : 2;
          let cs = (u * 0.9 + (v + 1) * 0.06 + t * 0.012 + pos * 0.11) % 1;
          if (cs < 0) cs += 1;
          buckets[(k * NC + Math.floor(cs * NC)) * 3 + ai].push(sx, sy, rad);
        }
      }
    }
  };

  const draw = (P: Look, sp: number, dr: number, infl: Influence[]) => {
    ctx.clearRect(0, 0, W, H);
    const gA = 1 + nightT * 0.9;
    const x1 = W * (0.12 + 0.1 * Math.sin(pos * 0.7 + t * 0.05)), x2 = W * 0.92, y2 = H * (0.32 + 0.05 * Math.cos(t * 0.04));
    const gl1 = P.c[4].join(','), gl2 = P.c[3].join(',');
    const g1 = ctx.createRadialGradient(x1, H * 0.08, 0, x1, H * 0.08, Math.max(W, H) * 0.75);
    g1.addColorStop(0, `rgba(${gl1},${(0.2 * gA).toFixed(3)})`);
    g1.addColorStop(1, `rgba(${gl1},0)`);
    ctx.fillStyle = g1;
    ctx.fillRect(0, 0, W, H);
    const g2 = ctx.createRadialGradient(x2, y2, 0, x2, y2, Math.max(W, H) * 0.65);
    g2.addColorStop(0, `rgba(${gl2},${(0.16 * gA).toFixed(3)})`);
    g2.addColorStop(1, `rgba(${gl2},0)`);
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, W, H);
    if (kick > 0.02) {
      const g3 = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7);
      g3.addColorStop(0, `rgba(${gl1},${(0.34 * kick * gA).toFixed(3)})`);
      g3.addColorStop(1, `rgba(${gl1},0)`);
      ctx.fillStyle = g3;
      ctx.fillRect(0, 0, W, H);
    }
    const base: number[][] = [];
    for (let ci = 0; ci < NC; ci++) {
      const x = ((ci + 0.5) / NC) * 5, i = Math.floor(x), f = x - i;
      const a = P.c[i % 5], b = P.c[(i + 1) % 5];
      base.push([mix(a[0], b[0], f), mix(a[1], b[1], f), mix(a[2], b[2], f)]);
    }
    const cols: string[] = [];
    for (let k = 0; k < NL; k++) {
      const laneCol = o.lanes[k]?.rgb ?? K;
      for (let ci = 0; ci < NC; ci++) {
        let c = base[ci].map((v, n) => mix(v, laneCol[n], sp * 0.7));
        if (nightT > 0.001) c = c.map((v, n) => mix(v, NIGHT[n], nightT * 0.62));
        cols.push(`${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])}`);
      }
    }
    ribbon(P, sp, dr, infl);
    for (let ci = 0; ci < NC * NL; ci++) {
      for (let ai = 0; ai < 3; ai++) {
        const b = buckets[ci * 3 + ai];
        if (!b.length) continue;
        ctx.fillStyle = `rgba(${cols[ci]},${ALPHA[ai]})`;
        ctx.beginPath();
        for (let k = 0; k < b.length; k += 3) {
          ctx.moveTo(b[k] + b[k + 2], b[k + 1]);
          ctx.arc(b[k], b[k + 1], b[k + 2], 0, 6.2832);
        }
        ctx.fill();
        b.length = 0;
      }
    }
    // lane names ride beside their strand while the ribbon is split (desktop only)
    if (sp > 0.05 && ie > 0.9 && W > 900) {
      ctx.font = '500 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const c2 = ctx as CanvasRenderingContext2D & { letterSpacing?: string };
      if ('letterSpacing' in c2) c2.letterSpacing = '2px';
      for (let k = 0; k < NL; k++) {
        const L = lab[k];
        const al = sp * (k === NL - 1 && NL > 1 ? 1 - 0.55 * Math.abs(dr) : 1);
        if (al < 0.05 || L.x < 30 || L.x > W - 150 || L.y < 60 || L.y > H - 60) continue;
        if (rects.some((q) => L.x + 130 > q[0] && L.x - 10 < q[2] && L.y + 10 > q[1] && L.y - 10 < q[3])) continue;
        ctx.globalAlpha = al;
        ctx.fillStyle = `rgba(${cols[k * NC + 4]},1)`;
        ctx.beginPath();
        ctx.arc(L.x, L.y, 4, 0, 6.2832);
        ctx.fill();
        ctx.lineWidth = 4;
        ctx.strokeStyle = nightT > 0.5 ? 'rgba(8,14,15,.85)' : 'rgba(255,255,255,.85)';
        const name = o.lanes[k].name.toUpperCase();
        ctx.strokeText(name, L.x + 14, L.y);
        ctx.fillStyle = nightT > 0.5 ? 'rgba(234,243,243,1)' : `rgba(${o.lanes[k].rgb.join(',')},1)`;
        ctx.fillText(name, L.x + 14, L.y);
      }
      ctx.globalAlpha = 1;
      if ('letterSpacing' in c2) c2.letterSpacing = '0px';
    }
  };

  const frame = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016);
    last = now;
    target = scenePosition(secs);
    pos += (target - pos) * (1 - Math.exp(-dt * 6));
    const near = Math.round(pos);
    if (near !== idx) {
      idx = near;
      kickGoal = 1;
      if (!still && ie >= 1 && tb > 0.8) tb = 0;
    }
    tb = Math.min(99, tb + dt);
    kickGoal *= Math.exp(-dt * 1.1);
    kick += (kickGoal - kick) * (1 - Math.exp(-dt * 4));
    const P = look(pos);
    // reduced motion: the ribbon keeps flowing, only much slower
    t += dt * P.sp * (1 + 0.6 * kick) * (still ? 0.15 : 1);
    if (ie < 1) ie = clamp((now - t0i) / 3400, 0, 1);
    nightT += ((o.isNight() ? 1 : 0) - nightT) * (1 - Math.exp(-dt * 4));
    mx += (mtx - mx) * (1 - Math.exp(-dt * 14));
    my += (mty - my) * (1 - Math.exp(-dt * 14));
    minf += ((mouseIn && !still ? 1 : 0) - minf) * (1 - Math.exp(-dt * 5));
    const infl: Influence[] = [];
    if (minf > 0.01) infl.push({ x: mx, y: my, r: 150, p: 46 * minf, dx: 0, dy: 0, sw: 0, gl: 1 });
    const fx = o.flyer();
    if (fx && !still) {
      const spd = Math.hypot(fx.vx, fx.vy);
      infl.push({ x: fx.x, y: fx.y, r: 240, p: 62, dx: fx.vx * 0.05, dy: fx.vy * 0.05, sw: Math.min(34, spd * 0.05), gl: 0 });
      infl.push({ x: fx.x - fx.vx * 0.14, y: fx.y - fx.vy * 0.14, r: 200, p: 0, dx: fx.vx * 0.03, dy: fx.vy * 0.03, sw: Math.min(46, spd * 0.07), gl: 0 });
    }
    const sr = story(pos);
    ch = sr.chart;
    // at the reveal the swarm opens a ring around the gap number and holds it open
    const hole = sstep(B.reveal - 0.45, B.reveal, pos) * (1 - sstep(B.reveal, B.reveal + 0.6, pos));
    if (hole > 0.02) infl.push({ x: W / 2, y: H / 2, r: Math.min(W, H) * 0.42, p: 260 * hole, dx: 0, dy: 0, sw: 40 * hole, gl: 0 });
    // text blocks the dots keep clear of; the list is refreshed twice a second so new panels count
    if (now - tElsAt > 500) {
      tEls = Array.from(document.querySelectorAll<HTMLElement>(TEXT_SELECTOR));
      tElsAt = now;
    }
    rects = [];
    for (const el of tEls) {
      if (el.style.visibility === 'hidden') continue;
      let r: DOMRect;
      if (el.matches('h1,h2,.lead,.eyebrow')) {
        rg.selectNodeContents(el);
        r = rg.getBoundingClientRect();
      } else r = el.getBoundingClientRect();
      if (r.width > 0 && r.bottom > 0 && r.top < H && r.right > 0 && r.left < W) rects.push([r.left - 10, r.top - 8, r.right + 10, r.bottom + 8]);
    }
    draw(P, sr.split, sr.drop, infl);
    // readouts
    const yy = window.scrollY || document.documentElement.scrollTop || 0;
    const mxs = Math.max(1, document.documentElement.scrollHeight - H);
    const cn = Math.round(clamp(yy / mxs, 0, 1) * o.nCases);
    if (o.hudCase && cn !== lastCase) {
      lastCase = cn;
      o.hudCase.textContent = cn >= 1000 ? cn.toLocaleString('en-US') : String(cn).padStart(4, '0');
    }
    const sn = clamp(Math.round(pos), 0, secs.length - 1) + 1;
    if (o.hudScene && sn !== lastScene) {
      lastScene = sn;
      o.hudScene.textContent = String(sn).padStart(2, '0');
    }
    raf = requestAnimationFrame(frame);
  };

  const onMove = (e: PointerEvent) => {
    if (e.pointerType === 'touch') return;
    mtx = e.clientX;
    mty = e.clientY;
    if (!mouseIn) {
      mx = mtx;
      my = mty;
    }
    mouseIn = true;
  };
  const onLeave = () => {
    mouseIn = false;
  };
  pos = target = scenePosition(secs);
  idx = Math.round(pos);
  size();
  t0i = performance.now();
  window.addEventListener('resize', size);
  window.addEventListener('pointermove', onMove, { passive: true });
  document.documentElement.addEventListener('mouseleave', onLeave);
  raf = requestAnimationFrame(frame);
  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', size);
    window.removeEventListener('pointermove', onMove);
    document.documentElement.removeEventListener('mouseleave', onLeave);
  };
}
