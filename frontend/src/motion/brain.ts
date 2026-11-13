// Message graph background (mockup v7). A glowing 3D cloud of every message we know: the
// practice library (small nodes) and the run's own messages (large nodes), one bright cluster per
// kind of message, tied to their most similar messages. It turns slowly; drag to spin it, hover a
// node to read it, click a run node to pick it. It wakes up as the reader scrolls:
//   1 ask:      dim, slow turn; the kinds the reader types light up in their colour and are named
//   2 race:     the run's nodes flash cyan / pink / amber as the test scores them
//   3 mistake:  the camera closes in on the picked node; a white arc runs through its neighbours
//   4-6:        settles back, still touchable
// Canvas 2D with a hand-rolled perspective projection; the layout is seeded.
import { scenePosition } from './ribbon.ts';
import type { WebMessage } from './webfield.ts';
import { hubName, matchLabels } from './webfield.ts';

export interface BrainOptions {
  messages: WebMessage[];
  library: { text: string; label: string }[];
  query: () => string;
  focus: () => number[];
  featured: () => number;
  onPick?: (index: number) => void;
  onRace?: (shown: number) => void;
  onHover: (hit: { kind: 'run' | 'lib'; index: number; x: number; y: number } | null) => void;
  still: boolean;
  isNight: () => boolean;
  /** maps the page's screen position onto the graph's own steps (ask, race, close-up, answer) */
  remap?: (pos: number) => number;
}

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const mix = (a: number, b: number, f: number) => a + (b - a) * f;
const sstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const hash = (n: number) => {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
};
const gauss = (n: number) => (hash(n) + hash(n + 0.37) + hash(n + 0.71) - 1.5) / 0.5;

// themes of the Banking77 kinds (first match wins); anything else is Account
const FAMILIES = [
  { name: 'Top-ups', keys: ['top_up', 'topping'] },
  { name: 'Cash & ATM', keys: ['atm', 'cash'] },
  { name: 'Exchange', keys: ['exchange', 'fiat', 'currency', 'country', 'supported'] },
  { name: 'Transfers', keys: ['transfer', 'beneficiary', 'receiving'] },
  { name: 'Cards', keys: ['card', 'contactless', 'visa', 'virtual', 'disposable', 'apple', 'pin', 'lost', 'activate', 'order'] },
  { name: 'Payments', keys: ['refund', 'declined', 'pending', 'payment', 'reverted', 'direct_debit', 'transaction', 'extra_charge', 'balance'] },
  { name: 'Account', keys: [] as string[] },
];
// bold, bright cluster colours
// cool, distinct theme colours; pink and amber are kept for wrong answers only
const KIND_COLOURS = ['56,189,248', '167,139,250', '74,222,128', '45,212,191', '129,140,248', '125,211,252', '190,242,100', '34,227,240'];
export const CYAN = '34,227,240', PINK = '255,77,141', AMBER = '255,194,61';

interface P {
  x: number;
  y: number;
  z: number;
  col: string;
  r: number;
  run: number; // index into messages, or -1
  lib: number; // index into library, or -1
  kind: string;
  level: number; // 0 message, 1 kind hub, 2 theme hub
}

export function startBrain(cv: HTMLCanvasElement, o: BrainOptions): () => void {
  const ctx = cv.getContext('2d');
  if (!ctx) return () => {};
  const secs = Array.from(document.querySelectorAll<HTMLElement>('section.scene'));
  const M = o.messages.length;

  // structure: kinds of message grouped into themes; each theme is a lobe on a tilted ring,
  // its kinds sit in a small circle around the lobe's hub, messages hang from their kind
  const kinds = [...new Set([...o.library.map((l) => l.label), ...o.messages.map((m) => m.true)])];
  // banking kinds fall into named themes; any other dataset gets generic lobes: one per kind when
  // there are few (Emotion's six moods), else alphabetical groups named by their first and last kind
  const bankFam = (k: string) => FAMILIES.find((f) => f.keys.some((w) => k.toLowerCase().includes(w)))?.name ?? 'Account';
  const banking = kinds.filter((k) => bankFam(k) !== 'Account').length >= kinds.length * 0.5;
  const generic = new Map<string, string>();
  if (!banking) {
    const sorted = [...kinds].sort();
    if (sorted.length <= 8) sorted.forEach((k) => generic.set(k, hubName(k)));
    else {
      const per = Math.ceil(sorted.length / 7);
      for (let g = 0; g < sorted.length; g += per) {
        const part = sorted.slice(g, g + per);
        const first = (k: string) => hubName(k).split(' ')[0];
        const name = first(part[0]) === first(part[part.length - 1]) ? first(part[0]) : `${first(part[0])} – ${first(part[part.length - 1])}`;
        part.forEach((k) => generic.set(k, name));
      }
    }
  }
  const famOf = (k: string) => (banking ? bankFam(k) : generic.get(k) ?? 'other');
  const fams = banking ? FAMILIES.map((f) => f.name).filter((n) => kinds.some((k) => famOf(k) === n)) : [...new Set(kinds.map(famOf))];
  const famHub = new Map<string, { x: number; y: number; z: number; col: string }>();
  fams.forEach((n, i) => {
    const a = (i / fams.length) * Math.PI * 2 + 0.3;
    famHub.set(n, { x: Math.cos(a) * 1.0, y: Math.sin(a * 2) * 0.1, z: Math.sin(a) * 0.75, col: KIND_COLOURS[i % KIND_COLOURS.length] });
  });
  const centre = new Map<string, { x: number; y: number; z: number; col: string; fam: string }>();
  fams.forEach((n) => {
    const h = famHub.get(n)!;
    const members = kinds.filter((k) => famOf(k) === n);
    const rad = 0.16 + 0.025 * Math.sqrt(members.length) * 3;
    members.forEach((k, j) => {
      // a little sphere of kinds around the hub (golden spiral), so lobes look round from any side
      const yy = members.length === 1 ? 0 : 1 - (2 * (j + 0.5)) / members.length;
      const rr = Math.sqrt(1 - yy * yy), th = j * 2.399963;
      centre.set(k, { x: h.x + Math.cos(th) * rr * rad, y: h.y + yy * rad * 0.9, z: h.z + Math.sin(th) * rr * rad, col: h.col, fam: n });
    });
  });
  const pts: P[] = [];
  o.library.forEach((l, i) => {
    const c = centre.get(l.label)!;
    pts.push({ x: c.x + gauss(i + 11) * 0.035, y: c.y + gauss(i + 13) * 0.03, z: c.z + gauss(i + 17) * 0.035, col: c.col, r: 2.2, run: -1, lib: i, kind: l.label, level: 0 });
  });
  const runPt: number[] = [];
  o.messages.forEach((m, i) => {
    const c = centre.get(m.true)!;
    runPt.push(pts.length);
    pts.push({ x: c.x + gauss(i + 501) * 0.045, y: c.y + gauss(i + 503) * 0.04, z: c.z + gauss(i + 507) * 0.045, col: c.col, r: 4.4, run: i, lib: -1, kind: m.true, level: 0 });
  });
  // hubs: one per kind (small) and one per theme (large), joined as a tree
  const kindHub = new Map<string, number>();
  centre.forEach((c, k) => {
    kindHub.set(k, pts.length);
    pts.push({ x: c.x, y: c.y, z: c.z, col: c.col, r: 2.6, run: -1, lib: -1, kind: k, level: 1 });
  });
  const famPt = new Map<string, number>();
  famHub.forEach((h, n) => {
    famPt.set(n, pts.length);
    pts.push({ x: h.x, y: h.y, z: h.z, col: h.col, r: 7, run: -1, lib: -1, kind: 'theme:' + n, level: 2 });
  });
  const kindLinks: [number, number][] = [];
  pts.forEach((p, i) => {
    if (p.level === 0) kindLinks.push([i, kindHub.get(p.kind)!]);
    if (p.level === 1) kindLinks.push([i, famPt.get(centre.get(p.kind)!.fam)!]);
  });
  // the ring that ties the themes together
  const ring: [number, number][] = fams.map((n, i) => [famPt.get(n)!, famPt.get(fams[(i + 1) % fams.length])!]);
  // fine glowing dust: a cloud around each lobe and a faint band along the ring
  const dust = Array.from({ length: 2600 }, (_, k) => {
    if (k % 3 === 0) {
      const a = hash(k + 2000) * Math.PI * 2;
      return { x: Math.cos(a) * 1.0 + gauss(k + 2100) * 0.05, y: Math.sin(a * 2) * 0.1 + gauss(k + 2150) * 0.04, z: Math.sin(a) * 0.75 + gauss(k + 2200) * 0.05, col: KIND_COLOURS[1] };
    }
    const h = famHub.get(fams[k % fams.length])!;
    return { x: h.x + gauss(k + 2300) * 0.17, y: h.y + gauss(k + 2400) * 0.15, z: h.z + gauss(k + 2500) * 0.17, col: h.col };
  });
  // links: each run message to its similar messages (plain-code TF-IDF neighbours)
  const links: [number, number][] = [];
  o.messages.forEach((m, i) => m.nb.forEach((j) => (i < j || !o.messages[j].nb.includes(i) ? links.push([runPt[i], runPt[j]]) : null)));

  let W = 0, H = 0, S = 1, raf = 0, last = performance.now(), t = 0, pos = 0;
  let yaw = 0.4, pitch = -0.42, spin = 0.06, dragging = false, dragX = 0, dragY = 0, moved = 0;
  let mx = -1e4, my = -1e4, hovered = -1, raceStart = -1, lastShown = -1;
  let camZ = 1, cx = 0, cy = 0;
  const lit = new Float32Array(M);
  const scr = new Float32Array(pts.length * 3); // screen x, y, scale
  const size = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = cv.clientWidth || window.innerWidth;
    H = cv.clientHeight || window.innerHeight;
    S = W > 760 ? Math.min(W * 0.34, H * 0.56) : Math.min(W * 0.6, H * 0.4);
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  const project = (x: number, y: number, z: number, out: Float32Array, k: number) => {
    const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = x * cyw - z * syw, z1 = x * syw + z * cyw;
    const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
    const f = 3.6 / (3.6 + z2);
    out[k] = cx + x1 * S * f * camZ;
    out[k + 1] = cy + y1 * S * f * camZ;
    out[k + 2] = f;
  };
  const awake = () => sstep(0.35, 0.95, pos);
  const nodeAt = (sx: number, sy: number, runOnly: boolean) => {
    let best = -1, bestD = 16;
    for (let i = 0; i < pts.length; i++) {
      if ((runOnly && pts[i].run < 0) || pts[i].level > 0) continue;
      const d = Math.hypot(scr[i * 3] - sx, scr[i * 3 + 1] - sy) - (pts[i].run >= 0 ? 4 : 0);
      if (d < bestD) {
        best = i;
        bestD = d;
      }
    }
    return best;
  };
  const onUi = (e: Event) => !!(e.target as Element)?.closest?.('a,button,input,select,textarea,label,.glass,.say,.r-body,.verdict-hero');
  const move = (e: PointerEvent) => {
    if (dragging) {
      yaw += (e.clientX - dragX) * 0.006;
      pitch = clamp(pitch + (e.clientY - dragY) * 0.004, -0.9, 0.9);
      moved += Math.abs(e.clientX - dragX) + Math.abs(e.clientY - dragY);
      dragX = e.clientX;
      dragY = e.clientY;
    }
    mx = e.clientX;
    my = e.clientY;
    if (!dragging) document.body.style.cursor = !onUi(e) && awake() > 0.5 && nodeAt(mx, my, true) >= 0 ? 'pointer' : '';
  };
  const down = (e: PointerEvent) => {
    if (onUi(e)) return;
    const i = awake() > 0.5 ? nodeAt(e.clientX, e.clientY, true) : -1;
    if (i >= 0 && o.onPick) {
      o.onPick(pts[i].run);
      return;
    }
    dragging = true;
    moved = 0;
    dragX = e.clientX;
    dragY = e.clientY;
    document.body.style.cursor = 'grabbing';
  };
  const up = () => {
    if (dragging) document.body.style.cursor = '';
    dragging = false;
  };
  const leave = () => (mx = my = -1e4);
  size();
  window.addEventListener('resize', size);
  window.addEventListener('pointermove', move, { passive: true });
  window.addEventListener('pointerdown', down);
  window.addEventListener('pointerup', up);
  window.addEventListener('pointerleave', leave);

  let rects: DOMRect[] = [];
  let rectsAt = -1e9;
  const behindText = (x: number, y: number) => rects.some((r) => x > r.left - 10 && x < r.right + 10 && y > r.top - 6 && y < r.bottom + 6);
  // soft version: fades the graph behind the page's words over a 60px edge, so there is no hard box
  const veil = (x: number, y: number, floor: number) => {
    let k = 1;
    for (const r of rects) {
      const dx = Math.max(r.left - x, 0, x - r.right), dy = Math.max(r.top - y, 0, y - r.bottom);
      k = Math.min(k, floor + (1 - floor) * Math.min(1, Math.hypot(dx, dy) / 60));
    }
    return k;
  };
  let matched = new Set<string>(), lastQuery: string | null = null;
  // typing reshapes the graph: home positions, and where each matched kind flies to
  const home = pts.map((p) => ({ x: p.x, y: p.y, z: p.z }));
  let pull = new Map<string, { x: number; y: number; z: number }>();
  let pullYaw = 0;
  const layoutPull = () => {
    // the front of the cloud for the current turn, and a sideways axis across it
    pullYaw = yaw;
    const fx = -Math.sin(yaw), fz = -Math.cos(yaw), rx = Math.cos(yaw), rz = -Math.sin(yaw);
    const list = [...matched];
    pull = new Map(
      list.map((k, j) => {
        // centred a little right of the cloud's middle, in the open space beside the headline
        const off = 0.3 + (j - (list.length - 1) / 2) * Math.min(0.42, 1.5 / Math.max(1, list.length));
        const lift = (j % 2 ? 0.12 : -0.12) * (list.length > 3 ? 1 : 0.4);
        return [k, { x: fx * 0.6 + rx * off, y: lift, z: fz * 0.6 + rz * off }];
      }),
    );
  };
  const order = pts.map((_, i) => i);
  const dscr = new Float32Array(dust.length * 3);

  const frame = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const slow = o.still ? 0.25 : 1;
    t += dt * slow;
    pos += ((o.remap ? o.remap(scenePosition(secs)) : scenePosition(secs)) - pos) * (1 - Math.exp(-dt * 3));
    const wake = awake();
    if (now - rectsAt > 250) {
      rectsAt = now;
      rects = Array.from(document.querySelectorAll<HTMLElement>('.scene.in h1, .scene.in h2, .scene.in .field, .scene.in .glass, .scene.in .duel, .say')).map((e) => e.getBoundingClientRect());
    }
    const q = o.query();
    if (q !== lastQuery) {
      lastQuery = q;
      matched = new Set(matchLabels(q, o.library));
      layoutPull();
    }
    const feat = clamp(o.featured(), 0, M - 1);
    const race = sstep(0.5, 1.0, pos);
    const dive = sstep(1.5, 2.0, pos) * (1 - sstep(2.3, 2.9, pos));
    if (race > 0.05 && raceStart < 0) raceStart = t;
    if (race < 0.02) raceStart = -1;
    const shown = raceStart < 0 ? 0 : pos > 2.4 || o.still ? M : clamp(Math.floor((t - raceStart) * 12), 0, M);
    if (shown !== lastShown) {
      lastShown = shown;
      o.onRace?.(shown);
    }

    // camera: the cloud sits right of centre, turns on its own, closes in for the close-up
    // while the reader is typing on the first screen the graph holds still around their kinds
    const shaping = matched.size > 0 ? 1 - sstep(0.4, 0.9, pos) : 0;
    if (!dragging) yaw += spin * dt * slow * (1 - 0.8 * dive) * (1 - 0.9 * shaping);
    const bfx = -Math.sin(pullYaw), bfz = -Math.cos(pullYaw);
    {
      const ease = 1 - Math.exp(-dt * 3.5);
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i], h = home[i];
        const c = centre.get(p.kind), to = pull.get(p.kind);
        let tx = h.x, ty = h.y, tz = h.z;
        if (shaping > 0.01 && p.level < 2) {
          if (to && c) {
            // matched: fly to the front and open up
            tx = mix(h.x, to.x + (h.x - c.x) * 1.6, shaping);
            ty = mix(h.y, to.y + (h.y - c.y) * 1.6, shaping);
            tz = mix(h.z, to.z + (h.z - c.z) * 1.6, shaping);
          } else {
            // the rest steps back and draws in
            tx = h.x * (1 - 0.18 * shaping);
            ty = h.y * (1 - 0.18 * shaping);
            tz = h.z * (1 - 0.18 * shaping);
            tx -= bfx * 0.35 * shaping;
            tz -= bfz * 0.35 * shaping;
          }
        }
        p.x += (tx - p.x) * ease;
        p.y += (ty - p.y) * ease;
        p.z += (tz - p.z) * ease;
      }
    }
    const wide = W > 760;
    camZ = mix(1, 1.6, dive);
    const fp = pts[runPt[feat]];
    // aim: blend the cloud's centre toward the picked node during the close-up
    const baseX = wide ? W * 0.56 : W * 0.5, baseY = H * 0.5;
    cx = baseX;
    cy = baseY;
    if (dive > 0.01) {
      const tmp = new Float32Array(3);
      project(fp.x, fp.y, fp.z, tmp, 0);
      // land the picked node clear of the close-up panel (left) and the robot's bubble (right)
      cx = baseX + (W * 0.7 - tmp[0]) * dive;
      cy = baseY + (H * 0.2 - tmp[1]) * dive;
    }
    for (let i = 0; i < pts.length; i++) project(pts[i].x, pts[i].y, pts[i].z, scr, i * 3);
    for (let i = 0; i < dust.length; i++) project(dust[i].x, dust[i].y, dust[i].z, dscr, i * 3);

    // hover
    const h = wake > 0.5 && !dragging ? nodeAt(mx, my, false) : -1;
    if (h !== hovered) {
      hovered = h;
      o.onHover(h < 0 ? null : pts[h].run >= 0 ? { kind: 'run', index: pts[h].run, x: mx, y: my } : { kind: 'lib', index: pts[h].lib, x: mx, y: my });
    }
    const hoverKind = hovered >= 0 ? pts[hovered].kind : null;

    ctx.clearRect(0, 0, W, H);
    // night: glowing light that adds up; day: solid, deeper colours that read on white
    const day = !o.isNight();
    const tone = (c: string) => (day ? c.split(',').map((v) => Math.round(+v * 0.6)).join(',') : c);
    const white = day ? '20,32,60' : '255,255,255';
    ctx.globalCompositeOperation = day ? 'source-over' : 'lighter';
    const base = 0.3 + 0.7 * wake;
    // dust
    for (let i = 0; i < dust.length; i++) {
      const x = dscr[i * 3], y = dscr[i * 3 + 1], f = dscr[i * 3 + 2];
      ctx.globalAlpha = (0.16 + 0.22 * base) * f * f * Math.min(1, S / 300) * veil(x, y, 0.25);
      ctx.fillStyle = `rgb(${tone(dust[i].col)})`;
      ctx.fillRect(x, y, 1.6 * f, 1.6 * f);
    }
    // kind skeletons
    ctx.lineWidth = 0.6;
    for (const [a, b] of [...ring, ...kindLinks]) {
      const p = pts[a];
      const trunk = p.level === 1 ? 1 : p.level === 2 ? 2 : 0;
      ctx.lineWidth = trunk === 2 ? 1.4 : trunk ? 1 : 0.5;
      const on = matched.has(p.kind) && pos < 0.6 ? 1 : 0;
      ctx.globalAlpha = ((trunk === 2 ? 0.16 : trunk ? 0.22 : 0.06) * base + 0.3 * on + (p.kind === hoverKind ? 0.35 : 0)) * (1 - 0.6 * dive);
      ctx.strokeStyle = `rgb(${tone(p.col)})`;
      ctx.beginPath();
      ctx.moveTo(scr[a * 3], scr[a * 3 + 1]);
      ctx.lineTo(scr[b * 3], scr[b * 3 + 1]);
      ctx.stroke();
    }
    // similarity links between run messages
    ctx.lineWidth = 1;
    for (const [a, b] of links) {
      const hot = a === hovered || b === hovered;
      ctx.globalAlpha = (0.035 * base * race + (hot ? 0.6 : 0)) * (1 - 0.5 * dive);
      ctx.strokeStyle = `rgb(${hot ? white : tone(CYAN)})`;
      ctx.beginPath();
      ctx.moveTo(scr[a * 3], scr[a * 3 + 1]);
      ctx.lineTo(scr[b * 3], scr[b * 3 + 1]);
      ctx.stroke();
    }
    // nodes, far to near
    order.sort((a, b) => scr[a * 3 + 2] - scr[b * 3 + 2]);
    const focus = new Set(o.focus());
    for (const i of order) {
      const p = pts[i];
      const x = scr[i * 3], y = scr[i * 3 + 1], f = scr[i * 3 + 2];
      if (x < -30 || x > W + 30 || y < -30 || y > H + 30) continue;
      let col = p.col, a = 0.45 * base, r = p.r * f * Math.sqrt(camZ);
      if (p.run >= 0) {
        const m = o.messages[p.run];
        if (p.run < shown && race > 0.1) lit[p.run] = Math.min(1, lit[p.run] + dt * 4);
        else lit[p.run] = Math.max(0, lit[p.run] - dt * 2);
        if (lit[p.run] > 0.05) col = !m.agent_ok ? PINK : !m.simpler_ok ? AMBER : CYAN;
        // the wrong answers are the story: bigger, and they pulse
        if (lit[p.run] > 0.05 && (!m.agent_ok || !m.simpler_ok)) r *= 1.7 + 0.25 * Math.sin(t * 4 + p.run);
        a = (0.55 + 0.45 * race) * base + lit[p.run] * 0.3;
        if (dive > 0.3 && p.run !== feat && !o.messages[feat].nb.includes(p.run)) a *= 1 - 0.7 * dive;
        if (p.run === feat) r += 4 * dive;
      } else {
        if (race > 0.3) a *= 0.55;
        if (dive > 0.3) a *= 1 - 0.7 * dive;
      }
      if (pos < 0.6 && matched.has(p.kind)) {
        a = Math.min(1, a + 0.5);
        r *= 1.9 - 0.65 * (1 - shaping);
      } else if (shaping > 0.01 && p.level < 2) a *= 1 - 0.65 * shaping;
      if (p.kind === hoverKind) a = Math.min(1, a + 0.3);
      if (i === hovered) r += 3;
      a *= veil(x, y, 0.3);
      // glow, then core
      ctx.globalAlpha = a * 0.22 * Math.min(1, S / 300);
      ctx.fillStyle = `rgb(${tone(col)})`;
      ctx.beginPath();
      ctx.arc(x, y, r * 2.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      if (p.run >= 0 && (focus.has(p.run) || (p.run === feat && dive > 0.05))) {
        ctx.globalAlpha = (p.run === feat ? dive : wake) * (0.7 + 0.3 * Math.sin(t * 4));
        ctx.strokeStyle = `rgb(${white})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, r + 8, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    // the close-up: a white arc from the picked message through its similar messages
    if (dive > 0.02) {
      const fi = runPt[feat];
      const fx = scr[fi * 3], fy = scr[fi * 3 + 1];
      const cyc = (t * 0.4) % 1;
      ctx.lineWidth = 2.2;
      o.messages[feat].nb.forEach((j, k) => {
        const ji = runPt[j];
        const x1 = scr[ji * 3], y1 = scr[ji * 3 + 1];
        const on = sstep(k * 0.15, k * 0.15 + 0.25, cyc);
        const mxp = (fx + x1) / 2, myp = (fy + y1) / 2 - 60 - 20 * k;
        ctx.globalAlpha = dive * (0.35 + 0.55 * on);
        ctx.strokeStyle = `rgb(${white})`;
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.quadraticCurveTo(mxp, myp, mix(fx, x1, 0.2 + 0.8 * on), mix(fy, y1, 0.2 + 0.8 * on));
        ctx.stroke();
      });
    }
    ctx.globalCompositeOperation = 'source-over';
    // names: the kinds the reader typed, and the hovered kind
    ctx.font = '600 12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    const named = new Set<string>(pos < 0.6 ? matched : []);
    if (hoverKind) named.add(hoverKind);
    const placed: { x: number; y: number; w: number }[] = [];
    ctx.font = '700 11px "JetBrains Mono", monospace';
    famPt.forEach((pi, n) => {
      const x = scr[pi * 3], y = scr[pi * 3 + 1] + 26 * scr[pi * 3 + 2];
      if (behindText(x, y) || dive > 0.5) return;
      ctx.globalAlpha = (0.45 + 0.4 * wake) * scr[pi * 3 + 2];
      ctx.fillStyle = `rgb(${tone(pts[pi].col)})`;
      ctx.fillText(n.toUpperCase(), x, y);
      placed.push({ x, y, w: ctx.measureText(n).width + 16 });
    });
    ctx.font = '600 12px "JetBrains Mono", monospace';
    for (const k of named) {
      const c = centre.get(k);
      if (!c) continue;
      const tmp = new Float32Array(3);
      project(c.x, c.y - 0.13, c.z, tmp, 0);
      const w = ctx.measureText(hubName(k)).width + 16;
      // never over the page's words, and never on top of another name
      if (behindText(tmp[0], tmp[1])) continue;
      while (placed.some((q) => Math.abs(q.x - tmp[0]) < (q.w + w) / 2 && Math.abs(q.y - tmp[1]) < 24)) tmp[1] += 26;
      placed.push({ x: tmp[0], y: tmp[1], w });
      ctx.globalAlpha = 0.95;
      ctx.fillStyle = day ? 'rgba(255,255,255,.85)' : 'rgba(5,8,16,.75)';
      ctx.fillRect(tmp[0] - w / 2, tmp[1] - 15, w, 22);
      ctx.fillStyle = `rgb(${tone(c.col)})`;
      ctx.fillText(hubName(k), tmp[0], tmp[1]);
    }
    ctx.textAlign = 'start';
    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', size);
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerdown', down);
    window.removeEventListener('pointerup', up);
    window.removeEventListener('pointerleave', leave);
    document.body.style.cursor = '';
  };
}
