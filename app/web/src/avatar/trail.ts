// The hand trail: while the avatar signs in teaching mode, each moving hand leaves a short go stroke along its path,
// the way ASL dictionaries draw a sign's movement. TrailPath is the pure part (which samples count, how they fade);
// attachTrail draws it on a 2D canvas laid over the avatar's.
import type { StageFrame } from "./stage";

export type Pt = [number, number];

const CAP = 1024; // samples per hand: one sign at 60 fps is far fewer
const HEAD = 12; // arrowhead arm, px
const DIR_PX = 16; // the arrowhead follows this much of the path's end, so a last jitter doesn't turn it

/** One hand's trail. Times are ms; `minSpeed` is in canvas heights per second, so a resting base hand leaves nothing. */
export class TrailPath {
  height: number;
  private readonly fadeMs: number;
  private readonly minSpeed: number;
  private readonly holdMs: number;
  private readonly stillMs: number;
  private readonly still: boolean;
  private readonly ts = new Float64Array(CAP);
  private readonly xs = new Float64Array(CAP);
  private readonly ys = new Float64Array(CAP);
  private start = 0;
  private len = 0;
  private last: { t: number; x: number; y: number } | null = null;
  private speed = 1;
  private inSign = false;
  private endAt: number | null = null;

  constructor(o: { height: number; fadeMs?: number; minSpeed?: number; holdMs?: number; stillMs?: number; still?: boolean }) {
    this.height = o.height;
    this.fadeMs = o.fadeMs ?? 700;
    this.minSpeed = o.minSpeed ?? 0.15;
    this.holdMs = o.holdMs ?? 600;
    this.stillMs = o.stillMs ?? 1500;
    this.still = o.still ?? false;
  }

  /** One frame: the palm at `p` (null when out of view), whether a sign itself is playing, and the playback speed. */
  push(now: number, p: Pt | null, signing: boolean, speed = 1): void {
    this.speed = speed;
    if (signing && this.endAt !== null) this.clear(); // a new sign starts clean
    if (!signing) {
      if (this.inSign) {
        this.inSign = false;
        this.endAt = now;
        if (this.pathLength(now) < 2 * HEAD) this.clear(); // a last stroke shorter than its arrowhead reads as a glitch
      }
    } else {
      this.inSign = true;
      if (p && this.last) {
        const secs = (now - this.last.t) / 1000;
        const moved = Math.hypot(p[0] - this.last.x, p[1] - this.last.y) / this.height;
        if (secs > 0 && moved / secs >= this.minSpeed * speed) this.add(now, p); // in the sign's own time, so slow replay keeps it
      }
    }
    this.last = p ? { t: now, x: p[0], y: p[1] } : null;
  }

  /** Calls `fn` for each sample still showing at `now`, oldest first (alpha 0-1); nothing is allocated. */
  each(now: number, fn: (x: number, y: number, alpha: number) => void): void {
    if (this.still) {
      if (this.endAt === null || this.inSign || now - this.endAt >= this.stillMs) return;
      for (let k = 0; k < this.len; k++) { const i = (this.start + k) % CAP; fn(this.xs[i], this.ys[i], 1); }
      return;
    }
    const at = this.clock(now), fade = this.fadeMs / this.speed;
    for (let k = 0; k < this.len; k++) {
      const i = (this.start + k) % CAP, alpha = 1 - (at - this.ts[i]) / fade;
      if (alpha > 0) fn(this.xs[i], this.ys[i], Math.min(1, alpha));
    }
  }

  visible(now: number): { p: Pt; alpha: number }[] {
    const out: { p: Pt; alpha: number }[] = [];
    this.each(now, (x, y, alpha) => out.push({ p: [x, y], alpha }));
    return out;
  }

  /** After a sign ends: its last sample, the direction of its last DIR_PX of path, and how visible it still is. */
  end(now: number): { tip: Pt; dir: Pt; alpha: number } | null {
    if (this.endAt === null || this.len < 2) return null;
    const b = (this.start + this.len - 1) % CAP;
    let a = b;
    for (let k = this.len - 2, run = 0; k >= 0 && run < DIR_PX; k--) {
      const p = (this.start + k) % CAP;
      run += Math.hypot(this.xs[a] - this.xs[p], this.ys[a] - this.ys[p]);
      a = p;
    }
    const dx = this.xs[b] - this.xs[a], dy = this.ys[b] - this.ys[a], d = Math.hypot(dx, dy) || 1;
    const alpha = this.still
      ? (now - this.endAt < this.stillMs ? 1 : 0)
      : 1 - (this.clock(now) - this.ts[b]) / (this.fadeMs / this.speed);
    return alpha > 0 ? { tip: [this.xs[b], this.ys[b]], dir: [dx / d, dy / d], alpha: Math.min(1, alpha) } : null;
  }

  clear(): void {
    this.start = this.len = 0;
    this.endAt = null;
  }

  private pathLength(now: number): number {
    let len = 0, px = NaN, py = NaN;
    this.each(now, (x, y) => { if (px === px) len += Math.hypot(x - px, y - py); px = x; py = y; });
    return len;
  }

  /** The fade's clock: it stops for `holdMs` when the sign ends, so its last stroke stays a moment before fading. */
  private clock(now: number): number {
    if (this.endAt === null || now <= this.endAt) return now;
    return now <= this.endAt + this.holdMs ? this.endAt : now - this.holdMs;
  }

  private add(t: number, p: Pt): void {
    const i = (this.start + this.len) % CAP;
    this.ts[i] = t; this.xs[i] = p[0]; this.ys[i] = p[1];
    if (this.len < CAP) this.len++;
    else this.start = (this.start + 1) % CAP;
  }
}

/** The two arm ends of an arrowhead at `tip` pointing along `dir` (a unit vector), `spreadDeg` either side. */
export function arrowHead(tip: Pt, dir: Pt, len: number, spreadDeg = 28): [Pt, Pt] {
  const r = (spreadDeg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r), bx = -dir[0], by = -dir[1];
  return [
    [tip[0] + len * (bx * c - by * s), tip[1] + len * (bx * s + by * c)],
    [tip[0] + len * (bx * c + by * s), tip[1] + len * (-bx * s + by * c)],
  ];
}

/** A point in clip space (three.js Vector3.project) to canvas CSS pixels; null outside the camera's depth range. */
export function ndcToCanvas(ndc: { x: number; y: number; z: number }, w: number, h: number): Pt | null {
  if (ndc.z < -1 || ndc.z > 1) return null;
  return [((ndc.x + 1) / 2) * w, ((1 - ndc.y) / 2) * h];
}

const GO = "#F2592A";

/** Draws both hands' trails on `canvas` (a 2D canvas over the avatar's), fed one stage frame at a time. */
export function attachTrail(canvas: HTMLCanvasElement, palms: () => Record<"L" | "R", Pt | null> | undefined, opts: { still: boolean }) {
  const hands = [new TrailPath({ height: 1, still: opts.still }), new TrailPath({ height: 1, still: opts.still })];
  const ctx = canvas.getContext("2d");
  let raf = 0;
  const draw = () => {
    raf = 0;
    if (!ctx || !canvas.isConnected) return; // off the page: nothing to draw, and the loop stops
    const now = performance.now(), dpr = Math.min(devicePixelRatio, 2), w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = GO;
    ctx.lineCap = ctx.lineJoin = "round";
    let alive = false;
    for (const t of hands) {
      let px = NaN, py = NaN;
      // Fading thins the line instead of making it see-through: see-through segments overlap into a bead at every joint.
      t.each(now, (x, y, alpha) => {
        if (px === px) { ctx.lineWidth = 4 * alpha; ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y); ctx.stroke(); }
        px = x; py = y;
        alive = true;
      });
      const e = t.end(now);
      if (e) {
        const [a, b] = arrowHead(e.tip, e.dir, HEAD);
        ctx.lineWidth = 4 * e.alpha;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(e.tip[0], e.tip[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        alive = true;
      }
    }
    if (alive) raf = requestAnimationFrame(draw);
  };
  return {
    frame(f: StageFrame): void {
      const p = palms();
      hands[0].height = hands[1].height = canvas.clientHeight || 1;
      hands[0].push(f.now, p?.L ?? null, f.signing, f.speed);
      hands[1].push(f.now, p?.R ?? null, f.signing, f.speed);
      if (!raf) raf = requestAnimationFrame(draw);
    },
    clear(): void {
      for (const t of hands) t.clear();
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    },
  };
}
