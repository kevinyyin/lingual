import { Quaternion } from "three";

const alpha = (cutoffHz: number, dt: number) => {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dt);
};

/**
 * One-euro filter on a rotation, time-based (dt from tMs, so 30 and 120 fps agree).
 * speed = angle(prevOut, q)/dt, low-passed at dCutoffHz; cutoff = minCutoff + beta*speed;
 * out = slerp(prevOut, q, alpha(cutoff, dt)), alpha = 1/(1 + tau/dt), tau = 1/(2*pi*cutoff).
 * filter() returns an INTERNAL quaternion that is overwritten on the next call: copy it to keep it.
 */
export class QuatFilter {
  private out = new Quaternion();
  private tmp = new Quaternion();
  private speed = 0;
  private lastT = 0;
  private primed = false;

  private minCutoffHz: number;
  private beta: number;
  private dCutoffHz: number;

  constructor(minCutoffHz = 1.5, beta = 0.5, dCutoffHz = 1.0) { // plain fields: tsc's erasableSyntaxOnly rejects parameter properties
    this.minCutoffHz = minCutoffHz;
    this.beta = beta;
    this.dCutoffHz = dCutoffHz;
  }

  reset() {
    this.primed = false;
    this.speed = 0;
  }

  filter(q: Quaternion, tMs: number): Quaternion {
    if (!this.primed) {
      this.primed = true;
      this.lastT = tMs;
      this.speed = 0;
      return this.out.copy(q);
    }
    const dt = (tMs - this.lastT) / 1000;
    if (!(dt > 0)) return this.out;
    this.lastT = tMs;
    // q and -q are one rotation: pick the hemisphere of the previous output.
    this.tmp.copy(q);
    if (this.tmp.dot(this.out) < 0) {
      this.tmp.set(-this.tmp.x, -this.tmp.y, -this.tmp.z, -this.tmp.w);
    }
    const raw = this.out.angleTo(this.tmp) / dt;
    this.speed += alpha(this.dCutoffHz, dt) * (raw - this.speed);
    const cutoff = this.minCutoffHz + this.beta * Math.abs(this.speed);
    this.out.slerp(this.tmp, alpha(cutoff, dt));
    return this.out;
  }
}
