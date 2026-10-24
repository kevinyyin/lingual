// Last stage of the avatar's motion: every bone's final rotation follows its solved rotation through a critically damped
// spring (a PD controller: the pull is proportional to the angle still to go, the brake to the angular velocity) with a
// cap on angular acceleration. Critical damping never overshoots, so the bone never swings past where the solver put it
// and springs back ("rubberbanding"); the acceleration cap turns a sudden jump of the target (a contact or collision
// correction switching on, a held hand being released) into a short smooth move instead of a snap. It runs after every
// correction, because those switching on and off between frames are what jerked the bones. Natural frequencies sit
// above most signing motion (mostly under 3 Hz), so it removes the fast back-and-forth, not the movement.
import * as THREE from "three";

/** hz: the spring's natural frequency; maxAccel: angular acceleration cap (rad/s^2). */
export interface SpringParams { hz: number; maxAccel: number }

const SUBSTEP = 1 / 240; // integration step (s): stable for natural frequencies well past 20 Hz

/** One bone's rotation spring (local rotation; the angular velocity lives in the parent's frame). */
export class RotationSpring {
  private q: THREE.Quaternion | null = null;
  private w = new THREE.Vector3();
  p: SpringParams;
  constructor(p: SpringParams) {
    this.p = p;
  }

  reset(): void {
    this.q = null;
    this.w.set(0, 0, 0);
  }

  /** The smoothed rotation after dt seconds of following `target`. The first call, a reset, a gap over 0.25 s or a
   *  non-finite input snaps to the target. */
  step(target: THREE.Quaternion, dt: number): THREE.Quaternion {
    if (!Number.isFinite(target.x + target.y + target.z + target.w)) return this.q ? this.q.clone() : target.clone();
    if (!this.q || !(dt > 0) || dt > 0.25) {
      this.q = target.clone();
      this.w.set(0, 0, 0);
      return this.q.clone();
    }
    const w0 = 2 * Math.PI * this.p.hz, n = Math.max(1, Math.ceil(dt / SUBSTEP)), h = dt / n;
    const d = new THREE.Quaternion(), e = new THREE.Vector3(), a = new THREE.Vector3(), turn = new THREE.Quaternion();
    for (let i = 0; i < n; i++) {
      d.copy(target).multiply(this.q.clone().invert()); // rotation still to go, in the parent's frame
      if (d.w < 0) d.set(-d.x, -d.y, -d.z, -d.w); // the short way round
      const s = Math.sqrt(Math.max(0, 1 - d.w * d.w)), angle = 2 * Math.atan2(s, d.w);
      if (s > 1e-9) e.set(d.x / s, d.y / s, d.z / s).multiplyScalar(angle);
      else e.set(0, 0, 0);
      // PD with critical damping: a = w0^2 * error - 2 w0 * velocity, capped
      a.copy(e).multiplyScalar(w0 * w0).addScaledVector(this.w, -2 * w0);
      const am = a.length();
      if (am > this.p.maxAccel) a.multiplyScalar(this.p.maxAccel / am);
      this.w.addScaledVector(a, h); // semi-implicit Euler: velocity first
      const wm = this.w.length();
      if (wm > 1e-12) this.q.premultiply(turn.setFromAxisAngle(this.w.clone().divideScalar(wm), wm * h)).normalize();
    }
    return this.q.clone();
  }
}
