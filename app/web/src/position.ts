// The camera's framing check, instead of a calibrate button: it tells the signer how to sit, the backend calibrates
// only on frames where they're in position, and it calibrates again by itself when they move. Image coordinates are
// MediaPipe's, unmirrored (0-1, y down).
import type { Frame } from "./ws";

const SEEN = 0.5; // pose visibility that counts as in the picture
const DRIFT_MS = 1500; // sat elsewhere this long: calibrate again (a lean while signing is shorter)

/** What the signer should do to be read well, or null when they're in position. `rest`: the backend is learning the
 *  resting pose (calibrating), so the hands must be down too. */
export function positionHint(f: Frame, rest: boolean): string | null {
  const [nose, ls, rs] = [f.pose[0], f.pose[11], f.pose[12]];
  if (nose[3] < SEEN || ls[3] < SEEN || rs[3] < SEEN) return "Get your face and shoulders in the picture";
  const width = Math.abs(ls[0] - rs[0]), mid = (ls[0] + rs[0]) / 2;
  if (width > 0.6) return "Move back a little";
  if (width < 0.2) return "Come a little closer";
  if (mid < 0.3 || mid > 0.7) return "Scoot to the middle of the picture";
  // the rest line is learned from where the wrists sit now: hands held at the chest would hide signs made there
  if (rest && Math.min(f.pose[15][1], f.pose[16][1]) < (ls[1] + rs[1]) / 2 + 0.25) return "Rest your hands in your lap";
  return null;
}

/** Notices when the signer has sat clearly elsewhere (closer, farther, to a side) since the last calibration. */
export class PositionWatch {
  private ref: { width: number; mid: number } | null = null;
  private since: number | null = null; // when the current drift began

  /** One frame while the backend is ready; true when it's time to calibrate again. */
  moved(f: Frame, now: number): boolean {
    const [ls, rs] = [f.pose[11], f.pose[12]];
    const box = { width: Math.abs(ls[0] - rs[0]), mid: (ls[0] + rs[0]) / 2 };
    if (!this.ref) {
      this.ref = box;
      return false;
    }
    if (Math.abs(box.width / this.ref.width - 1) <= 0.25 && Math.abs(box.mid - this.ref.mid) <= 0.15) {
      this.since = null;
      return false;
    }
    this.since ??= now;
    return now - this.since >= DRIFT_MS;
  }

  /** Calibrating again: the next ready frame is the new reference. */
  reset(): void {
    this.ref = this.since = null;
  }

  /** One camera frame, given the backend's state: the hint to show (null: none; never mid-sign), whether the backend
   *  should get the frame (calibration learns only from frames in position), and whether to calibrate again. */
  check(f: Frame, state: string | null, now: number): { hint: string | null; send: boolean; recalibrate: boolean } {
    const calibrating = state === "calibrating";
    if (calibrating) this.reset();
    const hint = calibrating || state === "ready" ? positionHint(f, calibrating) : null;
    const recalibrate = state === "ready" && this.moved(f, now);
    if (recalibrate) this.reset(); // asked once: frames until the backend says "calibrating" start a new reference
    return { hint, send: !(calibrating && hint), recalibrate };
  }
}
