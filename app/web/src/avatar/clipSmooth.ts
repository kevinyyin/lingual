// Clips are cleaned once as they load: the tracker's frame-to-frame wobble is removed by smoothing that looks both ways
// in time, so it neither lags nor blunts the real movement (the live filters in rocketbox.ts can only look back).
import type { Clip, ClipFrame } from "./player";

const KEYS = ["pose", "pose2d", "left", "right", "leftW", "rightW"] as const;

/** Every point of every frame averaged 1-2-1 with the frames either side. A frame without both neighbours (a clip's
 *  ends, either side of a missing hand) keeps its values, so a steady movement stays exactly as it was. */
export function smoothClip(clip: Clip): Clip {
  const F = clip.frames;
  return {
    ...clip,
    frames: F.map((f, t) => {
      const out: ClipFrame = { ...f };
      for (const k of KEYS) {
        const a = F[t - 1]?.[k], b = f[k], c = F[t + 1]?.[k];
        if (a && b && c && a.length === b.length && c.length === b.length) {
          out[k] = b.map((p, i) => p.map((v, j) => (a[i][j] + 2 * v + c[i][j]) / 4));
        }
      }
      return out;
    }),
  };
}
