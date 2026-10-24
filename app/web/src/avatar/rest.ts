// Arms-down rest pose used until signs/_REST.json (from a real reference video) exists.
import type { ClipFrame } from "./player";

const world: number[][] = Array.from({ length: 33 }, () => [0, -0.6, -0.1]); // face points near the nose
const set: [number, number, number, number][] = [
  // arms relaxed at the sides: elbows a little bent, hands a little forward and in, in front of the thighs (dead
  // straight arms read as a mannequin); wrists stay below the hip joints and clear of them (at hip height and close
  // in, the body push turned the rest into hands-on-hips)
  [0, 0, -0.65, -0.1], [11, 0.18, -0.5, 0], [12, -0.18, -0.5, 0], [13, 0.21, -0.22, 0.03], [14, -0.21, -0.22, 0.03],
  [15, 0.18, 0.03, -0.11], [16, -0.18, 0.03, -0.11], [17, 0.18, 0.09, -0.14], [18, -0.18, 0.09, -0.14],
  [19, 0.17, 0.1, -0.14], [20, -0.17, 0.1, -0.14], [21, 0.16, 0.03, -0.13], [22, -0.16, 0.03, -0.13], [23, 0.1, 0, 0], [24, -0.1, 0, 0], [25, 0.1, 0.45, 0], [26, -0.1, 0.45, 0],
  [27, 0.1, 0.9, 0], [28, -0.1, 0.9, 0], [29, 0.1, 0.95, 0], [30, -0.1, 0.95, 0], [31, 0.1, 0.95, -0.1], [32, -0.1, 0.95, -0.1],
];
for (const [i, x, y, z] of set) world[i] = [x, y, z];

export const DEFAULT_REST: ClipFrame = {
  pose: world,
  pose2d: world.map(([x, y]) => [0.5 + x * 0.8, 0.75 + y * 0.8]),
  left: null,
  right: null,
};
