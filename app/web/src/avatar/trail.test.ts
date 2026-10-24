import { expect, test } from "vitest";
import { arrowHead, ndcToCanvas, TrailPath } from "./trail";

const H = 300; // canvas height: speeds are canvas heights per second
const move = (t: TrailPath, signing = true, speed = 1) => { for (let i = 0; i <= 10; i++) t.push(i * 16, [100 + i * 6, 100], signing, speed); };

test("a still hand leaves nothing", () => {
  const t = new TrailPath({ height: H });
  for (let i = 0; i < 30; i++) t.push(i * 16, [100, 100], true);
  expect(t.visible(480)).toEqual([]);
});

test("a moving hand leaves a trail that fades out over 700 ms", () => {
  const t = new TrailPath({ height: H });
  move(t);
  const v = t.visible(160);
  expect(v.length).toBeGreaterThan(5);
  expect(v[0].alpha).toBeLessThan(v.at(-1)!.alpha);
  expect(t.visible(160 + 700)).toEqual([]);
});

test("slow replay stretches the fade", () => {
  const t = new TrailPath({ height: H });
  move(t, true, 0.5);
  expect(t.visible(160 + 700).length).toBeGreaterThan(0);
  expect(t.visible(160 + 1400)).toEqual([]);
});

test("slow replay keeps a slow hand's trail: speed is judged in the sign's own time", () => {
  const t = new TrailPath({ height: H });
  for (let i = 0; i <= 10; i++) t.push(i * 16, [100 + i * 0.48, 100], true, 0.5); // 30 px/s on screen is 60 px/s in the sign
  expect(t.visible(160).length).toBeGreaterThan(5);
});

test("samples outside a sign are ignored", () => {
  const t = new TrailPath({ height: H });
  move(t, false);
  expect(t.visible(160)).toEqual([]);
});

test("when a sign ends, its last stroke holds 600 ms with an arrowhead along it, then fades", () => {
  const t = new TrailPath({ height: H });
  move(t);
  t.push(176, [166, 100], false);
  expect(t.end(176)).toMatchObject({ tip: [160, 100] });
  expect(t.end(176)!.dir[0]).toBeCloseTo(1);
  expect(t.visible(176 + 600).length).toBe(t.visible(176).length);
  expect(t.end(176 + 600 + 700)).toBeNull();
});

test("a new sign clears the last one's arrowhead", () => {
  const t = new TrailPath({ height: H });
  move(t);
  t.push(176, [166, 100], false);
  t.push(400, [100, 100], true);
  expect(t.end(400)).toBeNull();
});

test("still mode (reduced motion) draws nothing while signing, then the whole path for 1.5 s", () => {
  const t = new TrailPath({ height: H, still: true });
  move(t);
  expect(t.visible(160)).toEqual([]);
  t.push(176, [166, 100], false);
  expect(t.visible(176)).toHaveLength(10); // the first sample has no speed yet
  expect(t.visible(176).every((s) => s.alpha === 1)).toBe(true);
  expect(t.visible(176 + 1500)).toEqual([]);
});

test("arrowHead's arms sit 28 degrees either side of the backward direction", () => {
  const [a, b] = arrowHead([0, 0], [1, 0], 10);
  expect(a[0]).toBeCloseTo(-10 * Math.cos((28 * Math.PI) / 180));
  expect(b[0]).toBeCloseTo(a[0]);
  expect(a[1]).toBeCloseTo(-b[1]);
});

test("ndcToCanvas maps clip space to canvas pixels and drops points outside the depth range", () => {
  expect(ndcToCanvas({ x: 0, y: 0, z: 0.5 }, 400, 300)).toEqual([200, 150]);
  expect(ndcToCanvas({ x: -1, y: 1, z: 0.5 }, 400, 300)).toEqual([0, 0]);
  expect(ndcToCanvas({ x: 0, y: 0, z: 1.2 }, 400, 300)).toBeNull();
});

test("a sign whose last stroke is under 24 px (two arrowhead arms) leaves nothing behind when it ends", () => {
  const t = new TrailPath({ height: H });
  for (let i = 0; i <= 10; i++) t.push(i * 16, [100 + i * 2, 100], true);
  t.push(176, [120, 100], false);
  expect(t.end(176)).toBeNull();
  expect(t.visible(176)).toEqual([]);
});

test("the arrowhead points along the last 16 px of path, not the last step's jitter", () => {
  const t = new TrailPath({ height: H });
  move(t);
  t.push(176, [161, 97], true);
  t.push(192, [161, 97], false);
  expect(t.end(192)!.dir[0]).toBeGreaterThan(0.9);
});
