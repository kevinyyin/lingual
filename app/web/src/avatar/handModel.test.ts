import * as THREE from "three";
import { describe, expect, test } from "vitest";
import { angles, capsuleContact, carry, constrainBone, crossOverlap, DIP_COUPLE, FINGER, implausibleHand, jointFrame, limitDir, limitThumbBase, palm, relaxedDir, selfOverlap, THUMB } from "./handModel";

const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const deg = (r: number) => (r * Math.PI) / 180;
// a right hand held up, palm toward the viewer (+z): fingers up (+y), index toward the thumb side (+x)
const RIGHT = palm(v(0, 1, 0), v(1, 0, 0), true);
const LEFT = palm(v(0, 1, 0), v(-1, 0, 0), false); // the left hand mirrored: thumb side -x, palm still toward +z

describe("palm frame", () => {
  test("the palm normal points out of the palm for both hands", () => {
    expect(RIGHT.N.z).toBeCloseTo(1);
    expect(LEFT.N.z).toBeCloseTo(1);
  });
  test("a finger curling toward the palm bends positively, a finger bent back negatively", () => {
    for (const P of [RIGHT, LEFT]) {
      const { p, h } = jointFrame(P, P.F, null, false);
      expect(angles(v(0, Math.cos(deg(40)), Math.sin(deg(40))), P.F, p, h).bend).toBeCloseTo(40);
      expect(angles(v(0, Math.cos(deg(30)), -Math.sin(deg(30))), P.F, p, h).bend).toBeCloseTo(-30);
    }
  });
});

describe("joint limits", () => {
  const { p, h } = jointFrame(RIGHT, RIGHT.F, null, false);
  test("a direction in range is kept", () => {
    const d = v(0, Math.cos(deg(60)), Math.sin(deg(60)));
    const out = limitDir(d, RIGHT.F, p, h, FINGER.pip);
    expect(out.clamped).toBe(false);
    expect(out.dir.distanceTo(d)).toBeLessThan(1e-9);
  });
  test("a middle joint bent backward 60 degrees is straightened to its limit", () => {
    const out = limitDir(v(0, Math.cos(deg(60)), -Math.sin(deg(60))), RIGHT.F, p, h, FINGER.pip);
    expect(out.clamped).toBe(true);
    expect(angles(out.dir, RIGHT.F, p, h).bend).toBeCloseTo(FINGER.pip.bend[0]);
    expect(out.dir.length()).toBeCloseTo(1);
  });
  test("a hinge joint bent sideways 40 degrees keeps only its tolerated sideways bend", () => {
    const out = limitDir(v(Math.sin(deg(40)), Math.cos(deg(40)), 0), RIGHT.F, p, h, FINGER.pip);
    expect(Math.abs(angles(out.dir, RIGHT.F, p, h).side)).toBeCloseTo(FINGER.pip.side);
  });
  test("the thumb metacarpal is kept in its cone", () => {
    const behind = v(0.3, 0.2, -0.9).normalize(); // far behind the palm plane
    const out = limitThumbBase(behind, RIGHT);
    expect(out.clamped).toBe(true);
    expect((Math.asin(out.dir.dot(RIGHT.N)) * 180) / Math.PI).toBeCloseTo(THUMB.cmc.palmar[0]);
  });
  test("the hinge carried down a finger turns with the knuckle's sideways swing", () => {
    const spread = v(Math.sin(deg(20)), Math.cos(deg(20)), 0); // the knuckle swung 20 degrees toward the thumb
    const hinge = carry(RIGHT.S, RIGHT.F, spread);
    expect(hinge.dot(spread)).toBeCloseTo(0);
    // so the middle joint of the spread finger bends in its own plane: curling it toward the palm is all bend, no side
    const { p: p2, h: h2 } = jointFrame(RIGHT, spread, hinge, false);
    const curl = spread.clone().multiplyScalar(Math.cos(deg(50))).add(v(0, 0, Math.sin(deg(50))));
    const a = angles(curl, spread, p2, h2);
    expect(a.bend).toBeCloseTo(50);
    expect(a.side).toBeCloseTo(0);
  });
  test("constrainBone reports how far past the range a direction was", () => {
    const out = constrainBone(RIGHT, 1, 1, RIGHT.F, RIGHT.S, v(0, Math.cos(deg(50)), -Math.sin(deg(50))));
    expect(out.outBy).toBeCloseTo(50 + FINGER.pip.bend[0]);
  });
});

/** 21 landmarks of a right hand, palm toward +z, fingers up, each finger's middle joint bent by `pip` degrees. */
function hand(pip: number): THREE.Vector3[] {
  const pts: THREE.Vector3[] = Array.from({ length: 21 }, () => v(0, 0, 0));
  const xs = [0.03, 0.01, -0.01, -0.03];
  [5, 9, 13, 17].forEach((m, i) => {
    const base = v(xs[i], 0.09, 0);
    pts[m] = base;
    pts[m + 1] = base.clone().add(v(0, 0.04, 0));
    const d = v(0, Math.cos(deg(pip)), Math.sin(deg(pip))).multiplyScalar(0.025);
    pts[m + 2] = pts[m + 1].clone().add(d);
    pts[m + 3] = pts[m + 2].clone().add(d);
  });
  for (let k = 1; k <= 4; k++) pts[k] = v(0.04 + 0.01 * k, 0.02 * k, 0.01);
  return pts;
}

describe("implausible hands", () => {
  test("a fist is a hand", () => expect(implausibleHand(hand(80), true)).toBe(false));
  test("middle joints bent backward 70 degrees are a mirrored or garbage hand", () => expect(implausibleHand(hand(-70), true)).toBe(true));
});

describe("capsule overlap", () => {
  const cap = (x: number, finger: number) => ({ a: v(x, 0, 0), b: v(x, 0.04, 0), r: 0.0085, finger });
  test("neighbouring fingers 1 cm apart overlap by the radii minus the gap and the slack", () => {
    expect(selfOverlap([cap(0, 1), cap(0.01, 2)])).toBeCloseTo(0.017 - 0.003 - 0.01, 6);
  });
  test("bones of one finger are not compared, and separated fingers do not overlap", () => {
    expect(selfOverlap([cap(0, 1), cap(0.001, 1)])).toBe(0);
    expect(selfOverlap([cap(0, 1), cap(0.03, 2)])).toBe(0);
  });
  test("two hands far apart do not overlap", () => expect(crossOverlap([cap(0, 1)], [cap(0.5, 1)])).toBe(0));
  test("contact pushes one capsule straight out of the other by the overlap", () => {
    const c = capsuleContact(cap(0.01, 1), cap(0, 2));
    expect(c.depth).toBeCloseTo(0.004, 6);
    expect(c.dir.x).toBeCloseTo(1);
    const moved = cap(0.01 + c.depth, 1);
    expect(capsuleContact(moved, cap(0, 2)).depth).toBeCloseTo(0, 6);
  });
});

describe("finger joints", () => {
  const { p, h } = jointFrame(RIGHT, RIGHT.F, null, false);
  test("the middle and end joints are hinges: any sideways bend is removed, curl kept", () => {
    for (const k of [1, 2]) {
      const d = v(Math.sin(deg(25)), Math.cos(deg(25)) * Math.cos(deg(40)), Math.cos(deg(25)) * Math.sin(deg(40)));
      const out = constrainBone(RIGHT, 2, k, RIGHT.F, RIGHT.S, d, k === 2 ? 40 : undefined);
      const a = angles(out.dir, RIGHT.F, p, h);
      expect(a.side).toBeCloseTo(0);
      expect(a.bend).toBeGreaterThan(30);
    }
  });
  test("the middle joints and the thumb never bend backward; the end joints give only a little", () => {
    expect(FINGER.pip.bend[0]).toBe(0);
    expect(THUMB.mcp.bend[0]).toBe(0);
    expect(THUMB.ip.bend[0]).toBe(0);
    expect(FINGER.dip.bend[0]).toBeGreaterThanOrEqual(-10);
  });
  test("the end joint follows the middle joint's curl", () => {
    const straight = v(0, 1, 0); // the end joint asked to stay straight while the middle joint is curled 80 degrees
    const out = constrainBone(RIGHT, 1, 2, RIGHT.F, RIGHT.S, straight, 80);
    expect(out.bendOut).toBeGreaterThanOrEqual(DIP_COUPLE[0] * 80 - 8 - 1e-6);
    const curled = constrainBone(RIGHT, 1, 2, RIGHT.F, RIGHT.S, v(0, Math.cos(deg(80)), Math.sin(deg(80))), 10); // end joint 80 with the middle at 10
    expect(curled.bendOut).toBeLessThanOrEqual(DIP_COUPLE[1] * 10 + 8 + 1e-6);
  });
  test("a curled knuckle spreads less than a straight one", () => {
    const spread = (curl: number) => {
      const d = v(Math.sin(deg(30)), Math.cos(deg(30)) * Math.cos(deg(curl)), Math.cos(deg(30)) * Math.sin(deg(curl)));
      return Math.abs(angles(constrainBone(RIGHT, 1, 0, RIGHT.F, RIGHT.S, d).dir, RIGHT.F, p, h).side);
    };
    expect(spread(0)).toBeCloseTo(FINGER.mcp.side, 0);
    expect(spread(85)).toBeLessThan(FINGER.mcp.side * 0.5);
  });
  test("the relaxed hand is within every joint's range, fingers lightly curled", () => {
    for (const P of [RIGHT, LEFT]) for (let c = 0; c < 5; c++) {
      let r = c === 0 ? P.F : P.F.clone(), hinge: THREE.Vector3 | null = null, bend: number | undefined;
      for (let k = 0; k < 3; k++) {
        const out = relaxedDir(P, c, k, r, hinge);
        const check = constrainBone(P, c, k, r, hinge, out.dir, bend);
        expect(check.outBy).toBeLessThan(0.5);
        if (c > 0) expect(check.bendOut).toBeGreaterThan(5);
        bend = check.bendOut;
        hinge = out.hinge;
        r = out.dir;
      }
    }
  });
});
