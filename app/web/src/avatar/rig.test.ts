import * as THREE from "three";
import { describe, expect, test } from "vitest";
import { Body } from "./body";
import { aimLocal, frameRotation, joint, toThree, twistAbout, twoBoneElbow } from "./rig";

const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const close = (a: THREE.Vector3, b: THREE.Vector3) => expect(a.distanceTo(b)).toBeLessThan(1e-6);

/** shoulder -> elbow -> wrist chain along +x, under a rotated parent (like a rig's odd bind axes) */
function arm() {
  const root = new THREE.Object3D();
  root.quaternion.setFromEuler(new THREE.Euler(0.3, -0.7, 0.2));
  const upper = new THREE.Object3D(), fore = new THREE.Object3D(), hand = new THREE.Object3D();
  upper.quaternion.setFromEuler(new THREE.Euler(0.1, 0.4, -0.2));
  fore.position.set(0.3, 0, 0);
  hand.position.set(0.25, 0, 0);
  root.add(upper);
  upper.add(fore);
  fore.add(hand);
  root.updateMatrixWorld(true);
  return { root, upper, fore, hand };
}
const world = (o: THREE.Object3D) => (o.updateWorldMatrix(true, false), o.getWorldPosition(new THREE.Vector3()));

describe("rig", () => {
  test("toThree flips y and z (MediaPipe y down, z away -> three.js y up, z toward the viewer)", () => {
    close(toThree([0.1, 0.2, 0.3]), v(0.1, -0.2, -0.3));
  });

  test("aimLocal points a bone along a world direction whatever its parent's and bind rotation", () => {
    const { upper, fore } = arm();
    const j = joint(upper, fore);
    for (const d of [v(0, -1, 0), v(0.5, 0.5, 0.7).normalize(), v(-1, 0, 0)]) {
      upper.quaternion.copy(aimLocal(j, d));
      const dir = world(fore).sub(world(upper)).normalize();
      close(dir, d);
    }
  });

  test("frameRotation maps one palm frame onto another (forward and side)", () => {
    const f0 = v(1, 0, 0), s0 = v(0, 0, 1), f = v(0, 1, 0), s = v(1, 0, 0);
    const q = frameRotation(f0, s0, f, s);
    close(f0.clone().applyQuaternion(q), f);
    close(s0.clone().applyQuaternion(q), s);
  });

  test("twistAbout keeps only the rotation about the axis", () => {
    const axis = v(1, 0, 0);
    const twist = new THREE.Quaternion().setFromAxisAngle(axis, 0.8);
    const swing = new THREE.Quaternion().setFromAxisAngle(v(0, 1, 0), 0.5);
    const t = twistAbout(swing.clone().multiply(twist), axis);
    expect(Math.abs(t.angleTo(twist))).toBeLessThan(1e-6);
  });

  test("twoBoneElbow reaches the target with the bone lengths and keeps the elbow's side", () => {
    const s = v(0, 0, 0), e = v(0.2, -0.2, -0.1), target = v(0.3, 0.1, 0.2);
    const el = twoBoneElbow(s, e, target, 0.3, 0.25);
    expect(el.distanceTo(s)).toBeCloseTo(0.3, 6);
    expect(el.distanceTo(target)).toBeCloseTo(0.25, 6);
    const n = target.clone().normalize();
    const side = (p: THREE.Vector3) => p.clone().sub(n.clone().multiplyScalar(p.dot(n)));
    expect(side(el).dot(side(e))).toBeGreaterThan(0); // same bend plane side as the signer's elbow
  });

  test("an unreachable target is reached as far as the arm allows", () => {
    const el = twoBoneElbow(v(0, 0, 0), v(0, -0.1, 0.1), v(2, 0, 0), 0.3, 0.25);
    expect(el.distanceTo(v(0, 0, 0))).toBeCloseTo(0.3, 6);
    expect(el.x).toBeGreaterThan(0.29);
  });
});

describe("body", () => {
  // a 30 x 20 cm torso column from y 1.0 to 1.5 m, and a head sphere of 10 cm at 1.65 m
  const ring = (y: number) => Array.from({ length: 24 }, (_, i) => v(0.15 * Math.cos((i / 24) * 2 * Math.PI), y, 0.1 * Math.sin((i / 24) * 2 * Math.PI)));
  const torso = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5].flatMap(ring);
  const head = [v(0.1, 1.65, 0), v(-0.1, 1.65, 0), v(0, 1.75, 0), v(0, 1.55, 0), v(0, 1.65, 0.1), v(0, 1.65, -0.1)];
  const body = Body.fromPoints(torso, head);

  test("a hand inside the chest is pushed forward to the surface plus the margin", () => {
    const d = body.push(v(0, 1.25, 0.05), 0.01);
    const out = v(0, 1.25, 0.05).add(d);
    expect(d.y).toBe(0);
    expect(out.z).toBeGreaterThan(0.1);
    expect(body.push(out, 0.0099).length()).toBeLessThan(1e-6);
  });

  test("a hand already outside is left alone", () => {
    expect(body.push(v(0, 1.25, 0.2), 0.01).length()).toBe(0);
    expect(body.push(v(0, 0.5, 0), 0.01).length()).toBe(0); // below the torso
  });

  test("a hand inside the face is pushed out of the head", () => {
    const p = v(0, 1.62, 0.07);
    const out = p.clone().add(body.push(p, 0.01));
    expect(out.distanceTo(v(0, 1.65, 0))).toBeGreaterThan(0.1);
  });

  test("the front surface is where a hand moved straight back first touches", () => {
    expect(body.frontZ(0, 1.25)).toBeCloseTo(0.1, 2); // chest centre
    expect(body.frontZ(0.12, 1.25)).toBeCloseTo(0.06, 2); // toward the side the torso curves away
    expect(body.frontZ(0, 1.65)).toBeCloseTo(0.1, 2); // face
    expect(body.frontZ(0.4, 1.25)).toBeNull(); // beside the body
  });

  test("a flat chest is followed off-centre, where an ellipse would let a hand sink in", () => {
    // a box torso 30 cm wide with a flat front at z = 0.1 (vertices every 1 cm)
    const pts: THREE.Vector3[] = [];
    for (let y = 1.0; y <= 1.5; y += 0.01) for (let x = -0.15; x <= 0.15; x += 0.01) pts.push(v(x, y, 0.1), v(x, y, -0.1));
    for (let y = 1.0; y <= 1.5; y += 0.01) for (let z = -0.1; z <= 0.1; z += 0.01) pts.push(v(-0.15, y, z), v(0.15, y, z));
    const box = Body.fromPoints(pts, []);
    expect(box.frontZ(0.13, 1.25)).toBeCloseTo(0.1, 2);
    const hand = v(0.13, 1.25, 0.08); // 2 cm inside the shirt, though outside the slice ellipse
    expect(hand.clone().add(box.push(hand, 0.01)).z).toBeCloseTo(0.11, 3);
  });
});
