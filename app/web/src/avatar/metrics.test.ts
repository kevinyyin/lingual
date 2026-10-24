import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { edgeMotion, elbowFlexion, frameMetrics, hingeAxis } from "./metrics";
import type { ArmState } from "./rocketbox";

const arm = (o: Partial<ArmState> = {}): ArmState => ({
  upper: new THREE.Quaternion(), fore: new THREE.Quaternion(), hand: new THREE.Quaternion(),
  elbowDeg: 0, twistDeg: 0, wrist: new THREE.Vector3(), elbow: new THREE.Vector3(), ...o,
});
const both = (L = arm(), R = arm()) => ({ L, R });
const turned = (deg: number) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (deg * Math.PI) / 180);
const forearm = (x: number) => ({ elbow: new THREE.Vector3(x, 0, 0), wrist: new THREE.Vector3(x, 0, 0.3) });

describe("frameMetrics", () => {
  it("reports the largest per-bone turn", () => {
    expect(frameMetrics(both(), both()).maxStepDeg).toBeCloseTo(0, 6);
    expect(frameMetrics(both(), both(arm(), arm({ fore: turned(30) }))).maxStepDeg).toBeCloseTo(30, 2);
    expect(frameMetrics(null, both()).maxStepDeg).toBe(0);
  });
  it("flags a backward elbow below -5 degrees", () => {
    expect(frameMetrics(null, both(arm({ elbowDeg: -6 }))).backwardElbow).toBe(true);
    expect(frameMetrics(null, both(arm({ elbowDeg: -4 }))).backwardElbow).toBe(false);
  });
  it("flags twist beyond 90 degrees", () => {
    expect(frameMetrics(null, both(arm({ twistDeg: 91 }))).twistOut).toBe(true);
    expect(frameMetrics(null, both(arm({ twistDeg: -89 }))).twistOut).toBe(false);
  });
  it("armGap is the forearm segment distance", () => {
    expect(frameMetrics(null, both(arm(forearm(0)), arm(forearm(0.2)))).armGap).toBeCloseTo(0.2, 6);
    const cross = arm({ elbow: new THREE.Vector3(-0.1, 0, 0.15), wrist: new THREE.Vector3(0.1, 0, 0.15) });
    expect(frameMetrics(null, both(arm(forearm(0)), cross)).armGap).toBeCloseTo(0, 6);
  });
});

describe("elbow flexion", () => {
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
  it("derives the hinge from a bent bind arm, A-posed or T-posed, and signs the fold toward the front", () => {
    for (const drop of [0, 40]) { // upper arm along +x, dropped by `drop` degrees; forearm bent slightly forward
      const du = v(Math.cos((drop * Math.PI) / 180), -Math.sin((drop * Math.PI) / 180), 0);
      const h = hingeAxis(du, du.clone().add(v(0, 0, 0.3)));
      expect(h.clone().cross(du).z).toBeGreaterThan(0);
      expect(elbowFlexion(du, du.clone().add(v(0, 0, 1)).normalize(), h)).toBeCloseTo(45, 4);
      expect(elbowFlexion(du, du.clone().add(v(0, 0, -1)).normalize(), h)).toBeCloseTo(-45, 4);
    }
  });
  it("falls back to folding toward +z when the bind arm is straight, and counts out-of-plane bends in full", () => {
    const du = v(1, -0.5, 0).normalize(), h = hingeAxis(du, du);
    expect(elbowFlexion(du, v(0, 0, 1), h)).toBeCloseTo(90, 4);
    expect(Math.abs(elbowFlexion(du, h, h))).toBeCloseTo(90, 4); // along the hinge: 90, not ~0
  });
});

describe("edgeMotion", () => {
  it("reports speed and jerkiness in millimetres per frame, split at sign edges", () => {
    // x = 0.0005·i² m: the speed at frame i is i mm and the jerkiness 1 mm everywhere
    const pts = Array.from({ length: 60 }, (_, i) => new THREE.Vector3(0.0005 * i * i, 0, 0));
    const m = edgeMotion(pts, pts.map((_, i) => (i * 1000) / 60), [500]);
    expect(m.edgeSpeed).toBeCloseTo(30, 6); // frames 28-32 lie within 40 ms of 500 ms
    expect(m.innerSpeed).toBeCloseTo(27, 6); // frames 1-27 and 33-58
    expect(m.edgeJerk).toBeCloseTo(1, 6);
    expect(m.innerJerk).toBeCloseTo(1, 6);
  });
});
