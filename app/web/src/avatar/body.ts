// Keeps the avatar's hands out of its own body. Signers touch the chest, chin and forehead, but the avatar copies arm
// *directions* from people with other proportions, so a hand resting on the signer's chest used to sink into the
// avatar's shirt. The torso is measured from the avatar's own mesh as stacked horizontal ellipses (2 cm slices, in its
// model space: the spine is not animated), the head as an ellipsoid; points inside are pushed out to the surface.
// An ellipse curves back faster than a chest, so off-centre a hand could be outside the ellipse but inside the shirt
// (the arm drawn "under" the torso): the front is also kept as a height map of the mesh itself (`front`).
import * as THREE from "three";

export interface Slice { cx: number; cz: number; rx: number; rz: number }
/** Front surface z of the mesh on an x-y grid (NaN: no body there). */
export interface Front { x0: number; y0: number; cell: number; nx: number; ny: number; z: Float32Array }

const FRONT_CELL = 0.015; // metres
const SHELL = 0.08; // a point up to this far behind the front surface is pushed forward out of it

export class Body {
  readonly y0: number; // height of the first slice
  readonly step: number;
  readonly slices: Slice[];
  readonly head: { c: THREE.Vector3; r: THREE.Vector3 } | null;
  readonly front: Front | null;

  constructor(y0: number, step: number, slices: Slice[], head: { c: THREE.Vector3; r: THREE.Vector3 } | null, front: Front | null = null) {
    this.y0 = y0;
    this.step = step;
    this.slices = slices;
    this.head = head;
    this.front = front;
  }

  /** Front height map of a point cloud: the most forward z per cell, single-cell holes between vertices closed. */
  static frontOf(pts: THREE.Vector3[], cell = FRONT_CELL): Front | null {
    if (!pts.length) return null;
    const box = new THREE.Box3().setFromPoints(pts);
    const x0 = box.min.x, y0 = box.min.y;
    const nx = Math.ceil((box.max.x - x0) / cell) + 1, ny = Math.ceil((box.max.y - y0) / cell) + 1;
    const z = new Float32Array(nx * ny).fill(NaN);
    for (const p of pts) {
      const k = Math.floor((p.y - y0) / cell) * nx + Math.floor((p.x - x0) / cell);
      if (!(z[k] >= p.z)) z[k] = p.z;
    }
    const filled = z.slice();
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (!Number.isNaN(z[j * nx + i])) continue;
      let n = 0, sum = 0;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const v = z[(j + dj) * nx + i + di];
        if (i + di >= 0 && i + di < nx && j + dj >= 0 && j + dj < ny && !Number.isNaN(v)) [n, sum] = [n + 1, sum + v];
      }
      if (n >= 3) filled[j * nx + i] = sum / n; // surrounded: a gap between vertices, not the body's edge
    }
    return { x0, y0, cell, nx, ny, z: filled };
  }

  /** The mesh's front surface z at (x, y), or null off the body. */
  frontAt(x: number, y: number): number | null {
    const f = this.front;
    if (!f) return null;
    const i = Math.floor((x - f.x0) / f.cell), j = Math.floor((y - f.y0) / f.cell);
    if (i < 0 || j < 0 || i >= f.nx || j >= f.ny) return null;
    const v = f.z[j * f.nx + i];
    return Number.isNaN(v) ? null : v;
  }

  /** Measure the torso and head from a skinned mesh in its current (bind) pose. `torso`/`head`: bone indices whose
   *  vertices (by dominant skin weight) belong to each part. */
  static fromMesh(mesh: THREE.SkinnedMesh, torso: Set<number>, head: Set<number>, step = 0.02): Body {
    mesh.updateWorldMatrix(true, true);
    const pos = mesh.geometry.getAttribute("position"), idx = mesh.geometry.getAttribute("skinIndex"), w = mesh.geometry.getAttribute("skinWeight");
    const tPts: THREE.Vector3[] = [], hPts: THREE.Vector3[] = [], v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      let best = 0, bone = -1;
      for (let k = 0; k < 4; k++) if (w.getComponent(i, k) > best) [best, bone] = [w.getComponent(i, k), idx.getComponent(i, k)];
      if (!torso.has(bone) && !head.has(bone)) continue;
      mesh.getVertexPosition(i, v);
      mesh.applyBoneTransform(i, v);
      const p = mesh.localToWorld(v.clone());
      (torso.has(bone) ? tPts : hPts).push(p);
    }
    return Body.fromPoints(tPts, hPts, step);
  }

  /** As fromMesh over several meshes (skin and the clothes over it), each vertex sorted by its dominant bone. */
  static fromMeshes(meshes: THREE.SkinnedMesh[], isTorso: (b: THREE.Object3D) => boolean, isHead: (b: THREE.Object3D) => boolean, step = 0.02): Body {
    const tPts: THREE.Vector3[] = [], hPts: THREE.Vector3[] = [], v = new THREE.Vector3();
    for (const mesh of meshes) {
      mesh.updateWorldMatrix(true, true);
      const pos = mesh.geometry.getAttribute("position"), idx = mesh.geometry.getAttribute("skinIndex"), w = mesh.geometry.getAttribute("skinWeight");
      for (let i = 0; i < pos.count; i++) {
        let best = 0, bone = -1;
        for (let k = 0; k < 4; k++) if (w.getComponent(i, k) > best) [best, bone] = [w.getComponent(i, k), idx.getComponent(i, k)];
        const b = mesh.skeleton.bones[bone];
        const part = b && isTorso(b) ? tPts : b && isHead(b) ? hPts : null;
        if (!part) continue;
        mesh.getVertexPosition(i, v);
        mesh.applyBoneTransform(i, v);
        part.push(mesh.localToWorld(v.clone()));
      }
    }
    return Body.fromPoints(tPts, hPts, step);
  }

  static fromPoints(torso: THREE.Vector3[], head: THREE.Vector3[], step = 0.02): Body {
    const ys = torso.map((p) => p.y);
    const y0 = Math.min(...ys), n = Math.max(1, Math.ceil((Math.max(...ys) - y0) / step));
    const ext = Array.from({ length: n }, () => ({ x0: Infinity, x1: -Infinity, z0: Infinity, z1: -Infinity }));
    for (const p of torso) {
      const e = ext[Math.min(n - 1, Math.floor((p.y - y0) / step))];
      e.x0 = Math.min(e.x0, p.x); e.x1 = Math.max(e.x1, p.x); e.z0 = Math.min(e.z0, p.z); e.z1 = Math.max(e.z1, p.z);
    }
    const slices: (Slice | null)[] = ext.map((e) => (e.x0 === Infinity ? null : { cx: (e.x0 + e.x1) / 2, cz: (e.z0 + e.z1) / 2, rx: (e.x1 - e.x0) / 2, rz: (e.z1 - e.z0) / 2 }));
    for (let i = 0; i < n; i++) slices[i] ??= slices[i - 1] ?? slices.find((s) => s) ?? null; // close gaps
    let hd = null;
    if (head.length) {
      const box = new THREE.Box3().setFromPoints(head);
      hd = { c: box.getCenter(new THREE.Vector3()), r: box.getSize(new THREE.Vector3()).multiplyScalar(0.5) };
    }
    return new Body(y0, step, slices as Slice[], hd, Body.frontOf([...torso, ...head]));
  }

  /** Height (z) of the body's front surface at (x, y), or null where neither torso nor head is: where a hand moved
   *  straight back would first touch. */
  frontZ(x: number, y: number): number | null {
    const m = this.frontAt(x, y);
    if (m !== null) return m;
    let z: number | null = null;
    const i = Math.floor((y - this.y0) / this.step);
    if (i >= 0 && i < this.slices.length) {
      const s = this.slices[i], u = (x - s.cx) / s.rx;
      if (Math.abs(u) < 1) z = s.cz + s.rz * Math.sqrt(1 - u * u);
    }
    if (this.head) {
      const { c, r } = this.head, u = (x - c.x) / r.x, v = (y - c.y) / r.y;
      if (u * u + v * v < 1) z = Math.max(z ?? -Infinity, c.z + r.z * Math.sqrt(1 - u * u - v * v));
    }
    return z;
  }

  /** How far point `p` must move to sit `margin` outside the body (zero vector when it already does). Torso pushes
   *  are horizontal, out from the slice centre; head pushes are radial. */
  push(p: THREE.Vector3, margin: number): THREE.Vector3 {
    const out = new THREE.Vector3();
    const i = Math.floor((p.y - this.y0) / this.step);
    if (i >= 0 && i < this.slices.length) {
      const s = this.slices[i], rx = s.rx + margin, rz = s.rz + margin;
      const dx = p.x - s.cx, dz = p.z - s.cz;
      const e = Math.hypot(dx / rx, dz / rz);
      if (e < 1) out.set(e < 1e-6 ? 0 : dx / e - dx, 0, e < 1e-6 ? s.cz + rz - p.z : dz / e - dz); // dead centre: forward
    }
    const f = this.frontAt(p.x, p.y);
    if (f !== null && p.z < f + margin && p.z > f - SHELL) { // in the front of the real chest or face: straight out
      const fwd = f + margin - p.z;
      if (out.lengthSq() === 0 || fwd < out.length()) out.set(0, 0, fwd);
    }
    if (this.head) {
      const { c, r } = this.head, d = p.clone().sub(c);
      const rr = r.clone().addScalar(margin);
      const e = Math.hypot(d.x / rr.x, d.y / rr.y, d.z / rr.z);
      if (e < 1) {
        const h = e < 1e-6 ? new THREE.Vector3(0, 0, rr.z) : d.clone().divideScalar(e);
        const hp = h.sub(d);
        if (hp.lengthSq() > out.lengthSq()) out.copy(hp);
      }
    }
    return out;
  }
}
