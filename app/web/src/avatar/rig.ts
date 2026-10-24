// Skeleton-agnostic retargeting math: turn any rig's bones so the avatar copies the signer's joint directions. Every
// rest direction is read from the rig itself (bind pose), so no per-model axis conventions are hard-coded.
import * as THREE from "three";

/** MediaPipe world/hand landmarks (x right in the image = the signer's left, y down, z away from the camera) ->
 *  three.js (y up, z toward the viewer). An avatar facing +z then moves its own left side with the signer's left. */
export const toThree = (p: number[]) => new THREE.Vector3(p[0], -p[1], -(p[2] ?? 0));

export interface Joint {
  bone: THREE.Object3D;
  bind: THREE.Quaternion; // bind-pose local rotation
  axis: THREE.Vector3; // bone-local unit direction toward the next joint
}

/** A bone with its bind rotation and its direction toward `child` (or `axis` when it has no child, e.g. fingertips). */
export function joint(bone: THREE.Object3D, child?: THREE.Object3D | null, axis?: THREE.Vector3): Joint {
  const dir = child && child.position.lengthSq() > 1e-12 ? child.position.clone() : (axis ?? new THREE.Vector3(1, 0, 0)).clone();
  return { bone, bind: bone.quaternion.clone(), axis: dir.normalize() };
}

/** The current world rotation of `o`'s parent (ancestors brought up to date first). */
export function parentWorld(o: THREE.Object3D): THREE.Quaternion {
  const q = new THREE.Quaternion();
  if (!o.parent) return q;
  o.parent.updateWorldMatrix(true, false);
  return o.parent.getWorldQuaternion(q);
}

/** Local rotation that points joint `j` along world direction `dir` with the least swing away from its bind pose
 *  under its current parent (no twist is invented). */
export function aimLocal(j: Joint, dir: THREE.Vector3): THREE.Quaternion {
  const P = parentWorld(j.bone);
  const carried = P.clone().multiply(j.bind);
  const cur = j.axis.clone().applyQuaternion(carried).normalize();
  const delta = new THREE.Quaternion().setFromUnitVectors(cur, dir.clone().normalize());
  return P.invert().multiply(delta).multiply(carried);
}

const basis = (f: THREE.Vector3, s: THREE.Vector3) => {
  const F = f.clone().normalize();
  const S = s.clone().sub(F.clone().multiplyScalar(s.dot(F))).normalize();
  return new THREE.Matrix4().makeBasis(F, S, new THREE.Vector3().crossVectors(F, S));
};

/** Rotation taking the frame (forward f0, side s0) onto (f, s); sides are made orthogonal to their forwards. */
export function frameRotation(f0: THREE.Vector3, s0: THREE.Vector3, f: THREE.Vector3, s: THREE.Vector3): THREE.Quaternion {
  return new THREE.Quaternion().setFromRotationMatrix(basis(f, s).multiply(basis(f0, s0).transpose()));
}

/** The twist part of rotation q about unit `axis` (swing-twist decomposition). */
export function twistAbout(q: THREE.Quaternion, axis: THREE.Vector3): THREE.Quaternion {
  const d = q.x * axis.x + q.y * axis.y + q.z * axis.z;
  const t = new THREE.Quaternion(axis.x * d, axis.y * d, axis.z * d, q.w);
  return t.lengthSq() < 1e-12 ? new THREE.Quaternion() : t.normalize();
}

/** Elbow position for a two-bone arm reaching `target` from shoulder `s`, kept in the plane of the current elbow `e`
 *  (its pole); out-of-reach targets are reached as far as the arm allows. */
export function twoBoneElbow(s: THREE.Vector3, e: THREE.Vector3, target: THREE.Vector3, l1: number, l2: number): THREE.Vector3 {
  const st = target.clone().sub(s);
  const d = Math.min(Math.max(st.length(), Math.abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4);
  const u = st.lengthSq() > 1e-12 ? st.normalize() : new THREE.Vector3(0, -1, 0);
  const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
  const pole = e.clone().sub(s);
  pole.sub(u.clone().multiplyScalar(pole.dot(u)));
  if (pole.lengthSq() < 1e-10) pole.set(0, -1, 0).sub(u.clone().multiplyScalar(-u.y)); // straight arm: bend downward
  return s.clone().add(u.clone().multiplyScalar(a)).add(pole.normalize().multiplyScalar(h));
}
