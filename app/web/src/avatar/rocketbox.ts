// Signing avatar. Default: the stylized "toon" character (scripts/build_toon_avatar.py, on MPFB's rig); ?model=Male_Adult_11 the
// realistic Microsoft Rocketbox one (MIT, public/avatar/rocketbox/LICENSE.txt); ?model=mpfb
// loads our own MPFB character (MakeHuman for Blender; CC0 assets, CC-BY hair; public/avatar/mpfb/ from scripts/build_avatar.py). Dark solid tops so skin-coloured hands read clearly, as interpreters dress. Arms, wrists and all
// 15 finger bones copy the signer's joint directions (rig.ts); hands and forearms are then kept out of the body
// (body.ts) by pushing the wrist to the surface and re-bending the arm. Face: the model's ARKit blendshapes.
import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { armFilters, armRig, DEFAULT_LIMITS, easeHand, settlePose, solveArmPose, stepArm, type ArmFilters, type ArmPose, type ArmRig, type ArmTarget, type HandEase } from "./armSolver";
import { Body } from "./body";
import { savedBuddy } from "../buddies";
import { elbowFlexion } from "./metrics";
import { DEFAULT_REST } from "./rest";
import { wristFrom2d } from "./placement";
import { OLD_MOTION, type ClipFrame } from "./player";
import { aimLocal, frameRotation, joint, parentWorld, toThree, twistAbout, type Joint } from "./rig";
import { QuatFilter } from "./rotationFilter";
import { ndcToCanvas } from "./trail";
import { createHeadMotion } from "./headMotion";
import { RotationSpring, type SpringParams } from "./motionSmoother";
import { backwardFingers, capsuleContact, carry, constrainBone, crossOverlap, coveredHand, jointFrame, palm, relaxedDir, selfOverlap, splay, type Capsule, type Palm } from "./handModel";

export interface FacePose { browUp: number; browDown: number; headYawDeg: number; headPitchDeg: number; expressions: Record<string, number> }

/** One arm's pose after the last frame (evaluation): world-space bone rotations, elbow flexion (0 straight, + bent the human
 *  way, - backward), forearm twist about its own axis relative to bind, and elbow / wrist positions. */
export interface ArmState {
  upper: THREE.Quaternion; fore: THREE.Quaternion; hand: THREE.Quaternion;
  elbowDeg: number; twistDeg: number; wrist: THREE.Vector3; elbow: THREE.Vector3;
}

export interface Avatar {
  /** `nowMs` is the frame's clock time (defaults to performance.now()); bench mode steps it at fixed intervals. */
  applyFrame(f: ClipFrame, nowMs?: number): void;
  applyFace(p: FacePose): void;
  /** Deepest a hand or forearm point sits inside the body after the last frame, metres (avatars that measure it). */
  penetration?(): number;
  /** Deepest elbow or lower upper-arm point inside the body, metres (evaluation; the shoulder end is always inside). */
  armPenetration?(): number;
  /** Both arms' pose after the last frame (avatars that measure it). */
  armState?(): Record<"L" | "R", ArmState>;
  /** Metres from the nearest hand probe to the avatar's mapped touched place, for the clip's own `touchAt` where
   *  `touchL`/`touchR` >= 0.5, or for `at` (a point in pose world coordinates) instead. Null when nothing is touched. */
  contactError?(f: ClipFrame, at?: { hand: "L" | "R"; place: number[] }): number | null;
  /** Head joint height in metres (evaluation). */
  headHeight?(): number;
  /** Face shapes currently applied, per mesh (evaluation). */
  activeMorphs?(): unknown;
  /** Right hand knuckle / middle-joint bend per finger in degrees (evaluation). */
  fingerBends?(): number[][];
  /** Both hands after the last frame, measured against the anatomical hand (handModel.ts; evaluation). */
  handReport?(): HandReport;
  /** Fingertips, wrists and elbows of tracked arms plus framing references (evaluation). */
  framePoints?(all?: boolean): { pts: number[][]; pelvisY: number; headTopY: number; shoulderHalf: number; armLength: number; upperArm: number };
  /** Both palm centers (halfway from the wrist to the middle knuckle) in canvas CSS pixels after the last frame, null
   *  outside the view: the points the hand trail draws through. */
  palmScreen?(): Record<"L" | "R", [number, number] | null>;
  /** Turn the final motion springs on or off (bench before/after; default on). */
  setSmoothing?(on: boolean): void;
  /** Turn the anatomical hand on or off (bench before/after; default on). */
  setHandLimits?(on: boolean): void;
}

export interface HandReport {
  outJoints: number; // finger joints (of 30) past their range by more than 1 degree
  maxOutDeg: number; // how far the worst one is past it
  selfOverlap: number; // deepest overlap between two fingers of one hand, metres
  handOverlap: number; // deepest overlap between the two arms' hands and forearms, metres
  fingerBody: number; // deepest finger joint or tip inside the body, metres
  held: number; // hands whose landmarks were rejected this frame (mirrored or garbage) and held from the last good frame
  joints?: number[]; // degrees past range per joint: L then R, thumb..little, 3 joints each (evaluation detail)
}

// Rocketbox characters in public/avatar/rocketbox/ (model -> texture prefix): ?model=Male_Adult_07 (blazer) or
// ?model=Male_Adult_11 (plain shirt). ?model=mpfb: the MPFB character. Anything else: the stylized toon character
// (scripts/build_toon_avatar.py) as the signing buddy picked on the Profile tab, built on MPFB's rig, so everything below
// that says MPFB applies to it too.
const ROCKETBOX: Record<string, string> = { Male_Adult_07: "m013", Male_Adult_11: "m001" };
const PICK = new URLSearchParams(location.search).get("model");
const CHARACTER = PICK === "mpfb" ? "mpfb" : PICK && PICK in ROCKETBOX ? PICK : "toon";
const IS_TOON = CHARACTER === "toon";
const IS_MPFB = CHARACTER === "mpfb" || IS_TOON; // MPFB's rig: bone and face shape names
const BASE = IS_TOON ? "/avatar/toon/" : IS_MPFB ? "/avatar/mpfb/" : "/avatar/rocketbox/";
const TEX = ROCKETBOX[CHARACTER];
const DEFAULT_FRAME = CHARACTER === "Male_Adult_07" ? 1.2 : 1; // broader shoulders (see FRAME below); MPFB's are modelled
// Rocketbox (3ds Max Biped) bone names -> MPFB's game-engine rig
const MPFB_BONES: Record<string, string> = { Head: "head", Spine2: "spine_03", Neck: "neck_01", Pelvis: "pelvis" };
const MPFB_PARTS: Record<string, string> = { UpperArm: "upperarm", Forearm: "lowerarm", Hand: "hand", Clavicle: "clavicle", Thigh: "thigh" };
const MPFB_FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
const boneName = (n: string): string => {
  if (!IS_MPFB) return `Bip01_${n}`;
  if (n in MPFB_BONES) return MPFB_BONES[n];
  const m = /^([LR])_(?:Finger(\d)(\d?)|(\w+))$/.exec(n);
  if (!m) return n;
  const side = m[1].toLowerCase();
  return m[2] !== undefined ? `${MPFB_FINGERS[+m[2]]}_0${m[3] ? +m[3] + 1 : 1}_${side}` : `${MPFB_PARTS[m[4]]}_${side}`;
};
const TORSO_BONES = IS_MPFB ? /^(pelvis|spine_0\d|neck_01)$/ : /^Bip01_(Pelvis|Spine\d?|Neck)$/;
// ARKit shape names: Rocketbox "blendShape1.AK_01_BrowDownLeft", MPFB (faceunits01) "browDownLeft"
const morphName = (ak: string) => (IS_MPFB ? ak.replace(/^AK_\d+_(.)/, (_, c: string) => c.toLowerCase()) : `blendShape1.${ak}`);
const MARGIN = 0.012; // metres kept between hand and body: a touching sign rests on the shirt
const COLLIDE = !new URLSearchParams(location.search).has("nocollide"); // ?nocollide: compare without the body push
const TOUCH = !new URLSearchParams(location.search).has("notouch"); // ?notouch: compare without the ASL-LEX contact depth
const IDLE_EASE = 150; // ms time constant of an arm whose hand left the video easing toward the rest pose
// a twist held at the clamp read back a hair past it (float noise, the filter's slerp): clamp just inside the limit
const LIMITS = { ...DEFAULT_LIMITS, twistMaxDeg: DEFAULT_LIMITS.twistMaxDeg - 0.5 };
// Fingers: one-euro per joint, lighter than the arm's (FILTER_HZ in armSolver.ts) so handshape changes stay crisp
const FINGER_HZ = 8, FINGER_BETA = 0.5;
// Anatomical hand (handModel.ts): fingers are retargeted relative to the palm and every joint kept in its human range;
// ?nohandlimits shows the raw copy of the signer's world directions (before)
let HAND_LIMITS = !new URLSearchParams(location.search).has("nohandlimits");
const FINGER_R = 0.0085, THUMB_R = 0.0095; // capsule radii (m) for the overlap measures
const SHAPE_HOLD_MS = 500; // longest a handshape is held for a single finger the tracker bent backward
// Last stage (motionSmoother.ts): critically damped springs on every arm and finger bone's final rotation, so no bone
// overshoots and springs back or snaps when a correction switches on. ?nosmooth shows the motion without them
let SMOOTH = !new URLSearchParams(location.search).has("nosmooth");
const SPRING: Record<"shoulder" | "arm" | "hand" | "finger", SpringParams> = {
  shoulder: { hz: 4, maxAccel: 150 }, arm: { hz: 6, maxAccel: 250 }, hand: { hz: 7, maxAccel: 400 }, finger: { hz: 8, maxAccel: 900 },
};
const ELBOW_DOWN = 0.5; // share of a relaxed, hanging elbow mixed into the signer's (MediaPipe's elbow depth flared them)
const SHRUG = 0.35, SHRUG_MAX = 0.45; // collarbone raise per radian the upper arm rises above -10 degrees, and its cap
const STILL = new URLSearchParams(location.search).has("still"); // ?still: torso never turns (as before)
const TORSO_SHARE = 0.7, TORSO_MAX = 0.14, TORSO_EASE = 0.12; // share of the signer's shoulder turn, cap (8 deg), step per 60 fps frame
const PROTRACT_MAX = (8 * Math.PI) / 180, PROTRACT_REACH = 0.4; // collarbone forward turn, and the wrist-in-front-of-shoulder distance (m) that gives all of it
const HEAD_KEEP = 0.5; // share of the torso's turn the head undoes, so the avatar keeps facing the viewer
const PLACE_SLACK = { head: 0.03, body: 0.06 }; // metres a contact may land from the coded place: signers vary
const MEET_SIGMA = 0.02; // metres across and up within which the two hands' points count toward their contact depth
const DISAGREE = { min: 0.15, max: 0.35 }; // world vs image wrist offset (shoulder widths): none / fully from the image
const FACE_NEAR = 0.6, FACE_FAR = 1.3; // wrist-to-nose distance (shoulder widths): fully / not at all face-relative
// MediaPipe hand chains -> Rocketbox (3ds Max Biped) fingers: Finger0 = thumb ... Finger4 = little, 3 bones each
const CHAINS: [number[], number][] = [[[1, 2, 3, 4], 0], [[5, 6, 7, 8], 1], [[9, 10, 11, 12], 2], [[13, 14, 15, 16], 3], [[17, 18, 19, 20], 4]];
const BROW_UP = ["AK_03_BrowInnerUp", "AK_04_BrowOuterUpLeft", "AK_05_BrowOuterUpRight"];
const BROW_DOWN = ["AK_01_BrowDownLeft", "AK_02_BrowDownRight"];
const BLINK = ["AK_09_EyeBlinkLeft", "AK_10_EyeBlinkRight"];
// resting face: lower lids a little raised and brows a little lowered and level (hooded, "hunter" eyes); grammar's raised
// brows lift them as usual, and the brow rest fades while they do
const SQUINT = ["AK_19_EyeSquintLeft", "AK_20_EyeSquintRight"];
// MPFB's face is modelled hooded; the toon's rests open and friendly
const REST_SQUINT = IS_TOON ? 0 : IS_MPFB ? 0.1 : 0.35, REST_BROW_DOWN = IS_TOON ? 0 : IS_MPFB ? 0.1 : 0.25;
// face.ts tone keys (VRoid names) -> ARKit shapes; tone stays on mouth and eyes, never the brows grammar uses
const TONE: Record<string, [string, number][]> = {
  MTH_Joy: [["AK_44_MouthSmileLeft", 1], ["AK_45_MouthSmileRight", 1]],
  EYE_Joy: [["AK_07_CheekSquintLeft", 0.8], ["AK_08_CheekSquintRight", 0.8]],
  MTH_Sorrow: [["AK_30_MouthFrownLeft", 1], ["AK_31_MouthFrownRight", 1]],
  // the toon's mouth is modelled smiling: 0.3 of its frown draws it straight (0.5 already turns down); others rest straight
  MTH_Neutral: IS_TOON ? [["AK_30_MouthFrownLeft", 0.3], ["AK_31_MouthFrownRight", 0.3]] : [],
  EYE_Sorrow: [["AK_11_EyeLookDownLeft", 0.5], ["AK_12_EyeLookDownRight", 0.5]],
  MTH_Surprised: [["AK_25_JawOpen", 0.5]],
  EYE_Surprised: [["AK_21_EyeWideLeft", 1], ["AK_22_EyeWideRight", 1]],
};

interface Arm {
  upper: Joint; fore: Joint; hand: Joint;
  f0: THREE.Vector3; s0: THREE.Vector3; // hand-local palm frame: wrist -> middle knuckle, little -> index knuckle
  fingers: Joint[][];
  tips: [THREE.Object3D, THREE.Vector3][]; // distal bone + local fingertip offset (collision probes)
  rig: ArmRig; filters: ArmFilters; fingerFilters: QuatFilter[][];
  last: ArmPose | null; // the filtered pose on the bones, carried into the next solve
  heldGap: number | null; // last contact depth measured while touching (contactGap)
  held: boolean; // this frame's hand landmarks were rejected (mirrored or garbage, or covered by the other hand) and the last good hand kept
  goodPts: number[][] | null; // the last hand landmarks that passed every check (a held handshape is placed from these)
  shapeHeldMs: number; // how long the handshape alone has been held for one finger bent back (released after SHAPE_HOLD_MS)
  solvedAt: number; // clock time of the frame this arm was last posed for (the other hand avoids it only once it is current)
  seen: THREE.Quaternion | null; // the hand's world rotation from its last landmarks (held while the hand is untracked)
  ease: HandEase | null; // the hand's turn while it appears or leaves (easeHand)
  pose: [number, number, number]; // shoulder, elbow, wrist pose landmarks
}

interface BodyFrame { o: THREE.Vector3; x: THREE.Vector3; y: THREE.Vector3; z: THREE.Vector3; w: number }

/** Frame on the upper body from pose landmarks (MediaPipe indices, three.js space): origin between the shoulders,
 *  x toward the left shoulder, y up the spine (hips -> shoulders), z forward; w = shoulder width, the scale. */
function bodyFrame(p: THREE.Vector3[]): BodyFrame {
  const o = p[11].clone().add(p[12]).multiplyScalar(0.5);
  const x = p[11].clone().sub(p[12]);
  const w = Math.max(x.length(), 1e-6);
  x.normalize();
  const y = o.clone().sub(p[23].clone().add(p[24]).multiplyScalar(0.5));
  y.sub(x.clone().multiplyScalar(y.dot(x))).normalize();
  return { o, x, y, z: new THREE.Vector3().crossVectors(x, y), w };
}

/** The point at the same body-relative place (in shoulder widths) on another body. */
function mapPoint(p: THREE.Vector3, from: BodyFrame, to: BodyFrame): THREE.Vector3 {
  const r = p.clone().sub(from.o).divideScalar(from.w);
  return to.o.clone()
    .add(to.x.clone().multiplyScalar(r.dot(from.x) * to.w))
    .add(to.y.clone().multiplyScalar(r.dot(from.y) * to.w))
    .add(to.z.clone().multiplyScalar(r.dot(from.z) * to.w));
}

/** The tip of the nose: the most forward point of the face near its midline (avatars without a nose bone). */
function noseTip(mesh: THREE.SkinnedMesh, head: THREE.Object3D): THREE.Vector3 {
  mesh.updateWorldMatrix(true, true);
  const hx = head.getWorldPosition(new THREE.Vector3()).x, v = new THREE.Vector3();
  const pos = mesh.geometry.getAttribute("position"), idx = mesh.geometry.getAttribute("skinIndex"), w = mesh.geometry.getAttribute("skinWeight");
  let best: THREE.Vector3 | null = null;
  for (let i = 0; i < pos.count; i++) {
    let top = 0, b = -1;
    for (let k = 0; k < 4; k++) if (w.getComponent(i, k) > top) [top, b] = [w.getComponent(i, k), idx.getComponent(i, k)];
    if (mesh.skeleton.bones[b] !== head) continue;
    mesh.getVertexPosition(i, v);
    mesh.applyBoneTransform(i, v);
    const p = mesh.localToWorld(v.clone());
    if (Math.abs(p.x - hx) < 0.015 && (!best || p.z > best.z)) best = p;
  }
  return best ?? head.getWorldPosition(new THREE.Vector3());
}

function isDescendant(o: THREE.Object3D, ancestor: THREE.Object3D): boolean {
  for (let p = o.parent; p; p = p.parent) if (p === ancestor) return true;
  return false;
}

/** `maxPixelRatio`, `maxFps`: a lighter draw where the camera's trackers share the GPU with the avatar (mirror.ts). */
export interface AvatarOptions { maxPixelRatio?: number; maxFps?: number }

export async function loadRocketbox(canvas: HTMLCanvasElement, opts: AvatarOptions = {}): Promise<Avatar> {
  // bench mode reads frames back with toDataURL, which needs the drawing buffer kept after presenting
  const bench = new URLSearchParams(location.search).get("mode") === "bench";
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: bench });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, opts.maxPixelRatio ?? 2)); // an iPhone's 3x drew 2.25x the pixels for no visible gain
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = IS_TOON ? THREE.NoToneMapping : THREE.ACESFilmicToneMapping; // filmic tone mapping greys flat colours
  const scene = new THREE.Scene();
  // interpreter framing: waist to just above the head, hands large in frame (fitCamera sets it once the avatar is measured)
  const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 20);
  camera.position.set(0, 1.42, 1.95);
  if (new URLSearchParams(location.search).get("view") === "side") camera.position.set(1.95, 1.42, 0.2); // depth checks
  camera.lookAt(0, 1.4, 0);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); // soft front key so fingers never fall into shadow
  key.position.set(0.6, 2.2, 2.4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.8);
  fill.position.set(-1.5, 1.4, 1.2);
  scene.add(fill);

  const texLoader = new THREE.TextureLoader();
  const texture = (file: string, color: boolean) => {
    const t = texLoader.load(BASE + file);
    t.flipY = !IS_MPFB; // glTF UVs have v down (the FBX's up)
    if (color) t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return t;
  };
  let obj: THREE.Object3D;
  let found: THREE.SkinnedMesh | null = null;
  const morphMeshes: THREE.Mesh[] = []; // meshes carrying the ARKit face shapes (MPFB: body, brows and lashes)
  const collideMeshes: THREE.SkinnedMesh[] = []; // the body's surface for the hands: skin, and clothes worn over it
  if (IS_TOON) {
    obj = (await new GLTFLoader().loadAsync(`${BASE}${savedBuddy().id}.glb`)).scene;
    // flat vertex colours, cel-shaded in three steps; toon_body is the skin and shirt the hands collide with, toon_face
    // the painted-on features carrying the face shapes, toon_hair (some buddies) hair the hands pass
    const ramp = new THREE.DataTexture(new Uint8Array([120, 200, 255]), 3, 1, THREE.RedFormat);
    ramp.minFilter = ramp.magFilter = THREE.NearestFilter;
    ramp.needsUpdate = true;
    const toon = new THREE.MeshToonMaterial({ vertexColors: true, gradientMap: ramp, side: THREE.DoubleSide }); // the tee's openings show its inside
    obj.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.frustumCulled = false; // raised arms leave the bind-pose bounds
      m.material = toon;
      if (m.morphTargetDictionary) morphMeshes.push(m);
      if ((m as THREE.SkinnedMesh).isSkinnedMesh && m.name === "toon_body") {
        found = m as THREE.SkinnedMesh;
        collideMeshes.push(found);
      }
    });
  } else if (IS_MPFB) {
    obj = (await new GLTFLoader().loadAsync(BASE + "avatar.glb")).scene;
    // hair, brow and lash cards: colour + alpha in one texture, cut out rather than blended (no depth sorting)
    const cards = (file: string, roughness = 0.8, alphaTest = 0.35) =>
      new THREE.MeshStandardMaterial({ map: texture(file, true), alphaTest, side: THREE.DoubleSide, roughness, metalness: 0 });
    const skin = new THREE.MeshStandardMaterial({ map: texture("skin.jpg", true), roughness: 0.6, metalness: 0 });
    const top = new THREE.MeshStandardMaterial({ color: 0x1d1e22, normalMap: texture("top_normal.jpg", false), roughness: 0.85, metalness: 0 });
    const byName: [RegExp, THREE.Material][] = [
      // fine brow and lash hairs are mostly half-transparent: a lower cut-off keeps them
      [/eyebrow/i, cards("brows.png", 0.8, 0.25)], [/eyelash/i, cards("lashes.png", 0.8, 0.2)], [/hair|curly|long|short/i, cards("hair.png", 0.6)],
      // eyeball and the clear cornea over it share a texture whose alpha is the cornea: cut out, since a blended
      // cornea drawn first hid the eyeball behind it (depth)
      [/highpoly|high-poly/i, new THREE.MeshStandardMaterial({ map: texture("eyes.png", true), roughness: 0.1, metalness: 0, alphaTest: 0.5 })],
      [/teeth/i, new THREE.MeshStandardMaterial({ map: texture("teeth.jpg", true), roughness: 0.4, metalness: 0 })],
      [/suit/i, top],
    ];
    obj.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.frustumCulled = false; // raised arms leave the bind-pose bounds
      m.material = byName.find(([re]) => re.test(m.name))?.[1] ?? skin;
      if (m.morphTargetDictionary) morphMeshes.push(m);
      if (!(m as THREE.SkinnedMesh).isSkinnedMesh) return;
      if (m.material === skin) found = m as THREE.SkinnedMesh;
      if (m.material === skin || m.material === top) collideMeshes.push(m as THREE.SkinnedMesh);
    });
  } else {
    const manager = new THREE.LoadingManager();
    manager.setURLModifier((u) => (u.startsWith("blob:") || u.startsWith("data:") ? u : BASE + u.split(/[\\/]/).pop()!.replace(/\.tga$/i, ".jpg")));
    obj = await new FBXLoader(manager).loadAsync(`${BASE}${CHARACTER}.fbx`);
    obj.scale.setScalar(0.01); // centimetres -> metres
    const tex = (name: string, color: boolean) => texture(`${TEX}_${name}.jpg`, color);
    const mats = {
      body: new THREE.MeshStandardMaterial({ map: tex("body_color", true), normalMap: tex("body_normal", false), roughness: 0.85, metalness: 0 }),
      head: new THREE.MeshStandardMaterial({ map: tex("head_color", true), normalMap: tex("head_normal", false), roughness: 0.7, metalness: 0 }),
      // hair and eyelash cards (characters with longer hair): colour + alpha in one texture, cut out rather than
      // blended so they need no depth sorting
      opacity: new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, alphaTest: 0.35, side: THREE.DoubleSide }),
    };
    const hasOpacity = (m: THREE.Material | THREE.Material[]) => [m].flat().some((x) => /opacity/i.test(x.name));
    obj.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.frustumCulled = false; // raised arms leave the bind-pose bounds
      if (hasOpacity(m.material) && !mats.opacity.map) mats.opacity.map = texture(`${TEX}_opacity_color.png`, true);
      m.material = [m.material].flat().map((x) => (/opacity/i.test(x.name) ? mats.opacity : /head/i.test(x.name) ? mats.head : mats.body));
      if ((m as THREE.SkinnedMesh).isSkinnedMesh) found = m as THREE.SkinnedMesh;
    });
    if (found) {
      morphMeshes.push(found);
      collideMeshes.push(found);
    }
  }
  if (!found) throw new Error("avatar has no skinned mesh");
  const skinned: THREE.SkinnedMesh = found;
  scene.add(obj);

  const bone = (name: string) => {
    const b = obj.getObjectByName(boneName(name));
    if (!b) throw new Error(`avatar bone missing: ${boneName(name)}`);
    return b;
  };
  const wpos = (o: THREE.Object3D) => {
    o.updateWorldMatrix(true, false);
    return o.getWorldPosition(new THREE.Vector3());
  };
  // face the camera (+z): the model's left side must be +x so the signer's left drives the avatar's left
  if (wpos(bone("L_UpperArm")).x < wpos(bone("R_UpperArm")).x) obj.rotation.y += Math.PI;
  obj.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(obj);
  obj.position.y -= box.min.y; // feet on the floor
  obj.position.x -= (box.min.x + box.max.x) / 2;
  obj.updateMatrixWorld(true);

  // ?frame=1.1 broadens the shoulders: the upper arms sit further out along the collarbones (the arms keep their size)
  const FRAME = Number(new URLSearchParams(location.search).get("frame") ?? DEFAULT_FRAME);
  const naturalWidth = wpos(bone("L_UpperArm")).distanceTo(wpos(bone("R_UpperArm")));
  if (FRAME !== 1) {
    for (const s of ["L", "R"]) bone(`${s}_UpperArm`).position.multiplyScalar(FRAME);
    obj.updateMatrixWorld(true);
  }

  const makeArm = (s: "L" | "R"): Arm => {
    const up = bone(`${s}_UpperArm`), fo = bone(`${s}_Forearm`), ha = bone(`${s}_Hand`);
    const f = (n: number, k = "") => bone(`${s}_Finger${n}${k}`);
    const fingers = CHAINS.map(([, n]) => {
      const [b0, b1, b2] = [f(n), f(n, "1"), f(n, "2")];
      const tipAxis = b2.position.clone().applyQuaternion(b2.quaternion.clone().invert()); // continue the middle bone
      return [joint(b0, b1), joint(b1, b2), joint(b2, null, tipAxis)];
    });
    const [upper, fore, hand] = [joint(up, fo), joint(fo, ha), joint(ha, f(2))];
    return {
      upper, fore, hand,
      f0: f(2).position.clone().normalize(), s0: f(1).position.clone().sub(f(4).position),
      fingers,
      tips: fingers.map((ch) => [ch[2].bone, ch[2].axis.clone().multiplyScalar(ch[2].bone.position.length() * 0.8)]),
      // the palm frame gives the wrist its own ranges (bend toward the palm / back, tilt toward the thumb / little finger)
      rig: armRig(upper, fore, hand, { f: f(2).position.clone(), s: f(1).position.clone().sub(f(4).position), right: s === "R" }), filters: armFilters(), fingerFilters: fingers.map((ch) => ch.map(() => new QuatFilter(FINGER_HZ, FINGER_BETA))),
      last: null, seen: null, ease: null, heldGap: null, held: false, goodPts: null, shapeHeldMs: 0, solvedAt: -Infinity,
      pose: s === "L" ? [11, 13, 15] : [12, 14, 16],
    };
  };
  const arms = { L: makeArm("L"), R: makeArm("R") };
  const clavicle = { L: bone("L_Clavicle"), R: bone("R_Clavicle") };
  const clavicleFilter = { L: new QuatFilter(1.5, 0.5, 1), R: new QuatFilter(1.5, 0.5, 1) };
  const clavicleBind = { L: clavicle.L.quaternion.clone(), R: clavicle.R.quaternion.clone() };
  // the avatar's body frame (shoulder joints, hip joints), in the bind pose: the spine is not animated
  const avatarFrame = bodyFrame(Object.assign([] as THREE.Vector3[], {
    11: wpos(bone("L_UpperArm")), 12: wpos(bone("R_UpperArm")), 23: wpos(bone("L_Thigh")), 24: wpos(bone("R_Thigh")),
  }));
  avatarFrame.w = naturalWidth; // signs keep their size on a broadened frame (and face contacts their offsets)
  const headBone = bone("Head"), headBind = headBone.quaternion.clone();
  // the framing at rest (fitCamera): the waist to just above the head, tight enough that hands hanging at rest are out
  // of frame; measured on the avatar, so any height or head size frames the same
  obj.updateMatrixWorld(true);
  const SPACE = { lo: wpos(bone("Pelvis")).y + 0.16, hi: new THREE.Box3().setFromObject(obj).max.y + 0.03, half: naturalWidth / 2 + 0.12 };
  // The fixed framing: the whole signing space, so the camera never has to move. Measured over all 167 clips on the toon
  // (raised hands only): fingertips went down to 1.05-1.08 m (pelvis 1.20; NICE 1.14, DOCTOR 1.05), up to 2.14 m (p98; head
  // top 2.18) and out to 0.54 m from the middle (p98; shoulder half-width 0.26; only BIG reached 0.86)
  const VIEW = { lo: wpos(bone("Pelvis")).y - 0.14, hi: SPACE.hi, half: naturalWidth / 2 + 0.29 };
  const view = new URLSearchParams(location.search).get("view");
  if (view === "face" || view === "eyes") { // close-ups for checking the face
    const near = view === "eyes";
    camera.position.set(0, wpos(headBone).y + (near ? 0.04 : 0.08), near ? 0.35 : 0.75);
    camera.lookAt(0, wpos(headBone).y + (near ? 0.04 : 0.06), 0);
  }
  const avatarNose = IS_MPFB ? noseTip(skinned, headBone) : wpos(bone("MNose"));

  // collision volume, measured in the bind pose: torso = pelvis/spine/neck vertices, head = head + face bones
  const isHead = (b: THREE.Object3D) => b === headBone || isDescendant(b, headBone);
  // The collision volume is measured in the bind pose; when the chest turns (torsoYaw) points are taken back into that
  // pose before testing it and the push is turned forward again.
  const bindBody = Body.fromMeshes(collideMeshes, (b) => TORSO_BONES.test(b.name), isHead);
  // the forearm's radius on this avatar: the 75th percentile of its vertices' distance from the elbow-wrist line, bind
  // pose (the toon's arms are much thicker than Rocketbox's)
  const FOREARM_R = (() => {
    const fore = bone("L_Forearm"), a = wpos(fore), b = wpos(bone("L_Hand")), ab = b.clone().sub(a), v = new THREE.Vector3(), d: number[] = [];
    for (const mesh of collideMeshes) {
      const pos = mesh.geometry.getAttribute("position"), idx = mesh.geometry.getAttribute("skinIndex"), w = mesh.geometry.getAttribute("skinWeight");
      for (let i = 0; i < pos.count; i++) {
        let best = 0, bi = -1;
        for (let k = 0; k < 4; k++) if (w.getComponent(i, k) > best) [best, bi] = [w.getComponent(i, k), idx.getComponent(i, k)];
        if (mesh.skeleton.bones[bi] !== fore) continue;
        mesh.getVertexPosition(i, v);
        mesh.applyBoneTransform(i, v);
        const p = mesh.localToWorld(v.clone()), t = Math.min(1, Math.max(0, p.clone().sub(a).dot(ab) / ab.lengthSq()));
        d.push(p.distanceTo(a.clone().addScaledVector(ab, t)));
      }
    }
    d.sort((x, y) => x - y);
    return d.length ? d[Math.floor(d.length * 0.75)] : 0.035;
  })();
  const pivot = wpos(bone("Spine2")), UP = new THREE.Vector3(0, 1, 0);
  let torsoYaw = 0;
  const yaw = (sign = 1) => new THREE.Quaternion().setFromAxisAngle(UP, sign * torsoYaw);
  const toBind = (p: THREE.Vector3) => p.clone().sub(pivot).applyQuaternion(yaw(-1)).add(pivot);
  const body = {
    push: (p: THREE.Vector3, margin: number) => bindBody.push(toBind(p), margin).applyQuaternion(yaw()),
    frontZ(x: number, y: number): number | null {
      const z0 = bindBody.frontZ(x, y);
      if (z0 === null) return null;
      const q = toBind(new THREE.Vector3(x, y, z0)), z1 = bindBody.frontZ(q.x, q.y);
      return z1 === null ? null : new THREE.Vector3(q.x, q.y, z1).sub(pivot).applyQuaternion(yaw()).add(pivot).z;
    },
  };

  const armState = (k: "L" | "R"): ArmState => {
    const a = arms[k], wq = (o: THREE.Object3D) => o.getWorldQuaternion(new THREE.Quaternion());
    const s = wpos(a.upper.bone), elbow = wpos(a.fore.bone), wrist = wpos(a.hand.bone);
    const du = elbow.clone().sub(s).normalize(), df = wrist.clone().sub(elbow).normalize();
    const h = a.rig.hinge.clone().applyQuaternion(wq(a.upper.bone)); // elbow hinge, from the bind pose (armRig)
    const elbowDeg = elbowFlexion(du, df, h);
    const t = twistAbout(a.fore.bind.clone().invert().multiply(a.fore.bone.quaternion), a.fore.axis);
    let twist = 2 * Math.atan2(t.x * a.fore.axis.x + t.y * a.fore.axis.y + t.z * a.fore.axis.z, t.w);
    twist = Math.atan2(Math.sin(twist), Math.cos(twist)); // into (-180, 180]
    return { upper: wq(a.upper.bone), fore: wq(a.fore.bone), hand: wq(a.hand.bone), elbowDeg, twistDeg: (twist * 180) / Math.PI, wrist, elbow };
  };

  /** The hand's world rotation from its landmarks (palm frame: wrist -> middle knuckle, little -> index knuckle). */
  const handRotation = (arm: Arm, pts: number[][]) =>
    frameRotation(arm.f0, arm.s0, toThree(pts[9]).sub(toThree(pts[0])), toThree(pts[5]).sub(toThree(pts[17])));

  /** The avatar hand's palm frame (handModel.palm) from its joint positions, as the landmarks' is: wrist -> middle
   *  knuckle, little -> index knuckle (positions, not the bone's rotation: a mirrored side's rotation would flip it). */
  const avatarPalm = (arm: Arm): Palm => {
    const w = wpos(arm.hand.bone), k = (c: number) => wpos(arm.fingers[c][0].bone);
    return palm(k(2).sub(w), k(1).sub(k(4)), arm === arms.R);
  };
  const handPalm = (P: THREE.Vector3[], right: boolean) => palm(P[9].clone().sub(P[0]), P[5].clone().sub(P[17]), right);
  /** Rotation taking the signer's palm frame onto the avatar's: finger directions are copied relative to the palm, so a
   *  wrist the arm solver had to limit does not leave the fingers pointing where the signer's did in the world. */
  const palmToPalm = (from: Palm, to: Palm) => frameRotation(from.F, from.S, to.F, to.S);

  /** Fingers copy the signer's joint directions; `t` filters them (one QuatFilter per joint), else set raw (trial).
   *  With the anatomical hand, each direction is taken relative to the palm and clamped into its joint's range, joint by
   *  joint from the knuckle out (a joint's range depends on where its parent bone actually points). */
  const placeFingers = (arm: Arm, pts: number[][], t?: number) => {
    const L = pts.map(toThree);
    const right = arm === arms.R;
    const Pa = HAND_LIMITS ? avatarPalm(arm) : null;
    const toAvatarPalm = Pa ? palmToPalm(handPalm(L, right), Pa) : null;
    const handPos = wpos(arm.hand.bone);
    const boneDir = (j: Joint) => j.axis.clone().applyQuaternion(j.bone.getWorldQuaternion(new THREE.Quaternion())).normalize();
    arm.fingers.forEach((chain, c) => {
      const idx = CHAINS[c][0];
      let r = c === 0 ? Pa?.F ?? null : wpos(chain[0].bone).sub(handPos).normalize(); // the metacarpal: wrist -> knuckle
      let hinge: THREE.Vector3 | null = null;
      let bend: number | undefined; // the previous joint's bend as posed: the end joint follows the middle one
      chain.forEach((j, k) => {
        let want = L[idx[k + 1]].clone().sub(L[idx[k]]);
        if (Pa && toAvatarPalm && r && want.lengthSq() > 1e-12) want = constrainBone(Pa, c, k, r, hinge, splay(Pa, c, k, r, want.applyQuaternion(toAvatarPalm)), bend).dir;
        const q = aimLocal(j, want);
        if (Number.isFinite(q.x + q.y + q.z + q.w)) j.bone.quaternion.copy(t === undefined ? q : arm.fingerFilters[c][k].filter(q, t));
        const now = boneDir(j);
        if (Pa && r) { // the filter blends between in-range poses and can still leave the range: clamp where the bone ended up
          const out = constrainBone(Pa, c, k, r, hinge, now, bend);
          if (out.clamped) {
            const q2 = aimLocal(j, out.dir);
            if (Number.isFinite(q2.x + q2.y + q2.z + q2.w)) j.bone.quaternion.copy(q2);
          }
          hinge = out.hinge;
          bend = out.bendOut;
        }
        r = boneDir(j); // where this bone points now: the next joint's parent
      });
    });
    if (Pa) separateFingers(arm, Pa);
  };

  /** A hand with no landmarks (out of the video, or at rest) eases into a relaxed hand, fingers lightly curled and the
   *  thumb resting by the index finger, instead of keeping the bind pose's flat hand and outstretched thumb. */
  const relaxFingers = (arm: Arm, t: number) => {
    const Pa = avatarPalm(arm), handPos = wpos(arm.hand.bone);
    arm.fingers.forEach((chain, c) => {
      let r = c === 0 ? Pa.F : wpos(chain[0].bone).sub(handPos).normalize();
      let hinge: THREE.Vector3 | null = null;
      chain.forEach((j, k) => {
        const out = relaxedDir(Pa, c, k, r, hinge);
        const q = aimLocal(j, out.dir);
        if (Number.isFinite(q.x + q.y + q.z + q.w)) j.bone.quaternion.copy(arm.fingerFilters[c][k].filter(q, t));
        hinge = out.hinge;
        r = j.axis.clone().applyQuaternion(j.bone.getWorldQuaternion(new THREE.Quaternion())).normalize();
      });
    });
  };

  /** Fingers of one hand passing through each other (the thumb inside a fist's curled fingers): turn one finger at the
   *  joint of the overlapping bone until the capsules only touch, through the joint limits; bones past it follow. The
   *  thumb moves when it is involved, else the finger nearer the little finger. Up to 3 passes, worst overlap first. */
  const separateFingers = (arm: Arm, Pa: Palm) => {
    for (let pass = 0; pass < 3; pass++) {
      const caps = capsules(arm);
      let worst: { mover: Capsule; other: Capsule; depth: number } | null = null;
      for (let i = 0; i < caps.length; i++) for (let k = i + 1; k < caps.length; k++) {
        const [x, y] = [caps[i], caps[k]];
        if (x.finger === y.finger) continue;
        const mover = x.finger === 0 ? x : y.finger === 0 ? y : x.finger > y.finger ? x : y, other = mover === x ? y : x;
        const depth = capsuleContact(mover, other).depth;
        if (depth > 5e-4 && (!worst || depth > worst.depth)) worst = { mover, other, depth };
      }
      if (!worst) return;
      // the preferred finger may be held by its own range (a thumb against its cone): if turning it does not clear at
      // least a fifth of the overlap, undo it and turn the other finger instead
      const tryMove = (mover: Capsule, other: Capsule) => {
        const c = mover.finger, k = mover.bone!, j = arm.fingers[c][k], before = j.bone.quaternion.clone();
        const contact = capsuleContact(mover, other, 0.003, mover.finger === 0 ? Pa.S : Pa.S.clone().negate());
        const pivot = wpos(j.bone), lever = mover.a.clone().lerp(mover.b, contact.s).sub(pivot);
        if (lever.length() < 0.004) return false; // the overlap is at the joint itself: turning the bone cannot clear it
        const dir = j.axis.clone().applyQuaternion(j.bone.getWorldQuaternion(new THREE.Quaternion())).normalize();
        const turn = new THREE.Quaternion().setFromUnitVectors(lever.clone().normalize(), lever.clone().addScaledVector(contact.dir, contact.depth).normalize());
        // the joint's frame from where its parent bones are now (an earlier pass may have turned one of them)
        const f = chainFrame(arm, Pa, c, k), want = dir.applyQuaternion(turn);
        const q = aimLocal(j, constrainBone(Pa, c, k, f.r, f.hinge, want, f.bend).dir);
        if (!Number.isFinite(q.x + q.y + q.z + q.w)) return false;
        j.bone.quaternion.copy(q);
        const now = capsules(arm), m2 = now.find((x) => x.finger === c && x.bone === k)!, o2 = now.find((x) => x.finger === other.finger && x.bone === other.bone)!;
        if (capsuleContact(m2, o2).depth < 0.8 * contact.depth) return true;
        j.bone.quaternion.copy(before);
        return false;
      };
      if (!tryMove(worst.mover, worst.other) && !tryMove(worst.other, worst.mover)) return;
    }
  };

  /** Where chain c's bone k hangs from as the bones are posed: its parent's direction, the hinge carried down to it and
   *  its parent joint's bend (constrainBone's inputs). */
  const chainFrame = (arm: Arm, Pa: Palm, c: number, k: number) => {
    let r = c === 0 ? Pa.F : wpos(arm.fingers[c][0].bone).sub(wpos(arm.hand.bone)).normalize();
    let hinge: THREE.Vector3 | null = null, bend: number | undefined;
    for (let i = 0; i < k; i++) {
      const j = arm.fingers[c][i], dir = j.axis.clone().applyQuaternion(j.bone.getWorldQuaternion(new THREE.Quaternion())).normalize();
      const out = constrainBone(Pa, c, i, r, hinge, dir, bend);
      hinge = c === 0 ? null : carry(jointFrame(Pa, r, hinge, false).h, r, dir); // carried by where the bone is, not its clamp
      bend = out.bendOut;
      r = dir;
    }
    return { r, hinge, bend };
  };

  /** Finger bone capsules (thumb metacarpal and the knuckle end of each proximal bone left out: the palm and
   *  neighbouring knuckles touch by design). */
  /** The forearm as a capsule (elbow to wrist; finger -1), with the radius measured on the avatar's own mesh. */
  const forearmCapsule = (arm: Arm): Capsule => ({ a: wpos(arm.fore.bone), b: wpos(arm.hand.bone), r: FOREARM_R, finger: -1 });
  /** Finger capsules and the forearm: what one arm must keep out of the other (NICE's forearms passed through each other). */
  const limbCapsules = (arm: Arm): Capsule[] => [...capsules(arm), forearmCapsule(arm)];

  const capsules = (arm: Arm): Capsule[] => {
    const out: Capsule[] = [];
    arm.fingers.forEach((ch, c) => {
      const pts = [...ch.map((j) => wpos(j.bone)), arm.tips[c][0].localToWorld(arm.tips[c][1].clone())];
      for (let k = c === 0 ? 1 : 0; k < 3; k++) {
        const a = k === 0 ? pts[0].clone().lerp(pts[1], 0.4) : pts[k];
        out.push({ a, b: pts[k + 1], r: c === 0 ? THUMB_R : FINGER_R, finger: c, bone: k });
      }
    });
    return out;
  };

  /** Every finger joint against the anatomical hand, from the posed bones (evaluation). */
  const handReport = (): HandReport => {
    obj.updateMatrixWorld(true);
    let outJoints = 0, maxOutDeg = 0, selfMax = 0, fingerBody = 0, held = 0;
    const joints: number[] = [];
    const wq = new THREE.Quaternion();
    for (const arm of [arms.L, arms.R]) {
      held += +arm.held;
      const Pa = avatarPalm(arm), handPos = wpos(arm.hand.bone);
      arm.fingers.forEach((ch, c) => {
        let r = c === 0 ? Pa.F : wpos(ch[0].bone).sub(handPos).normalize();
        let hinge: THREE.Vector3 | null = null;
        ch.forEach((j, k) => {
          const dir = j.axis.clone().applyQuaternion(j.bone.getWorldQuaternion(wq)).normalize();
          const out = constrainBone(Pa, c, k, r, hinge, dir);
          if (out.outBy > 1) outJoints++;
          joints.push(out.outBy);
          maxOutDeg = Math.max(maxOutDeg, out.outBy);
          hinge = c === 0 ? null : carry(jointFrame(Pa, r, hinge, false).h, r, dir); // carried by where the bone is, not its clamp
          r = dir;
        });
      });
      selfMax = Math.max(selfMax, selfOverlap(capsules(arm)));
      for (const ch of arm.fingers) for (const j of ch) fingerBody = Math.max(fingerBody, body.push(wpos(j.bone), 0).length());
      for (const [b, off] of arm.tips) fingerBody = Math.max(fingerBody, body.push(b.localToWorld(off.clone()), 0).length());
    }
    return { outJoints, maxOutDeg, selfOverlap: selfMax, handOverlap: crossOverlap(limbCapsules(arms.L), limbCapsules(arms.R)), fingerBody, held, joints };
  };

  const setArm = (arm: Arm, p: ArmPose) => {
    arm.upper.bone.quaternion.copy(p.upper);
    arm.fore.bone.quaternion.copy(p.fore);
    arm.hand.bone.quaternion.copy(p.hand);
  };

  const probes = (arm: Arm): THREE.Vector3[] => {
    obj.updateMatrixWorld(true);
    const e = wpos(arm.fore.bone), w = wpos(arm.hand.bone);
    const out = [w, e.clone().lerp(w, 0.5), e.clone().lerp(w, 0.8)];
    // every finger joint with the anatomical hand (the middle joints were not probed before)
    for (const ch of arm.fingers) out.push(wpos(ch[0].bone), ...(HAND_LIMITS ? [wpos(ch[1].bone)] : []), wpos(ch[2].bone));
    for (const [b, off] of arm.tips) out.push(b.localToWorld(off.clone()));
    return out;
  };

  const handProbes = (arm: Arm) => probes(arm).filter((_, i) => i !== 1 && i !== 2); // without the forearm

  /** How far (metres, along z: the avatar faces +z) the hand should move back to make the contact ASL-LEX codes for
   *  this frame. MediaPipe's depth is a guess, so a hand on the chest or chin hovered in front of it; across and up
   *  were right, so only depth changes. `touch`: the nearest hand point onto the body or head surface; `meet`: onto
   *  the other hand, at the pair of points closest across and up. */
  const contactGap = (arm: Arm, touch: number, meet: number, other: Arm | null): number => {
    let gap: number | null = null;
    if (touch > 0.01) {
      for (const p of handProbes(arm)) {
        const fz = body.frontZ(p.x, p.y);
        if (fz !== null && (gap === null || p.z - fz < gap)) gap = p.z - fz;
      }
      // no hand point over the body this frame (the hand at the silhouette's edge): keep the last depth, not snap to none
      if (gap === null) gap = arm.heldGap;
      else arm.heldGap = gap;
      return gap === null ? 0 : touch * (gap - MARGIN);
    }
    if (meet > 0.01 && other) {
      // every pair's depth gap, weighted by how close across and up: the single closest pair switched from frame to
      // frame as one hand slid over the other, and the depth jumped with it (NICE's arm wobbled)
      const theirs = handProbes(other);
      let w = 0, sum = 0;
      for (const p of handProbes(arm)) for (const q of theirs) {
        const k = Math.exp(-((p.x - q.x) ** 2 + (p.y - q.y) ** 2) / (2 * MEET_SIGMA ** 2));
        w += k;
        sum += k * (p.z - q.z);
      }
      if (w > 0) gap = sum / w;
    }
    return gap === null ? 0 : meet * gap;
  };

  /** Pose one arm from the frame, then push hand and forearm out of the body (up to 3 passes). The wrist goes to the
   *  same place *relative to the body* as the signer's (location is part of every sign: chin, forehead, chest);
   *  copying arm directions instead put hands too low and wide on a body with other proportions. */
  /** Where the avatar's elbow and wrist go for this pose: at the same places relative to the body as the signer's. */
  /** A signer point at the same place on the avatar. Near the face it goes relative to the nose instead of the
   *  shoulders: faces sit at different heights and depths on different bodies (and MediaPipe's depth is
   *  exaggerated), and a sign at the chin must end at the chin. */
  const toAvatar = (P: THREE.Vector3[], signer: BodyFrame, q: THREE.Vector3, img?: { pose: number[][]; pose2d: number[][]; aspect: number; wrist: 15 | 16 }) => {
    const near = q.distanceTo(P[0]) / signer.w;
    const wFace = Math.min(1, Math.max(0, (FACE_FAR - near) / (FACE_FAR - FACE_NEAR)));
    // world x and depth can disagree with the image (GRANDMOTHER: thumb at the chin, world wrist by the shoulder): near the face a wrist goes where the image has it across and up
    // only where the two disagree (from DISAGREE.min, fully at .max): where they agree the world pose stays, which the contact data is coded in
    if (img && wFace > 0) {
      const c = wristFrom2d(img.pose, img.pose2d, img.wrist, img.aspect), c3 = new THREE.Vector3(c.x, -c.y, -c.z);
      const off = Math.hypot(c3.x - q.x, c3.y - q.y) / signer.w, t = Math.min(1, Math.max(0, (off - DISAGREE.min) / (DISAGREE.max - DISAGREE.min)));
      q = q.clone().lerp(c3, wFace * t);
    }
    return mapPoint(q, signer, avatarFrame).lerp(mapPoint(q, { ...signer, o: P[0] }, { ...avatarFrame, o: avatarNose }), wFace);
  };

  const armTargets = (arm: Arm, pose: number[][], pose2d?: number[][], aspect?: number) => {
    const P = pose.map(toThree);
    const signer = bodyFrame(P);
    const img = pose2d && aspect ? { pose, pose2d, aspect, wrist: arm.pose[2] as 15 | 16 } : undefined;
    return { el: mapPoint(P[arm.pose[1]], signer, avatarFrame), wrist: toAvatar(P, signer, P[arm.pose[2]], img), P, signer };
  };

  /** Across/up move (metres) bringing the hand point nearest the touched place to within PLACE_SLACK of the avatar's
   *  own version of it: the signer's hand at their forehead could land beside the avatar's (FATHER). */
  const placeShift = (arm: Arm, P: THREE.Vector3[], signer: BodyFrame, at: number[], touch: number) => {
    const target = toAvatar(P, signer, toThree(at));
    let best: THREE.Vector3 | null = null;
    for (const p of handProbes(arm)) if (!best || Math.hypot(p.x - target.x, p.y - target.y) < Math.hypot(best.x - target.x, best.y - target.y)) best = p;
    const d = new THREE.Vector3(target.x - best!.x, target.y - best!.y, 0), len = d.length();
    const slack = target.distanceTo(avatarNose) < 0.15 ? PLACE_SLACK.head : PLACE_SLACK.body;
    return len > slack ? d.multiplyScalar((touch * (len - slack)) / len) : new THREE.Vector3();
  };

  /** A relaxed elbow for this reach: hanging below the shoulder-wrist line, a little out from the side. */
  const hangingElbow = (s: THREE.Vector3, wrist: THREE.Vector3) =>
    s.clone().add(wrist).multiplyScalar(0.5).add(new THREE.Vector3(Math.sign(s.x) * 0.08, -0.25, -0.05));

  /** Pose one arm for this frame at time tMs: every correction (rest, place, contact depth, body push) moves the
   *  solver's targets; the solve then goes through the rotation filters (stepArm), so no correction snaps a bone. */
  const solveArm = (arm: Arm, pose: number[][], pts: number[][] | null, tMs: number, touch = 0, meet = 0, other: Arm | null = null,
    idle = 0, touchAt?: number[]) => {
    const untracked = !pts; // no landmarks at all: the hand relaxes (relaxFingers)
    // a mirrored or garbage hand from the tracker: keep the last good hand rotation and finger shape for this frame
    // the hand checks: a mirrored or garbage hand (two or more fingers bent back), or one covered by the other hand,
    // keeps the last good hand (rotation and shape); one finger bent back keeps only the last good shape, for at most
    // SHAPE_HOLD_MS, so a sign the tracker reads poorly throughout (GIVE: 93% of frames) does not freeze
    let shapePts = pts;
    if (pts && HAND_LIMITS) {
      const right = arm === arms.R, P3 = pts.map(toThree), back = backwardFingers(P3, right);
      const mine = right ? curFrame?.right : curFrame?.left, theirs = right ? curFrame?.left : curFrame?.right;
      const covered = !!mine && !!theirs && !!curFrame && coveredHand(mine, theirs, curFrame.pose[arm.pose[2]][2], curFrame.pose[right ? 15 : 16][2]);
      arm.held = back >= 2 || covered;
      if (!arm.held && back === 1 && arm.goodPts && arm.shapeHeldMs < SHAPE_HOLD_MS) {
        shapePts = arm.goodPts;
        arm.shapeHeldMs += Math.min(tMs - arm.solvedAt, 100) || 0;
      } else if (!arm.held) {
        if (back === 0) arm.goodPts = pts;
        arm.shapeHeldMs = 0;
      }
    } else arm.held = false;
    if (arm.held) pts = shapePts = null;
    const { el, wrist, P, signer } = armTargets(arm, pose, curFrame?.pose2d, curFrame?.aspect);
    if (idle > 0) { // the hand left the video: MediaPipe's guess of its wrist bent the arm onto the hip
      const r = armTargets(arm, DEFAULT_REST.pose);
      el.lerp(r.el, idle);
      wrist.lerp(r.wrist, idle);
    }
    const side = arm === arms.L ? "L" : "R";
    { // a raised arm lifts the collarbone too; pivoting at a fixed shoulder pinched it (HELLO). Set before the solves: it moves the shoulder
      clavicle[side].quaternion.copy(clavicleBind[side]);
      const s0 = wpos(arm.upper.bone), up = el.clone().sub(s0).normalize().y;
      const lift = Math.min(SHRUG_MAX, SHRUG * Math.max(0, Math.asin(up) + 0.17));
      const P = parentWorld(clavicle[side]);
      const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, Math.sign(s0.x)), lift);
      const fwd = PROTRACT_MAX * Math.max(0, Math.min(1, (wrist.z - s0.z) / PROTRACT_REACH)); // the shoulder reaches forward with the hand
      if (Number.isFinite(fwd)) q.premultiply(new THREE.Quaternion().setFromAxisAngle(UP, -Math.sign(s0.x) * fwd));
      const target = P.clone().invert().multiply(q).multiply(P).multiply(clavicleBind[side]);
      if (Number.isFinite(target.x + target.y + target.z + target.w)) clavicle[side].quaternion.copy(clavicleFilter[side].filter(target, tMs)); // NaN would poison the filter; // filtered: a lift never snaps the shoulder
    }
    const s = wpos(arm.upper.bone);
    el.lerp(hangingElbow(s, wrist), ELBOW_DOWN);
    if (pts) {
      const R = handRotation(arm, pts);
      if (Number.isFinite(R.x + R.y + R.z + R.w)) arm.seen = R; // a NaN landmark keeps the last good rotation
    }
    const t: ArmTarget = { shoulder: s, elbowHint: el, wrist, handRot: idle > 0.999 ? null : arm.seen };
    if (t.handRot && idle > 1e-3) { // the hand appears or leaves: its turn eases in and out with the arm, from the relaxed hand
      const p = solveArmPose(arm.rig, { ...t, handRot: null }, arm.last, LIMITS);
      if (OLD_MOTION) t.handRot = parentWorld(arm.upper.bone).multiply(p.upper).multiply(p.fore).multiply(p.hand).slerp(t.handRot, 1 - idle);
      else ({ rot: t.handRot, ease: arm.ease } = easeHand(arm.rig, p, t.handRot, 1 - idle, !untracked, arm.ease));
    } else arm.ease = null;
    // trial solves (unfiltered) show where the hand would land, for the place and contact corrections to the wrist target
    const trial = () => {
      setArm(arm, solveArmPose(arm.rig, t, arm.last, LIMITS));
      if (shapePts) placeFingers(arm, shapePts);
    };
    if (touch <= 0.01) arm.heldGap = null;
    if (TOUCH && touch > 0.01 && touchAt) {
      trial();
      const shift = placeShift(arm, P, signer, touchAt, touch);
      if (shift.lengthSq() > 1e-8) wrist.add(shift);
    }
    if (TOUCH && (touch > 0.01 || (meet > 0.01 && other))) {
      trial();
      wrist.z -= contactGap(arm, touch, meet, other);
    }
    for (let pass = 0; pass < (COLLIDE ? 3 : 0); pass++) { // the body push, on the targets: hand and forearm out of the body
      trial();
      const e = wpos(arm.fore.bone), pe = body.push(e, MARGIN * 2); // elbow tucked into the side: swing it out
      let worst = new THREE.Vector3();
      for (const p of probes(arm)) {
        const d = body.push(p, MARGIN);
        if (d.lengthSq() > worst.lengthSq()) worst = d;
      }
      const ph = handPush(arm, other, tMs); // and out of the other hand, once that hand is posed for this frame
      if (worst.lengthSq() < 1e-8 && pe.lengthSq() < 1e-8 && ph.lengthSq() < 1e-8) break;
      wrist.add(worst).add(ph);
      el.copy(e.add(pe)).add(ph); // the pushed elbow sets the bend plane; out of the other arm, the whole forearm moves
    }
    const prev = arm.last;
    let out = stepArm(arm.rig, arm.filters, t, prev, tMs, LIMITS);
    setArm(arm, out);
    if (shapePts) placeFingers(arm, shapePts, tMs);
    else if (untracked && HAND_LIMITS) relaxFingers(arm, tMs);
    // Safety: the filtered arm lags its targets and can cut into the body. Push it out from where it is, by its depth
    // (plus a margin that grows with the depth), so the correction starts at zero and stays continuous; it is a
    // function of the filtered out and is never fed back into the filters, and goes through the turn cap and limits
    const deepest = () => {
      let worst: THREE.Vector3 | null = null, depth = 0;
      for (const p of probes(arm)) {
        const d = body.push(p, 0).length();
        if (d > depth) [depth, worst] = [d, p];
      }
      return { worst, depth };
    };
    let cur = COLLIDE ? deepest() : { worst: null, depth: 0 }, forward = false;
    for (let pass = 0; pass < 3 && cur.worst && cur.depth > 1e-5; pass++) {
      const e = wpos(arm.fore.bone), m = Math.min(MARGIN, cur.depth);
      // a hand wedged between two surfaces (fingers under the chin, tips on the chest) is pushed straight forward instead
      const push = forward ? new THREE.Vector3(0, 0, cur.depth + m) : body.push(cur.worst, m);
      const fix: ArmTarget = { shoulder: s, wrist: wpos(arm.hand.bone).add(push), elbowHint: e.add(body.push(e, 2 * m)),
        handRot: arm.hand.bone.getWorldQuaternion(new THREE.Quaternion()) };
      const was = out;
      out = settlePose(arm.rig, solveArmPose(arm.rig, fix, out, LIMITS), prev, tMs, LIMITS);
      setArm(arm, out);
      const next = deepest();
      if (next.depth > cur.depth && !forward) { // that push only moved another hand point deeper: undo it, go forward
        setArm(arm, (out = was));
        forward = true;
      } else cur = next;
    }
    // the same for the other hand, on the pose as it will be shown (the push on the targets above is undone in part by
    // the filters): out of its fingers, unless that would put this hand back into the body
    for (let pass = 0; pass < 2; pass++) {
      const ph = handPush(arm, other, tMs);
      if (ph.lengthSq() < 1e-8) break;
      const e = wpos(arm.fore.bone), was = out, depth0 = deepest().depth;
      const fix: ArmTarget = { shoulder: s, wrist: wpos(arm.hand.bone).add(ph), elbowHint: e.clone().add(ph),
        handRot: arm.hand.bone.getWorldQuaternion(new THREE.Quaternion()) };
      out = settlePose(arm.rig, solveArmPose(arm.rig, fix, out, LIMITS), prev, tMs, LIMITS);
      setArm(arm, out);
      if (COLLIDE && deepest().depth > depth0 + 1e-4) {
        setArm(arm, (out = was));
        break;
      }
    }
    arm.last = out;
    arm.solvedAt = tMs;
  };

  /** Wrist move (metres) taking this hand's fingers out of the other hand's, which was posed first this frame: hands
   *  that meet (NICE, READY) touch instead of passing through each other. */
  const handPush = (arm: Arm, other: Arm | null, tMs: number) => {
    const push = new THREE.Vector3();
    if (!HAND_LIMITS || !other || other.solvedAt !== tMs) return push;
    const theirs = limbCapsules(other);
    for (const x of limbCapsules(arm)) for (const y of theirs) {
      const { depth, dir } = capsuleContact(x, y, 0.003, new THREE.Vector3(0, 0, 1));
      if (depth > 0 && depth > push.length()) push.copy(dir).multiplyScalar(depth);
    }
    if (push.z < 0) push.z = 0; // never back into the torso: two hands working together in front of the chest (MAKE: 30 mm deep)
    return push;
  };

  /** Restart every filter and the carried pose (the clock went back: a new bench item); the next frame passes through
   *  unfiltered, and no twist demand carries over from the last plan. */
  const resetFilters = () => {
    for (const arm of [arms.L, arms.R]) {
      arm.last = arm.seen = arm.ease = arm.heldGap = arm.goodPts = null;
      arm.shapeHeldMs = 0;
      for (const f of Object.values(arm.filters)) f.reset();
      for (const ch of arm.fingerFilters) for (const f of ch) f.reset();
    }
    clavicleFilter.L.reset();
    clavicleFilter.R.reset();
    headMotion = createHeadMotion();
    for (const [, sp] of springs) sp.reset();
  };
  // one spring per driven bone: collarbone, upper arm, forearm, hand, the 15 finger bones, each side
  const springs: [THREE.Object3D, RotationSpring][] = [];
  for (const k of ["L", "R"] as const) {
    springs.push([clavicle[k], new RotationSpring(SPRING.shoulder)]);
    for (const j of [arms[k].upper, arms[k].fore]) springs.push([j.bone, new RotationSpring(SPRING.arm)]);
    springs.push([arms[k].hand.bone, new RotationSpring(SPRING.hand)]);
    for (const ch of arms[k].fingers) for (const j of ch) springs.push([j.bone, new RotationSpring(SPRING.finger)]);
  }
  let smoothAt = -Infinity;
  /** The solved pose on the bones becomes the springs' target; the springs' output goes back onto the bones. */
  const smoothBones = (nowMs: number) => {
    const dt = (nowMs - smoothAt) / 1000;
    smoothAt = nowMs;
    for (const [b, sp] of springs) b.quaternion.copy(SMOOTH ? sp.step(b.quaternion, dt) : (sp.reset(), b.quaternion));
  };
  let lastNow = -Infinity;
  let headMotion = createHeadMotion();
  let curFrame: ClipFrame | null = null; // the frame being applied: its image landmarks and aspect place the wrist near the face

  const idle = { L: 0, R: 0 }; // 1 = that hand is untracked and its arm rests
  const setMorph = (name: string, v: number) => {
    for (const m of morphMeshes) {
      const i = m.morphTargetDictionary![morphName(name)];
      if (i !== undefined) m.morphTargetInfluences![i] = v;
    }
  };
  const driven = new Set<string>();

  // idle life: slow breathing in the upper spine, blinks at irregular 2.5-6 s intervals
  const chest = bone("Spine2"), chestBind = chest.quaternion.clone(), side = new THREE.Vector3(1, 0, 0);
  let breath = 0;
  const setChest = () => {
    const P = parentWorld(chest);
    chest.quaternion.copy(P.clone().invert().multiply(yaw()).multiply(P).multiply(chestBind))
      .multiply(new THREE.Quaternion().setFromAxisAngle(side, breath));
  };
  let nextBlink = performance.now() + 2500, blinkAt = -Infinity;
  const resize = () => {
    const { clientWidth: w, clientHeight: h } = canvas;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
  // Framing: one fixed view of the whole signing space (VIEW), fitted to the canvas shape. It used to ease back to fit a
  // raised hand's fingertips and drift in again, so the avatar zoomed in and out with nearly every sign (the median sign
  // dipped 18 cm under the old lower edge). ?view=face|eyes|side keep their fixed cameras.
  const FIT_TAN = Math.tan((camera.fov * Math.PI) / 360);
  const fitCamera = () => {
    if (view) return;
    const y = (VIEW.lo + VIEW.hi) / 2, d = Math.max((VIEW.hi - VIEW.lo) / 2 / FIT_TAN, VIEW.half / (FIT_TAN * camera.aspect));
    camera.position.set(0, y, d);
    camera.lookAt(0, y, 0);
  };
  new ResizeObserver(fitCamera).observe(canvas);
  fitCamera();
  let drawnAt = -Infinity;
  renderer.setAnimationLoop((now) => {
    if (!canvas.isConnected) return; // teaching mode moves its one canvas between screens: nothing to draw while it is on none
    if (opts.maxFps && now - drawnAt < 1000 / opts.maxFps - 4) return; // the 4 ms: a 60 Hz frame landing early isn't skipped
    drawnAt = now;
    breath = 0.006 * Math.sin((2 * Math.PI * now) / 4300);
    setChest();
    if (now > nextBlink) {
      blinkAt = now;
      nextBlink = now + 2500 + Math.random() * 3500;
    }
    const t = (now - blinkAt) / 170;
    for (const b of BLINK) setMorph(b, t >= 0 && t <= 1 ? Math.sin(Math.PI * t) : 0);
    renderer.render(scene, camera);
  });

  for (const n of BROW_DOWN) setMorph(n, REST_BROW_DOWN); // the resting face before the first sign
  for (const n of SQUINT) setMorph(n, REST_SQUINT);

  return {
    applyFrame(s, nowMs = performance.now()) { // nowMs: the rotation filters' and idle ease's clock
      if (nowMs < lastNow) resetFilters();
      const dt = nowMs < lastNow ? Infinity : nowMs - lastNow; // restart or first frame: idle snaps
      lastNow = nowMs;
      curFrame = s;
      if (!STILL) { // signers turn their shoulders toward the working hand; MediaPipe's shoulder depth is loose: damped, capped
        const d = toThree(s.pose[11]).sub(toThree(s.pose[12]));
        const target = Math.max(-TORSO_MAX, Math.min(TORSO_MAX, TORSO_SHARE * Math.atan2(-d.z, d.x)));
        torsoYaw += (target - torsoYaw) * (1 - (1 - TORSO_EASE) ** (Math.min(dt, 100) / (1000 / 60))); // time-based; a gap counts as 100 ms
        setChest();
      }
      // clip hands are keyed by the signer's own side (training/build_motion_library.by_wrist); world landmarks preferred
      const ease = 1 - Math.exp(-dt / IDLE_EASE);
      for (const [k, hand] of [["L", s.leftW ?? s.left], ["R", s.rightW ?? s.right]] as const) idle[k] += ((hand ? 0 : 1) - idle[k]) * ease;
      const L = () => solveArm(arms.L, s.pose, (s.leftW ?? s.left) as number[][] | null, nowMs, s.touchL, s.meetL, arms.R, idle.L, s.touchAtL);
      const R = () => solveArm(arms.R, s.pose, (s.rightW ?? s.right) as number[][] | null, nowMs, s.touchR, s.meetR, arms.L, idle.R, s.touchAtR);
      if (s.meetL) { R(); L(); } else { L(); R(); } // a hand meeting the other is placed after it
      smoothBones(nowMs);
    },
    armState() {
      obj.updateMatrixWorld(true);
      return { L: armState("L"), R: armState("R") };
    },
    palmScreen() {
      const at = (k: "L" | "R") => ndcToCanvas(wpos(arms[k].hand.bone).add(wpos(arms[k].fingers[2][0].bone)).multiplyScalar(0.5).project(camera),
        canvas.clientWidth, canvas.clientHeight);
      return { L: at("L"), R: at("R") };
    },
    contactError(f, at) {
      obj.updateMatrixWorld(true);
      const P = f.pose.map(toThree), signer = bodyFrame(P), errs: number[] = [];
      for (const k of ["L", "R"] as const) {
        const touch = (k === "L" ? f.touchL : f.touchR) ?? 0, touchAt = k === "L" ? f.touchAtL : f.touchAtR;
        const place = at ? (at.hand === k ? at.place : undefined) : touch >= 0.5 ? touchAt : undefined;
        if (!place) continue;
        const target = toAvatar(P, signer, toThree(place));
        errs.push(Math.min(...handProbes(arms[k]).map((p) => p.distanceTo(target))));
      }
      return errs.length ? errs.reduce((a, b) => a + b) / errs.length : null;
    },
    /** Right hand knuckle / middle-joint bend per finger (degrees), measured on the posed skeleton (evaluation). */
    fingerBends() {
      const deg = (a: THREE.Vector3, b: THREE.Vector3) => Math.round((a.angleTo(b) * 180) / Math.PI);
      const w = wpos(arms.R.hand.bone), mid = wpos(arms.R.fingers[2][0].bone);
      return arms.R.fingers.slice(1).map((ch) => {
        const [k, p, d] = ch.map((j) => wpos(j.bone));
        return [deg(mid.clone().sub(w), p.clone().sub(k)), deg(p.clone().sub(k), d.clone().sub(p))];
      });
    },
    penetration() {
      let depth = 0;
      for (const arm of [arms.L, arms.R]) for (const p of probes(arm)) depth = Math.max(depth, body.push(p, 0).length());
      return depth;
    },
    armPenetration() {
      let depth = 0;
      for (const arm of [arms.L, arms.R]) {
        const sh = wpos(arm.upper.bone), e = wpos(arm.fore.bone);
        for (const p of [e, sh.clone().lerp(e, 0.75)]) depth = Math.max(depth, body.push(p, 0).length());
      }
      return depth;
    },
    handReport,
    /** Every fingertip and wrist (world metres) and the framing references, for sizing the camera (evaluation). */
    framePoints: (all = false) => {
      obj.updateMatrixWorld(true);
      const pts: number[][] = [];
      for (const k of ["L", "R"] as const) {
        if (!all && idle[k] > 0.5) continue;
        pts.push(wpos(arms[k].hand.bone).toArray());
        for (const [b, off] of arms[k].tips) pts.push(b.localToWorld(off.clone()).toArray());
        pts.push(wpos(arms[k].fore.bone).toArray());
      }
      return { pts, pelvisY: wpos(bone("Pelvis")).y, headTopY: SPACE.hi - 0.03, shoulderHalf: naturalWidth / 2, armLength: arms.L.rig.l1 + arms.L.rig.l2, upperArm: arms.L.rig.l1 };
    },
    setHandLimits(on: boolean) {
      HAND_LIMITS = on;
    },
    setSmoothing(on: boolean) {
      SMOOTH = on;
    },
    headHeight: () => wpos(headBone).y,
    activeMorphs: () => morphMeshes.map((m) => [m.name, Object.entries(m.morphTargetDictionary!).filter(([, i]) => m.morphTargetInfluences![i] > 0.01).map(([n, i]) => `${n}=${m.morphTargetInfluences![i].toFixed(2)}`)]),
    applyFace(p: FacePose) {
      const P = parentWorld(headBone);
      obj.updateMatrixWorld(true);
      const m = headMotion(lastNow, wpos(arms.L.hand.bone), wpos(arms.R.hand.bone), avatarNose); // idle drift and the turn toward the working hand, on top of the grammar pose
      const turn = new THREE.Quaternion().setFromEuler(new THREE.Euler(((p.headPitchDeg + m.pitchDeg) * Math.PI) / 180, ((p.headYawDeg + m.yawDeg) * Math.PI) / 180 - HEAD_KEEP * torsoYaw, (m.rollDeg * Math.PI) / 180, "YXZ"));
      headBone.quaternion.copy(P.clone().invert().multiply(turn).multiply(P).multiply(headBind)); // world-axis nod/shake
      const values = new Map<string, number>();
      for (const n of BROW_UP) values.set(n, p.browUp);
      for (const n of BROW_DOWN) values.set(n, Math.max(p.browDown, REST_BROW_DOWN * (1 - p.browUp)));
      for (const n of SQUINT) values.set(n, REST_SQUINT * (1 - p.browUp));
      for (const [k, v] of Object.entries(p.expressions)) for (const [n, w] of TONE[k] ?? []) values.set(n, Math.max(values.get(n) ?? 0, v * w));
      for (const n of driven) if (!values.has(n)) values.set(n, 0); // release shapes from the previous pose
      for (const [n, v] of values) {
        setMorph(n, v);
        driven.add(n);
      }
    },
  };
}
