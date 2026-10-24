"""Build the stylized "toon" signing avatar: a cute, simple, non-realistic character on the MPFB character's rig.

Style (after Duolingo's shape language and chibi guides): every part a rounded primitive (no points), flat saturated
colours, a large round head with big solid eyes, a dot nose and a small mouth, slimmer neck. Arms and hands keep adult
proportions on the same skeleton, so signing space, the arm solver and the anatomical hand (handModel.ts) are unchanged.
The skeleton (53 bones, names and bind pose) is copied from web/public/avatar/mpfb/avatar.glb, so web/src/avatar/rocketbox.ts
drives it like the MPFB character; the face carries the 20 ARKit shapes the app uses, named as MPFB's (browDownLeft, ...).

Two skinned meshes: toon_body (skin, shirt, hair, nose, ears; vertex colours; the hands' collision surface) and toon_face
(eyes, brows, mouth, cheeks; the morph targets, which are built by re-generating the features with changed parameters).

Run (from app/): uv run --with pygltflib python scripts/build_toon_avatar.py [web/public/avatar/toon]
"""
import sys
from pathlib import Path

import numpy as np
import pygltflib as G

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "public" / "avatar" / "mpfb" / "avatar.glb"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "web" / "public" / "avatar" / "toon"


def srgb(h: str) -> np.ndarray:  # glTF vertex colours are linear
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


SKIN, SHIRT, PANTS, SHOE = srgb("#F4C4A0"), srgb("#2F5FD0"), srgb("#2B2F4A"), srgb("#F2F2F2")
HAIR, EYE, SHINE, BROW = srgb("#6B3F2A"), srgb("#2A1E1A"), srgb("#FFFFFF"), srgb("#4A2E22")
MOUTH, BLUSH, NOSE = srgb("#8A3B3B"), srgb("#FF9AA2"), srgb("#EDB08E")
HAIR_LIGHT = srgb("#8C5A3E")  # the lighter strands in the hair's texture

# ---- skeleton from the MPFB glb ----
src = G.GLTF2().load(str(SRC))
blob = src.binary_blob()


def read(i: int) -> np.ndarray:
    a = src.accessors[i]
    bv = src.bufferViews[a.bufferView]
    n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a.type]
    return np.frombuffer(blob, np.float32, a.count * n, (bv.byteOffset or 0) + (a.byteOffset or 0)).reshape(a.count, n)


skin0 = src.skins[0]
NAMES = [src.nodes[j].name for j in skin0.joints]
IBM = read(skin0.inverseBindMatrices)
P = {n: np.linalg.inv(IBM[i].reshape(4, 4).T)[:3, 3] for i, n in enumerate(NAMES)}  # bind positions (glTF is column-major)
J = {n: i for i, n in enumerate(NAMES)}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def tail(n: str) -> np.ndarray:
    """End of bone n: its child's joint, else carried on from its parent."""
    nxt = {"pelvis": "spine_01", "spine_01": "spine_02", "spine_02": "spine_03", "spine_03": "neck_01", "neck_01": "head"}
    if n in nxt:
        return P[nxt[n]]
    if n == "head":
        return P["head"] + np.array([0, 0.2, 0])
    s = n[-1]
    for a, b in (("clavicle", "upperarm"), ("upperarm", "lowerarm"), ("lowerarm", "hand"), ("hand", "middle_01"),
                 ("thigh", "calf"), ("calf", "foot"), ("foot", "ball")):
        if n == f"{a}_{s}":
            return P[f"{b}_{s}"]
    if n.startswith("ball"):
        return P[n] + np.array([0, 0, 0.06])
    f, k = n.split("_")[0], int(n.split("_")[1])
    if k < 3:
        return P[f"{f}_0{k + 1}_{s}"]
    return P[n] + (P[n] - P[f"{f}_02_{s}"]) * (0.45 if f == "thumb" else 0.65)  # tip (short, rounded; the thumb shortest)


# ---- geometry: every part a loft of rings (closed with end fans) ----
class Part:
    def __init__(self, V, F, color, bones, sharp=4.0, bias=None):
        self.V, self.F, self.color, self.bones, self.sharp, self.bias = V, F, color, bones, sharp, bias or {}


def loft(centers, U, W, ru, rw, expo=2.0, seg=24, caps=True):
    """Rings of superellipses (exponent expo: 2 = ellipse, higher = rounder box) around centers in the U/W planes;
    caps=False leaves both ends open."""
    phi = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    cu, sw = np.sign(c) * np.abs(c) ** (2 / expo), np.sign(s) * np.abs(s) ** (2 / expo)
    rings = [ce + np.outer(cu * a, u) + np.outer(sw * b, w) for ce, u, w, a, b in zip(centers, U, W, ru, rw)]
    V = np.vstack(rings + [centers[0][None], centers[-1][None]])
    n, F = len(centers), []
    for i in range(n - 1):
        for j in range(seg):
            a, b, c2, d = i * seg + j, i * seg + (j + 1) % seg, (i + 1) * seg + (j + 1) % seg, (i + 1) * seg + j
            F += [(a, b, c2), (a, c2, d)]
    if not caps:
        return V[:-2], np.array(F)
    s0, s1 = n * seg, n * seg + 1
    for j in range(seg):
        F += [(s0, (j + 1) % seg, j), (s1, (n - 1) * seg + j, (n - 1) * seg + (j + 1) % seg)]
    return V, np.array(F)


def rim(center, U, W, ru, rw, tube, expo=2.0, seg=48, tseg=8):
    """A rounded lip (a tube of radius `tube`) around the superellipse ring at center: hems, collars and cuffs, so an
    opening reads as cloth with an edge."""
    phi = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s_ = np.cos(phi), np.sin(phi)
    path = center + np.outer(np.sign(c) * np.abs(c) ** (2 / expo) * ru, U) + np.outer(np.sign(s_) * np.abs(s_) ** (2 / expo) * rw, W)
    out_ = path - center
    out_ /= np.linalg.norm(out_, axis=1, keepdims=True)
    ax = np.cross(U, W)
    th = np.linspace(0, 2 * np.pi, tseg, endpoint=False)
    V = (path[:, None] + tube * (np.cos(th)[None, :, None] * out_[:, None] + np.sin(th)[None, :, None] * ax)).reshape(-1, 3)
    F = []
    for i in range(seg):
        for j in range(tseg):
            a, b = i * tseg + j, i * tseg + (j + 1) % tseg
            c2, d = ((i + 1) % seg) * tseg + (j + 1) % tseg, ((i + 1) % seg) * tseg + j
            F += [(a, b, c2), (a, c2, d)]
    return V, np.array(F)


def frame(axis):
    a = axis / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def capsule(p, q, rp, rq, seg=16, cap=6, mid=0):
    """Tapered capsule from p (radius rp) to q (rq); mid extra rings along the shaft (smooth bending when skinned)."""
    a, u, w = frame(q - p)
    L = np.linalg.norm(q - p)
    prof = [(-rp * np.sin(t), rp * np.cos(t)) for t in np.linspace(np.pi / 2, 0, cap)]
    prof += [(L * t, rp + (rq - rp) * t) for t in np.linspace(0, 1, mid + 2)[1:-1]]
    prof += [(L + rq * np.sin(t), rq * np.cos(t)) for t in np.linspace(0, np.pi / 2, cap)]
    prof = [(x, max(r, 1e-4)) for x, r in prof]
    cs = [p + a * x for x, _ in prof]
    return loft(cs, [u] * len(cs), [w] * len(cs), [r for _, r in prof], [r for _, r in prof], seg=seg)


def ellipsoid(c, r, axes=np.eye(3), seg=24, rings=16):
    """Ellipsoid at c, radii r along the rows of axes (x, y, z); rings run along y."""
    t = np.linspace(np.pi / 2, -np.pi / 2, rings)
    cs = [c + axes[1] * r[1] * np.sin(x) for x in t]
    k = [max(np.cos(x), 1e-3) for x in t]
    return loft(cs, [axes[0]] * rings, [axes[2]] * rings, [r[0] * x for x in k], [r[2] * x for x in k], seg=seg)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


parts: list[Part] = []
X, Y, Z = np.eye(3)

# The body under the clothes (hips in the trousers' colour, the chest skin, showing at the openings), and a loose tee
# over it: an open shell a little away from the body, flaring at the hem, with open sleeves and rounded lips at the hem,
# collar and cuffs, so it reads as a garment worn over the body rather than one painted onto it.
CORE = [  # y, half width, half depth, centre z
    (0.98, 0.150, 0.105, 0.005), (1.04, 0.166, 0.106, 0.005), (1.12, 0.158, 0.100, 0.005), (1.26, 0.148, 0.094, 0.0),
    (1.38, 0.152, 0.096, 0.0), (1.48, 0.168, 0.100, 0.004), (1.56, 0.190, 0.098, 0.003), (1.61, 0.196, 0.090, 0.0),
    (1.65, 0.178, 0.080, 0.0), (1.685, 0.144, 0.070, 0.004), (1.715, 0.100, 0.060, 0.008), (1.745, 0.060, 0.048, 0.012),
]
TEE = [
    (1.035, 0.194, 0.130, 0.006), (1.07, 0.186, 0.124, 0.006), (1.16, 0.176, 0.117, 0.004), (1.26, 0.170, 0.112, 0.0),
    (1.38, 0.175, 0.116, 0.0), (1.48, 0.190, 0.120, 0.004), (1.56, 0.210, 0.116, 0.003), (1.61, 0.216, 0.108, 0.0),
    (1.65, 0.198, 0.098, 0.0), (1.685, 0.164, 0.086, 0.004), (1.715, 0.118, 0.072, 0.008), (1.738, 0.084, 0.062, 0.01),
    (1.748, 0.072, 0.056, 0.012),
]  # shoulders slope down from the collar and round over into the sleeves
TORSO_BONES = ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r"]
TORSO_BIAS = {"upperarm_l": 0.12, "upperarm_r": 0.12, "thigh_l": 0.5, "thigh_r": 0.5}


def ring(rows):
    return [np.array([0, y, cz]) for y, _, _, cz in rows], [X] * len(rows), [Z] * len(rows), [w for _, w, _, _ in rows], [d for _, _, d, _ in rows]


for rows, col, bones in ((CORE[:3], PANTS, ["pelvis", "spine_01", "thigh_l", "thigh_r"]), (CORE[2:], SKIN, TORSO_BONES)):
    parts.append(Part(*loft(*ring(rows), expo=2.6, seg=40), col, bones, 4, TORSO_BIAS))
parts.append(Part(*loft(*ring(TEE), expo=2.6, seg=48, caps=False), SHIRT, TORSO_BONES, 4, TORSO_BIAS))
for (y, w, d, cz), tube, bones in ((TEE[0], 0.008, ["pelvis", "spine_01"]), (TEE[-1], 0.007, ["spine_03", "neck_01", "clavicle_l", "clavicle_r"])):
    parts.append(Part(*rim(np.array([0, y, cz]), X, Z, w, d, tube, expo=2.6), SHIRT, bones, 4))

for s in "lr":
    sh, el, wr = P[f"upperarm_{s}"], P[f"lowerarm_{s}"], P[f"hand_{s}"]
    # sleeve: an open tube that starts narrow, buried inside the shoulder, swells into a rounded cap over the joint and
    # flares a little to a lipped cuff at mid upper arm; its inner part follows the collarbone and its outer the arm, so
    # it stretches over the shoulder as the arm moves instead of sitting on it like a pad
    a, u, w = frame(el - sh)
    s0, s1 = sh + np.array([-0.07 if s == "l" else 0.07, -0.015, 0.0]), sh + 0.56 * (el - sh)
    ts = np.linspace(0, 1, 12)
    sl = [s0 + t * (s1 - s0) for t in ts]
    rs = [0.034 + 0.024 * smooth(0.0, 0.45, t) + 0.006 * t for t in ts]
    sleeve_bones = [f"clavicle_{s}", f"upperarm_{s}"]
    parts.append(Part(*loft(sl, [u] * 12, [w] * 12, rs, rs, seg=48, caps=False), SHIRT, sleeve_bones, 8))
    parts.append(Part(*rim(sl[-1], u, w, rs[-1], rs[-1], 0.007, seg=48), SHIRT, [f"upperarm_{s}"], 8))
    # arms: soft at the elbow; the forearm stays straight to the wrist (no rubbery bend there), a ball at the wrist joint.
    # Fine tessellation: seen up close on a phone, the toon ramp turns every facet of a coarse tube into a visible stripe
    parts.append(Part(*capsule(sh + 0.42 * (el - sh), el, 0.044, 0.040, seg=48, cap=10), SKIN, [f"upperarm_{s}", f"lowerarm_{s}"], 8))
    parts.append(Part(*capsule(el, wr, 0.040, 0.030, seg=48, cap=10), SKIN, [f"upperarm_{s}", f"lowerarm_{s}"], 8))
    parts.append(Part(*ellipsoid(wr, np.array([0.029, 0.029, 0.029]), seg=40, rings=24), SKIN, [f"lowerarm_{s}"]))
    # palm: rounded box from the wrist to the knuckles in the palm frame (wrist -> middle knuckle, little -> index), rigid
    Fd = P[f"middle_01_{s}"] - wr
    L = np.linalg.norm(Fd)
    Fd /= L
    Sd = P[f"index_01_{s}"] - P[f"pinky_01_{s}"]
    Sd -= Fd * Sd.dot(Fd)
    Sd /= np.linalg.norm(Sd)
    Nd = np.cross(Sd, Fd)
    half = 0.5 * np.linalg.norm(P[f"index_01_{s}"] - P[f"pinky_01_{s}"]) + 0.013
    off = Sd * (0.5 * (P[f"index_01_{s}"] + P[f"pinky_01_{s}"]) - wr).dot(Sd)  # the knuckle line's centre sits off the wrist-middle line
    ts = np.linspace(-0.012, L + 0.004, 9)
    cs = [wr + Fd * t + off * np.clip(t / L, 0, 1) for t in ts]
    wide = [0.032 + (half - 0.032) * np.clip(t / L, 0, 1) for t in ts]
    thick = [0.021 - 0.004 * np.clip(t / L, 0, 1) for t in ts]
    for i in (0, -1):  # round the ends
        wide[i] *= 0.75
        thick[i] *= 0.75
    parts.append(Part(*loft(cs, [Sd] * 9, [Nd] * 9, wide, thick, expo=2.8, seg=48), SKIN, [f"hand_{s}"], 8))
    # fingers: each segment rigid on its own bone (a finger bends only at its joints, never along a segment), rounded
    # ends overlapping at the joints; the thumb a little shorter and slimmer toward its tip
    for f, r in zip(FINGERS, (0.0150, 0.0110, 0.0113, 0.0106, 0.0097)):
        for k in (1, 2, 3):
            b = f"{f}_0{k}_{s}"
            rr = r * (1 - (0.09 if f == "thumb" else 0.06) * (k - 1))
            parts.append(Part(*capsule(P[b], tail(b), rr, rr * 0.97, seg=24, cap=7), SKIN, [b], 10))
    # legs and shoes (mostly below the interpreter framing)
    th, ca, fo, ba = P[f"thigh_{s}"], P[f"calf_{s}"], P[f"foot_{s}"], P[f"ball_{s}"]
    parts.append(Part(*capsule(th, ca, 0.075, 0.058), PANTS, ["pelvis", f"thigh_{s}", f"calf_{s}"], 6))
    parts.append(Part(*capsule(ca, fo, 0.056, 0.045), PANTS, [f"thigh_{s}", f"calf_{s}", f"foot_{s}"], 6))
    parts.append(Part(*capsule(fo + np.array([0, -0.03, -0.03]), ba + np.array([0, -0.045, 0.05]), 0.05, 0.045), SHOE, [f"foot_{s}", f"ball_{s}"], 6))

# head: big, round, a little wider at the cheeks; the face is painted on (features follow the surface)
HEAD = 1.15  # head size: the head, ears, hair and every face feature scale together; the chin stays where it was
HC, HR = np.array([0.0, 1.905 + (HEAD - 1) * 0.133, 0.045]), HEAD * np.array([0.130, 0.133, 0.123])  # nearly a ball: rounder reads younger
parts.append(Part(*capsule(P["neck_01"] + np.array([0, -0.02, -0.005]), HC + np.array([0, -0.68 * HR[1], -0.01]), 0.052, 0.050), SKIN, ["spine_03", "neck_01", "head"], 4))
parts.append(Part(*ellipsoid(HC, HR, seg=48, rings=32), SKIN, ["head"]))
for sx in (1, -1):
    parts.append(Part(*ellipsoid(HC + HEAD * np.array([sx * 0.126, -0.016, -0.004]), HEAD * np.array([0.016, 0.028, 0.022])), SKIN, ["head"]))

# hair: a side-part swoop, one closed shell (no cards) that sinks inside the head below its edge and around the face and
# ears. The part sits on the character's left; from it a soft roll of hair sweeps across the forehead and down to the
# right temple, its edge kept above the brows (they carry question grammar)
HH = HR * np.array([1.09, 1.1, 1.1])
Vh, Fh = ellipsoid(HC, HH, seg=200, rings=140)  # fine enough that the slanted edge and the shading on it stay smooth
d = (Vh - HC) / HH
ang = np.arctan2(d[:, 0], d[:, 2])
fringe = 0.53 + 0.15 * np.tanh(2.0 * ang) - 0.06 * smooth(0.7, 1.3, np.abs(ang))  # high at the part, sweeping low to the right temple
front = smooth(0.0, 0.35, d[:, 2]) * (1 - smooth(1.15, 1.45, np.abs(ang)))  # the face side, not over the ears
show = np.minimum(1 - front * (1 - smooth(fringe - 0.035, fringe + 0.035, d[:, 1])), smooth(-0.55, -0.35, d[:, 1] + 0.25 * np.clip(-d[:, 2], 0, 1)))
show = np.minimum(show, 1 - smooth(0.85, 1.0, np.abs(d[:, 0])) * (1 - smooth(0.05, 0.25, d[:, 1])))
# no sideburns: from the temples back past the ears the hair stops above ear level (behind the ears it falls to the nape)
show = np.minimum(show, 1 - smooth(0.75, 1.0, np.abs(ang)) * (1 - smooth(1.75, 2.0, np.abs(ang))) * (1 - smooth(0.18, 0.32, d[:, 1])))
# texture: a very slight wave along the way the hair lies (from the part across the forehead toward the right temple,
# down at the back), as soft lighter and darker strands in the colour only: a ripple in the surface met the cel shading's
# steps as zigzag ridges
flow = 11 * ang - 7 * d[:, 1]
part = np.exp(-(((ang - 0.62) / 0.05) ** 2)) * smooth(0.45, 0.6, d[:, 1]) * smooth(-0.2, 0.3, d[:, 2])  # a shallow groove from the hairline back over the crown
volume = (1 + 0.04 * smooth(0.3, 0.9, d[:, 1])  # a full crown
          + 0.07 * np.exp(-((ang + 0.15) ** 2) / 0.30 - ((d[:, 1] - 0.62) ** 2) / 0.035) * smooth(0.0, 0.4, d[:, 2])  # the roll over the forehead
          + 0.05 * front * np.exp(-(((d[:, 1] - fringe - 0.07) / 0.06) ** 2))  # a rounded lip along the edge: the hair has thickness
          - 0.03 * part)
Vh = HC + (Vh - HC) * ((0.86 + 0.14 * show) * np.where(show > 0.5, volume, 1.0))[:, None]  # hidden parts end up inside the head (0.86 x 1.1 < 1)
wave = 0.6 * np.sin(2.3 * flow + 0.7) + 0.4 * np.sin(3.9 * flow + 2.0)  # two gentle frequencies: strands of uneven width
streak = (0.5 + 0.5 * wave) * 0.18  # 0 = the base brown, at most 18% of the way to the lighter strand colour
parts.append(Part(Vh, Fh, HAIR[None] * (1 - streak[:, None]) + HAIR_LIGHT[None] * streak[:, None], ["head"]))


def surf(x, y):
    """Point and outward normal on the head ellipsoid in front of offsets (x, y) from its centre."""
    z = HR[2] * np.sqrt(np.clip(1 - (x / HR[0]) ** 2 - (y / HR[1]) ** 2, 1e-4, 1))
    n = np.stack([x / HR[0] ** 2, y / HR[1] ** 2, z / HR[2] ** 2], -1)
    return HC + np.stack([x, y, z], -1), n / np.linalg.norm(n, axis=-1, keepdims=True)


def decal(V, cx, cy):
    """Map a blob built in face coordinates (u across, v up, w out of the skin) onto the head around (cx, cy)."""
    p, n = surf(HEAD * (cx + V[:, 0]), HEAD * (cy + V[:, 1]))  # features are laid out for a head of size 1
    return p + n * V[:, 2:3] * HEAD


Vn, Fn = ellipsoid(np.array([0, 0, 0.004]), np.array([0.0105, 0.0085, 0.0095]))  # dot nose: the face's most forward point (rocketbox noseTip)
parts.append(Part(decal(Vn, 0, -0.035), Fn, NOSE, ["head"]))
EYE_X, EYE_Y = 0.047, -0.022  # big eyes set low and wide: the baby-face proportions that read as cute


def build_face(p: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Face features for shape weights p (ARKit names, 0-1): the same topology for any p, so differences are morphs."""
    g = lambda k: p.get(k, 0.0)  # noqa: E731
    out = []
    for sx, side in ((1, "Left"), (-1, "Right")):  # the character's left is +x
        blink, wide, down, cheek = g(f"eyeBlink{side}"), g(f"eyeWide{side}"), g(f"eyeLookDown{side}"), g(f"cheekSquint{side}")
        squint = min(1.0, g(f"eyeSquint{side}") + 0.45 * cheek)
        # eye: a solid oval, flush with the skin; closing flattens it to a dash, a squint raises its lower edge
        V, F = ellipsoid(np.zeros(3), np.ones(3), seg=32, rings=20)
        ru, rv, lo = 0.0168 * (1 + 0.15 * wide), 0.0215 * (1 + 0.15 * wide), 1 - 0.55 * squint
        v = V[:, 1] * rv * np.where(V[:, 1] < 0, lo, 1.0) * (1 - 0.9 * blink) - 0.003 * down - 0.004 * blink
        out.append((decal(np.stack([V[:, 0] * ru, v, 0.0003 + V[:, 2] * 0.0045], -1), sx * EYE_X, EYE_Y), F, EYE))
        # highlights, upper outer (the same side on both eyes, as one light); they sink into the eye as it closes
        V, F = ellipsoid(np.zeros(3), np.ones(3), seg=12, rings=8)
        for hu, hv0, hr in ((0.0052, 0.0078, 0.0050), (-0.0046, -0.0082, 0.0020)):
            hv = hv0 * (1 - blink) * (1 - (0.5 if hv0 > 0 else 0.9) * squint) - 0.003 * down
            out.append((decal(np.stack([hu + V[:, 0] * hr, hv + V[:, 1] * hr * (1 - 0.8 * blink),
                                        0.0038 - 0.006 * blink + V[:, 2] * 0.0025], -1), sx * EYE_X, EYE_Y), F, SHINE))
        # brow: a short, soft arch; raised for a yes/no question, lowered and drawn in toward the nose for a wh-question
        inner = 0.010 * g("browInnerUp") - 0.008 * g(f"browDown{side}")
        outer = 0.003 * g("browInnerUp") + 0.011 * g(f"browOuterUp{side}") - 0.003 * g(f"browDown{side}")
        t = np.linspace(0, 1, 8)
        us = sx * (0.029 + 0.025 * t)
        vs = 0.045 + 0.005 * np.sin(np.pi * t) + 0.003 * t + inner * (1 - t) + outer * t - 0.004 * g(f"browDown{side}")
        cs = [np.array([u, vv, 0.0]) for u, vv in zip(us, vs)]
        r = 0.0032 * np.sin(np.pi * (0.15 + 0.7 * t)) + 0.0010
        V, F = loft(cs, [Y] * 8, [Z] * 8, r, r * 0.6, seg=12)
        V[-2:] = [cs[0] - np.array([sx * 0.003, 0, 0]), cs[-1] + np.array([sx * 0.003, 0, 0])]  # rounded ends
        out.append((decal(V + np.array([0, 0, 0.0012]), 0, EYE_Y), F, BROW))
        V, F = ellipsoid(np.zeros(3), np.ones(3), seg=20, rings=10)  # cheek blush
        out.append((decal(np.stack([V[:, 0] * 0.021, V[:, 1] * 0.013 + 0.006 * cheek, 0.0004 + V[:, 2] * 0.0012], -1), sx * 0.075, -0.054), F, BLUSH))
    # mouth: a small curved bar, smiling a little at rest; open, its lower edge drops into a D
    t = np.linspace(-1, 1, 15)
    smile = np.where(t > 0, g("mouthSmileLeft") - g("mouthFrownLeft"), g("mouthSmileRight") - g("mouthFrownRight"))
    vs = 0.0035 * t ** 2 + 0.009 * smile * t ** 2 - 0.002 * np.abs(smile)
    phi = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    open_ = g("jawOpen") * 0.016 * (1 - t ** 2)
    rings = []
    for x, y, o in zip(t, vs, open_):  # cross-sections in (v, w): 2.4 mm up, deeper down when open, thinner at the corners
        rv = np.where(np.sin(phi) > 0, 0.0024, 0.0024 + o) * (0.55 + 0.45 * np.sqrt(1 - x ** 2))
        rings.append(np.stack([np.full_like(phi, 0.016 * x), y + np.sin(phi) * rv, 0.0008 + np.cos(phi) * 0.0016], -1))
    ends = [np.array([-0.0165, vs[0], 0.0008]), np.array([0.0165, vs[-1], 0.0008])]
    _, F = loft([np.zeros(3)] * 15, [Y] * 15, [Z] * 15, [1] * 15, [1] * 15, seg=14)
    out.append((decal(np.vstack(rings + [e[None] for e in ends]), 0, -0.064), F, MOUTH))
    return out


MORPHS = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekSquintLeft", "cheekSquintRight",
          "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft",
          "eyeWideRight", "jawOpen", "mouthFrownLeft", "mouthFrownRight", "mouthSmileLeft", "mouthSmileRight"]
base_face = build_face({})
face = [Part(V, F, c, ["head"]) for V, F, c in base_face]
deltas = [np.vstack([V for V, _, _ in build_face({m: 1.0})]) - np.vstack([V for V, _, _ in base_face]) for m in MORPHS]


# ---- skinning, normals, buffers ----
def seg_dist(V, a, b):
    ab = b - a
    t = np.clip((V - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(V - (a + t[:, None] * ab), axis=1)


def merge(ps: list[Part]):
    """One vertex array for the parts: weights from the distance to each candidate bone (1/d^sharp, top 4), area-weighted normals."""
    Vs, Fs, Cs, Js, Ws, Ns, base = [], [], [], [], [], [], 0
    for p in ps:
        tri = p.V[p.F] - p.V.mean(0)
        if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:  # mirrored parts come out inside-out
            p.F = p.F[:, ::-1]
        D = np.stack([seg_dist(p.V, P[b], tail(b)) for b in p.bones], 1)
        w = np.array([p.bias.get(b, 1.0) for b in p.bones]) / (D + 1e-3) ** p.sharp
        top = np.argsort(-w, 1)[:, :4]
        wt = np.take_along_axis(w, top, 1)
        wt /= wt.sum(1, keepdims=True)
        wt[wt < 0.01] = 0
        wt /= wt.sum(1, keepdims=True)
        j = np.array([J[b] for b in p.bones])[top]
        if top.shape[1] < 4:
            j = np.hstack([j, np.zeros((len(j), 4 - top.shape[1]), int)])
            wt = np.hstack([wt, np.zeros((len(wt), 4 - wt.shape[1]))])
        n = np.zeros_like(p.V)
        tri = p.V[p.F]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for k in range(3):
            np.add.at(n, p.F[:, k], fn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        Vs.append(p.V); Fs.append(p.F + base); Cs.append(p.color if p.color.ndim == 2 else np.tile(p.color, (len(p.V), 1))); Js.append(j); Ws.append(wt); Ns.append(n)  # noqa: E702
        base += len(p.V)
    return [np.vstack(x) for x in (Vs, Fs, Cs, Js, Ws, Ns)]


out = G.GLTF2(asset=G.Asset(version="2.0", generator="lingual build_toon_avatar.py"))
buf = bytearray()


def add(arr, ctype, typ, target=None, minmax=False):
    while len(buf) % 4:
        buf.append(0)
    arr = np.ascontiguousarray(arr)
    out.bufferViews.append(G.BufferView(buffer=0, byteOffset=len(buf), byteLength=arr.nbytes, target=target))
    buf.extend(arr.tobytes())
    acc = G.Accessor(bufferView=len(out.bufferViews) - 1, componentType=ctype, count=len(arr), type=typ)
    if minmax:
        acc.min, acc.max = arr.min(0).tolist(), arr.max(0).tolist()
    out.accessors.append(acc)
    return len(out.accessors) - 1


def mesh(name: str, ps: list[Part], targets: list[np.ndarray] | None = None) -> int:
    V, F, C, Jn, W, N = merge(ps)
    attrs = G.Attributes(POSITION=add(V.astype(np.float32), G.FLOAT, "VEC3", G.ARRAY_BUFFER, True),
                         NORMAL=add(N.astype(np.float32), G.FLOAT, "VEC3", G.ARRAY_BUFFER),
                         COLOR_0=add(C.astype(np.float32), G.FLOAT, "VEC3", G.ARRAY_BUFFER),
                         JOINTS_0=add(Jn.astype(np.uint16), G.UNSIGNED_SHORT, "VEC4", G.ARRAY_BUFFER),
                         WEIGHTS_0=add(W.astype(np.float32), G.FLOAT, "VEC4", G.ARRAY_BUFFER))
    prim = G.Primitive(attributes=attrs, indices=add(F.astype(np.uint32).reshape(-1), G.UNSIGNED_INT, "SCALAR", G.ELEMENT_ARRAY_BUFFER), material=0)
    if targets:
        prim.targets = [{"POSITION": add(t.astype(np.float32), G.FLOAT, "VEC3", G.ARRAY_BUFFER, True)} for t in targets]
    out.meshes.append(G.Mesh(name=name, primitives=[prim], weights=[0.0] * len(targets) if targets else None,
                             extras={"targetNames": MORPHS} if targets else None))
    print(f"{name}: {len(V)} vertices, {len(F)} triangles")
    return len(out.meshes) - 1


out.materials = [G.Material(name="toon", pbrMetallicRoughness=G.PbrMetallicRoughness(baseColorFactor=[1, 1, 1, 1], metallicFactor=0, roughnessFactor=0.8))]
mb, mf = mesh("toon_body", parts), mesh("toon_face", face, deltas)
# skeleton nodes copied as they are (the bones hang below "Human.rig"'s Root); the two meshes sit at the rig's origin
rig = next(i for i, n in enumerate(src.nodes) if n.name == "Human.rig")
keep = sorted(set(skin0.joints) | {i for i, n in enumerate(src.nodes) if n.mesh is None and i != rig})
remap = {o: k for k, o in enumerate(keep)}
for o in keep:
    n = src.nodes[o]
    out.nodes.append(G.Node(name=n.name, translation=n.translation, rotation=n.rotation, scale=n.scale, matrix=n.matrix,
                            children=[remap[c] for c in (n.children or []) if c in remap] or None))
top = [remap[c] for c in src.nodes[rig].children if c in remap]
nb, nf = len(out.nodes), len(out.nodes) + 1
out.nodes += [G.Node(name="toon_body", mesh=mb, skin=0), G.Node(name="toon_face", mesh=mf, skin=0)]
out.nodes.append(G.Node(name="Human.rig", children=top + [nb, nf]))
out.scenes = [G.Scene(nodes=[len(out.nodes) - 1])]
out.scene = 0
out.skins = [G.Skin(joints=[remap[j] for j in skin0.joints], inverseBindMatrices=add(IBM.astype(np.float32), G.FLOAT, "MAT4"))]
out.buffers = [G.Buffer(byteLength=len(buf))]
out.set_binary_blob(bytes(buf))
OUT.mkdir(parents=True, exist_ok=True)
out.save_binary(str(OUT / "avatar.glb"))
(OUT / "LICENSE.txt").write_text("Generated by app/scripts/build_toon_avatar.py (geometry, colours, face shapes): MIT, as the repository.\n"
                                 "Skeleton (bone names and bind pose) from the MPFB character in ../mpfb/ (MakeHuman for Blender, CC0).\n")
print(f"wrote {OUT / 'avatar.glb'} ({(OUT / 'avatar.glb').stat().st_size // 1024} KB)")
