"""Clean-up of a clip's hand track (21 hand world landmarks per frame, or None where MediaPipe lost the hand).

Single-video MediaPipe hands jitter by 4-15% of a palm length per frame, read curled fingers as half-straight, and
drop out for a few frames at a time. Here each frame is split into the palm's orientation and the handshape inside
the palm's frame, then:
  1. depth and knuckle bend corrected (as for the fingerspelling letters, training.letters_from_kaggle);
  2. gaps bridged by interpolation (not frozen and snapped back);
  3. orientation smoothed lightly; the handshape held: an ASL sign is mostly one or two handshapes, so each stretch
     where the handshape stays steady takes its most typical real frame (the medoid) for the whole stretch, with a
     blend between consecutive held shapes. Stretches that keep changing (wiggling fingers) keep their motion.

Opt-in (clips_from_wlasl --clean): it steadies the hands and closes fists, but on the round-trip test (150 signs
rendered on the avatar, read back by the ASL Citizen recognizer) clips without it read better: top-5 0.280 against
0.220 (0.240 without the knuckle step). The recognizer learned real signers through MediaPipe, noise and under-curl
included, so a cleaner hand reads as less typical. Worth re-testing with a human signer's judgement.
"""

import numpy as np

# finger chains (MediaPipe hand indices) and their bone lengths as fractions of wrist -> middle knuckle (adult averages)
FINGER_BONES = [([1, 2, 3, 4], [0.38, 0.30, 0.26]), ([5, 6, 7, 8], [0.45, 0.26, 0.22]), ([9, 10, 11, 12], [0.48, 0.30, 0.24]),
                ([13, 14, 15, 16], [0.44, 0.28, 0.23]), ([17, 18, 19, 20], [0.34, 0.21, 0.20])]
KNUCKLE_FROM, KNUCKLE_GAIN, KNUCKLE_MAX = 35.0, 2.6, 100.0  # degrees: knuckle bend correction (fix_knuckles)
SIGMA_S = 0.035  # orientation smoothing, seconds (Gaussian)
MAX_SEGMENTS = 6  # handshape stretches per clip (repeated open-close signs such as MILK need several)
MIN_SEGMENT_S = 0.1
SPLIT_COST = 3.0  # an extra stretch must explain this many frames' worth of noise
BLEND_S = 0.1  # handshape change between stretches, centred on the boundary
KEEP_MOTION = 0.3  # share of each frame's (smoothed) deviation from its stretch kept on top of the held shape


def fix_depth(hand: np.ndarray) -> np.ndarray:
    """MediaPipe's depth is compressed: a finger curled toward the camera (a fist facing it: A, S, E, M, N, T) came out
    nearly straight. Each finger bone's depth is re-derived from its known length and its reliable image-plane length,
    keeping MediaPipe's depth direction."""
    h = hand.copy()
    palm = max(np.linalg.norm(hand[9] - hand[0]), 1e-6)
    for chain, ratios in FINGER_BONES:
        for (a, b), r in zip(zip(chain, chain[1:]), ratios):
            d = hand[b] - hand[a]
            dz = np.sqrt(max(0.0, (r * palm) ** 2 - d[0] ** 2 - d[1] ** 2))
            h[b, 2] = h[a, 2] + (np.sign(d[2]) or -1.0) * max(dz, abs(d[2]))  # never flatter than MediaPipe saw
    return h


def fix_knuckles(h: np.ndarray) -> np.ndarray:
    """MediaPipe also under-reads knuckle bend on fingers curled toward the camera: Kaggle fists (A, S) measured 42-81
    degrees at the knuckles where a fist is ~90 (the posed avatar matched the data exactly, so this is the input).
    A clearly bent knuckle (> KNUCKLE_FROM) has its bend beyond that amplified, the whole finger turning about it."""
    h = h.copy()
    palm = h[9] - h[0]
    for chain, _ in FINGER_BONES[1:]:  # not the thumb
        k = chain[0]
        prox = h[chain[1]] - h[k]
        theta = np.degrees(np.arccos(np.clip(prox.dot(palm) / (np.linalg.norm(prox) * np.linalg.norm(palm) + 1e-9), -1, 1)))
        if theta <= KNUCKLE_FROM:
            continue
        axis = np.cross(palm, prox)
        if np.linalg.norm(axis) < 1e-9:
            continue
        axis /= np.linalg.norm(axis)
        extra = np.radians(min(KNUCKLE_MAX, KNUCKLE_FROM + (theta - KNUCKLE_FROM) * KNUCKLE_GAIN) - theta)
        c, s = np.cos(extra), np.sin(extra)  # Rodrigues rotation of the finger about the knuckle
        for i in chain[1:]:
            v = h[i] - h[k]
            h[i] = h[k] + v * c + np.cross(axis, v) * s + axis * axis.dot(v) * (1 - c)
    return h


def palm_frame(h: np.ndarray) -> np.ndarray:
    """3x3 rotation whose columns are the palm's axes: wrist -> middle knuckle, pinky -> index knuckle, palm normal."""
    f = h[9] - h[0]
    f = f / max(np.linalg.norm(f), 1e-9)
    s = h[5] - h[17]
    s = s - f * s.dot(f)
    s = s / max(np.linalg.norm(s), 1e-9)
    return np.stack([f, s, np.cross(f, s)], 1)


def _orthonormal(m: np.ndarray) -> np.ndarray:
    """Nearest palm frame to a blended 3x3 (its first two columns re-orthonormalized the way palm_frame does)."""
    f = m[:, 0] / max(np.linalg.norm(m[:, 0]), 1e-9)
    s = m[:, 1] - f * m[:, 1].dot(f)
    s = s / max(np.linalg.norm(s), 1e-9)
    return np.stack([f, s, np.cross(f, s)], 1)


def _interp(t: np.ndarray, ok: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Values at frames t from the frames where ok, linear inside gaps and held at the ends."""
    flat = x[ok].reshape(ok.sum(), -1)
    out = np.stack([np.interp(t, t[ok], flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((len(t),) + x.shape[1:])


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing over time (axis 0), sigma in frames, ends held."""
    if sigma < 0.3:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    flat = x.reshape(len(x), -1)
    pad = np.concatenate([flat[:1].repeat(r, 0), flat, flat[-1:].repeat(r, 0)])
    out = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def _medoid(xs: np.ndarray) -> int:
    flat = xs.reshape(len(xs), -1)
    return int(np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum(1).argmin())


def segments(shape: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """The clip split into contiguous [start, end) stretches of one handshape each: the split (up to MAX_SEGMENTS) that
    best explains the handshape as piecewise constant, each extra stretch paying SPLIT_COST x the frame noise.
    Frame-to-frame noise is as fast as a real handshape change, so speed thresholds could not tell them apart; a real
    change shifts the handshape for the rest of a stretch, noise does not."""
    n = len(shape)
    x = shape[:, 1:].reshape(n, -1)
    min_len = max(2, int(round(MIN_SEGMENT_S * fps)))
    if n < 2 * min_len:
        return [(0, n)]
    noise = float(np.median((np.diff(x, axis=0) ** 2).sum(1)) / 2)  # per-frame noise variance (summed over dims)
    s1 = np.r_[np.zeros((1, x.shape[1])), np.cumsum(x, 0)]
    s2 = np.r_[0.0, np.cumsum((x ** 2).sum(1))]

    def cost(a: np.ndarray, b: int) -> np.ndarray:  # squared deviation from the stretch's mean, stretches [a, b)
        return s2[b] - s2[a] - ((s1[b] - s1[a]) ** 2).sum(-1) / (b - a)

    k_max = min(MAX_SEGMENTS, n // min_len)
    best = np.full((k_max + 1, n + 1), np.inf)  # best[k, b]: frames [0, b) in k stretches
    back = np.zeros((k_max + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, k_max + 1):
        for b in range(k * min_len, n + 1):
            a = np.arange((k - 1) * min_len, b - min_len + 1)
            c = best[k - 1, a] + cost(a, b)
            i = int(np.argmin(c))
            best[k, b], back[k, b] = c[i], a[i]
    k = int(np.argmin([best[k, n] + k * SPLIT_COST * noise * x.shape[1] for k in range(k_max + 1)]))
    out, b = [], n
    for kk in range(k, 0, -1):
        a = back[kk, b]
        out.append((int(a), int(b)))
        b = a
    return out[::-1]


def clean_track(track: list, fps: float) -> list:
    """A clip's hand track (per frame 21x[x,y,z] or None) cleaned as described above; all None if never seen."""
    ok = np.array([h is not None for h in track])
    if not ok.any():
        return [None] * len(track)
    n = len(track)
    t = np.arange(n, dtype=np.float64)
    raw = np.zeros((n, 21, 3))
    raw[ok] = [fix_knuckles(fix_depth(np.asarray(h, dtype=np.float64))) for h in track if h is not None]
    palm_len = np.linalg.norm(raw[ok, 9] - raw[ok, 0], axis=-1)
    size = float(np.median(palm_len))
    rot = np.zeros((n, 3, 3))
    rot[ok] = [palm_frame(h) for h in raw[ok]]
    shape = np.zeros((n, 21, 3))  # points in the palm's frame, palm lengths, wrist at the origin
    shape[ok] = np.einsum("tji,tkj->tki", rot[ok], raw[ok] - raw[ok, :1]) / palm_len[:, None, None]
    wrist = _interp(t, ok, raw[:, 0])
    shape = _interp(t, ok, shape)
    rot = np.stack([_orthonormal(m) for m in _gauss(_interp(t, ok, rot), SIGMA_S * fps)])

    held = np.zeros_like(shape)
    motion = _gauss(shape, SIGMA_S * fps)
    segs = segments(shape, fps)
    for a, b in segs:
        real = [i for i in range(a, b) if ok[i]] or list(range(a, b))
        held[a:b] = shape[real[_medoid(shape[real])]] + KEEP_MOTION * (motion[a:b] - motion[a:b].mean(0))
    half = BLEND_S * fps / 2
    for (_, b), _ in zip(segs, segs[1:]):  # ramp across each boundary
        lo, hi = max(0, int(b - half)), min(n - 1, int(b + half))
        for i in range(lo + 1, hi):
            w = (i - lo) / (hi - lo)
            held[i] = held[lo] * (1 - w) + held[hi] * w
    return (wrist[:, None] + np.einsum("tij,tkj->tki", rot, held * size)).tolist()
