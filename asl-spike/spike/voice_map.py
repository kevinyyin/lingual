"""to_voice(grammar, affect, kinematics) -> Cartesia generation settings (S3 output, used by S4/S8).

Grammar decides punctuation only (a question gets "?" so Sonic's prosody rises naturally); it never
chooses the emotion. Emotion comes from the affect layer, restricted to Cartesia's best-supported set
plus "happy" (the spike's four-way label set maps happy -> "happy"; see S4 for whether "content" works better).
Speed follows signing tempo (clamped 0.8-1.3) and volume follows movement amplitude (clamped 0.8-1.5).
"""
from __future__ import annotations

from dataclasses import dataclass

AFFECT_TO_EMOTION = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry"}
# emotion is sent only when the affect layer is confident enough; otherwise Sonic reads the text itself
MIN_INTENSITY = 0.35


@dataclass
class Kinematics:
    tempo_ratio: float = 1.0      # signing tempo vs this signer's (or the population's) median, 1.0 = typical
    amplitude_ratio: float = 1.0  # movement amplitude vs median


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_voice(grammar: dict, affect: dict, kin: Kinematics | None = None, text: str = "",
             text_emotion: str | None = None) -> dict:
    """grammar: {"type": statement|yn|wh, "negation": bool}; affect: {"label": ..., "intensity": 0-1}.
    text_emotion: the LLM's reading of the sentence's own emotion; when it contradicts the affect label
    the emotion is dropped (S4 mismatch test: contradictory tags work poorly)."""
    kin = kin or Kinematics()
    t = text.strip()
    if grammar.get("type") in ("yn", "wh"):
        t = t.rstrip(".!") + ("?" if not t.endswith("?") else "")
    elif t and t[-1] not in ".!?":
        t += "."
    label = affect.get("label", "neutral")
    inten = float(affect.get("intensity", 0.0))
    emotion = AFFECT_TO_EMOTION.get(label, "neutral") if inten >= MIN_INTENSITY else None
    conflict = bool(emotion and text_emotion and text_emotion != "neutral" and emotion != "neutral"
                    and emotion != text_emotion)
    if conflict:
        emotion = None
    # speed/volume: kinematics, nudged by affect intensity for angry/sad
    speed = kin.tempo_ratio
    volume = kin.amplitude_ratio
    if label == "angry":
        volume *= 1 + 0.3 * inten
    if label == "sad":
        speed *= 1 - 0.15 * inten
        volume *= 1 - 0.15 * inten
    if label == "happy":
        speed *= 1 + 0.08 * inten
    out = {"text_punctuation": t, "emotion": emotion, "speed": round(clamp(speed, 0.8, 1.3), 2),
           "volume": round(clamp(volume, 0.8, 1.5), 2), "emotion_dropped_conflict": conflict}
    return out


# canonical settings per emotion at intensity 0.7 and typical kinematics for S4 condition (c), where the
# kinematics a real signer would supply are emulated: angry signing is larger and faster, sad slower and smaller
EMULATED_KINEMATICS = {"neutral": Kinematics(1.0, 1.0), "happy": Kinematics(1.08, 1.1),
                       "sad": Kinematics(0.9, 0.9), "angry": Kinematics(1.12, 1.2)}
