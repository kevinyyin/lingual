"""Does a clip look like its gloss's ASL-LEX entry? (step 1b) Four checks, each agree / disagree / not applicable:

- hands: ASL-LEX one-handed vs two-handed (any other SignType) against how much the second hand is raised
- location: MajorLocation Head against the hand coming near the face (Neutral/Hand/Body: not at the face throughout)
- fingers: SelectedFingers when Flexion is FullyOpen/Flat (those extended, others not) or FullyClosed (none extended),
  against each finger's measured extension; other flexions (curved, bent, stacked...) are not judged
- movement: Movement "None" against little wrist travel, any other against some

Score = agreeing / applicable. A low score marks a clip for the agents to look at; it is never a verdict on its own
(variants, a non-dominant base hand, 2D foreshortening of fingers all bend these measures).
"""
from training.touch_from_asllex import lexicon

EXT = 1.55  # tip-to-wrist / knuckle-to-wrist above this: the finger is extended


def check(gloss: str, feats: dict, lex: dict | None = None) -> dict:
    e = (lex if lex is not None else lexicon([gloss])).get(gloss)
    if not e or "ext_i" not in feats:
        return {"gloss": gloss, "asllex": bool(e), "score": None, "checks": {}}
    out: dict[str, bool | None] = {}
    st = e.get("SignType.2.0")
    if st:
        two = feats.get("other_raised_frac", 0) > 0.4
        out["hands"] = (st != "OneHanded") == two
    loc = e.get("MajorLocation.2.0")
    if loc in ("Head",):
        out["location"] = feats.get("near_face_frac", 0) > 0.15 or feats.get("wrist_nose_min", 9) < 0.8
    elif loc in ("Neutral", "Hand", "Body"):
        out["location"] = feats.get("near_face_frac", 0) < 0.6
    sel, flex = e.get("SelectedFingers.2.0") or "", e.get("Flexion.2.0")
    if flex in ("FullyOpen", "Flat") and sel:
        want = {f for f in "imrp" if f in sel}
        got = {f for f in "imrp" if feats[f"ext_{f}"] > EXT}
        out["fingers"] = len(want ^ got) <= 1  # one finger off is tracking noise or 2D foreshortening
    elif flex == "FullyClosed":
        out["fingers"] = sum(feats[f"ext_{f}"] > EXT for f in "imrp") <= 1
    mv = e.get("Movement.2.0")
    if mv:
        moved = feats.get("path_sw", 0) > 0.4
        out["movement"] = moved if mv != "None" else feats.get("path_sw", 0) < 1.5
    judged = [v for v in out.values() if v is not None]
    return {"gloss": gloss, "asllex": True, "score": sum(judged) / len(judged) if judged else None, "checks": out,
            "entry": {k: e.get(k) for k in ("EntryID", "SignType.2.0", "MajorLocation.2.0", "MinorLocation.2.0",
                                             "SelectedFingers.2.0", "Flexion.2.0", "Movement.2.0", "Handshape.2.0")}}
