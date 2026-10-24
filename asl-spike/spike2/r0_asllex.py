"""R0: link each of the 143 model classes to ASL-LEX 2.0 via ASL Citizen's ASL-LEX Code column.

ASL-LEX 2.0 source: asl-lex.org/visualization/data/sign_props.json (the OSF project osf.io/zpha4 was down for
maintenance on 2026-10-16). CC BY-NC 4.0. A class can contain several ASL Citizen glosses (variants such as
WANT1/WANT2), each with its own code: the primary entry is the one with the most ASL Citizen videos; the other
entries are listed as variants, and a class whose variants disagree on sign type is flagged as ambiguous.
"""
from __future__ import annotations

import json
from collections import Counter

from common import ROOT, citizen_class_map, citizen_rows, jdump, model_classes

FIELDS = {"sign_type": "SignType.2.0", "handshape": "Handshape.2.0", "nondominant_handshape": "NonDominantHandshape.2.0",
          "flexion": "Flexion.2.0", "flexion_change": "FlexionChange.2.0", "selected_fingers": "SelectedFingers.2.0",
          "spread": "Spread.2.0", "thumb_position": "ThumbPosition.2.0", "major_location": "MajorLocation.2.0",
          "minor_location": "MinorLocation.2.0", "path_movement": "Movement.2.0", "repeated_movement": "RepeatedMovement.2.0",
          "sign_onset_ms": "SignOnset(ms)", "sign_offset_ms": "SignOffset(ms)", "sign_duration_ms": "SignDuration(ms)",
          "lexical_class": "LexicalClass", "handshape_m2": "HandshapeM2.2.0"}


def main():
    lex = {e["Code"]: e for e in json.load(open(ROOT / "data/asl-lex/sign_props.json", encoding="utf-8"))}
    cmap = citizen_class_map()
    rows = citizen_rows()
    rows = rows[rows["Gloss"].isin(cmap)]
    classes = model_classes()
    out, ambiguous, missing = {}, [], []
    for c in classes:
        sub = rows[rows["Gloss"].map(cmap) == c]
        sub = sub.assign(code=sub["code"].fillna("nan").astype(str))
        codes = Counter(sub["code"])
        entries = []
        by_entry = {x["EntryID"]: x for x in lex.values()}
        for code, n in codes.most_common():
            e = lex.get(code)
            if e is None and code == "nan":
                # no ASL-LEX code in ASL Citizen's metadata: match the gloss name to an ASL-LEX EntryID
                for g in sorted(set(sub[sub["code"] == "nan"]["Gloss"])):
                    import re as _re
                    m = _re.match(r"^([A-Z]+?)(\d*)$", g)
                    cand = [m.group(1).lower() + ("" if m.group(2) in ("", "1") else "_" + m.group(2))] if m else []
                    hit = next((by_entry[c_] for c_ in cand if c_ in by_entry), None)
                    if hit:
                        k = sum(sub["Gloss"] == g)
                        entries.append({"code": hit["Code"], "entry_id": hit["EntryID"], "citizen_glosses": [g], "videos": int(k),
                                        "matched_by": "gloss name (no code in ASL Citizen)", **{k2: hit.get(v) for k2, v in FIELDS.items()}})
                    else:
                        missing.append({"class": c, "citizen_gloss": g, "reason": "no code and no EntryID match"})
                continue
            if e is None:
                missing.append({"class": c, "code": code, "videos": n})
                continue
            entries.append({"code": code, "entry_id": e["EntryID"], "citizen_glosses": sorted(set(sub[sub["code"] == code]["Gloss"])),
                            "videos": n, **{k: e.get(v) for k, v in FIELDS.items()}})
        if not entries:
            out[c] = {"primary": None, "variants": []}
            continue
        types = {x["sign_type"] for x in entries}
        if len(types) > 1:
            ambiguous.append({"class": c, "variants": [(x["entry_id"], x["sign_type"], x["videos"]) for x in entries]})
        out[c] = {"primary": entries[0], "variants": entries[1:], "sign_type_ambiguous": len(types) > 1}
    summary = {"n_classes": len(classes), "linked": sum(1 for v in out.values() if v["primary"]),
               "sign_type_counts": dict(Counter(v["primary"]["sign_type"] for v in out.values() if v["primary"])),
               "handshapes_needed_dominant": len({v["primary"]["handshape"] for v in out.values() if v["primary"]}),
               "flexion_change_signs": sum(1 for v in out.values() if v["primary"] and v["primary"]["flexion_change"]),
               "path_movement_none": sum(1 for v in out.values() if v["primary"] and v["primary"]["path_movement"] == "None"),
               "ambiguous_sign_type": ambiguous, "codes_missing_in_asllex": missing,
               "source": "asl-lex.org/visualization/data/sign_props.json (ASL-LEX 2.0, CC BY-NC 4.0); OSF was down"}
    jdump({"summary": summary, "classes": out}, "asllex_links.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("ambiguous_sign_type",)}, indent=1)[:1500])
    print("ambiguous:", [(a["class"], a["variants"]) for a in ambiguous])


if __name__ == "__main__":
    main()
