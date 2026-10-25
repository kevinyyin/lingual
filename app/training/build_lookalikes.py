"""Look-alike keys for the sign model's labels, from ASL-LEX phonology: gloss -> [handshape, major location].

Two signs with the same ASL-LEX 2.0 Handshape and MajorLocation differ only in movement, orientation or hand count
(GOOD/THANK-YOU, MINUTE/HOUR, MY/PLEASE): the parameters the sign model reads least surely. server/session.py sends a
confident read whose look-alike is close behind it to the picker. Writes vocab/lookalikes.json (committed), so the
server never reads ASL-LEX (data/, gitignored) at run time.

Run: uv run python -m training.build_lookalikes [--model models/signs.pt]
"""

import argparse
import json

from server.session import LOOKALIKES
from training.touch_from_asllex import lexicon

ALIASES = {"I": "ME"}  # ASL-LEX files I (index to chest) under ME


def keys(labels: list[str]) -> dict[str, list[str]]:
    lex = lexicon(list(labels) + list(ALIASES.values()))
    out = {}
    for g in labels:
        e = lex.get(g) or lex.get(ALIASES.get(g, "")) or {}
        if e.get("Handshape.2.0") and e.get("MajorLocation.2.0"):
            out[g] = [e["Handshape.2.0"], e["MajorLocation.2.0"]]
    return out


def main() -> None:
    from server.classifier import DEFAULT_MODEL, SignPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    labels = SignPredictor(args.model).labels
    out = keys(labels)
    rows = [f"  {json.dumps(g)}: {json.dumps(k)}" for g, k in sorted(out.items())]  # one sign per line: readable diffs
    LOOKALIKES.write_text("{\n" + ",\n".join(rows) + "\n}\n", encoding="utf-8")
    print(f"{len(out)}/{len(labels)} labels keyed, missing: {sorted(set(labels) - set(out))} -> {LOOKALIKES}")


if __name__ == "__main__":
    main()
