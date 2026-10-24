"""Round-trip legibility: render each sign alone on the avatar, run MediaPipe on the render, ask the ASL Citizen recognizer.

A weak signal for how legible the avatar is, for before/after comparison. Needs web :5173 and backend :8000 running.
Run (from app/): uv run --with playwright python scripts/roundtrip.py --label NAME [--model NAME (default toon; Male_Adult_11 or mpfb)] [--n 150]
Writes data/eval/roundtrip/NAME/<GLOSS>.mp4 (not committed) and reports/motion/roundtrip_NAME.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = "http://localhost:5173"


def score(rows: list[tuple[str, list[str]]]) -> dict[str, float]:
    n = len(rows)
    return {**{f"top{k}": sum(g in p[:k] for g, p in rows) / n if n else 0.0 for k in (1, 3, 5)}, "n": n}


def main() -> None:
    from playwright.sync_api import sync_playwright

    from motion_bench import write_video
    from server.citizen_classifier import CitizenSignPredictor
    from server.vocab import load_vocab
    from training.landmarks import video_to_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    vocab = load_vocab()
    glosses = sorted(s.gloss for s in vocab.all if s.curated and s.avatar_ok and (ROOT / "web/public/signs" / f"{s.gloss}.json").exists())[:a.n]
    if len(glosses) < a.n:
        sys.exit(f"only {len(glosses)} curated, avatar_ok signs with clips found; --n {a.n} needs more")
    predictor = CitizenSignPredictor(vocab, restrict_to_vocab=True)
    vids = ROOT / "data" / "eval" / "roundtrip" / a.label
    url = f"{BASE}/?mode=bench" + (f"&model={a.model}" if a.model else "")
    rows, preds, errors = [], {}, {}
    out = ROOT / "reports" / "motion"
    out.mkdir(parents=True, exist_ok=True)

    def save() -> None:  # after every sign, so a crash keeps what was scored
        r = {"label": a.label, "model": a.model or "toon", **score(rows), "glosses": glosses, "errors": errors, "predictions": preds}
        (out / f"roundtrip_{a.label}.json").write_text(json.dumps(r, indent=2) + "\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, args=["--window-size=700,800"])
        page = browser.new_page(viewport={"width": 640, "height": 720})
        page.goto(url)
        page.wait_for_function("() => window.bench", timeout=120000)
        for g in glosses:
            try:
                shots = page.evaluate("([i, o]) => window.bench.run(i, o)", [[g], {"fps": 30, "capture": True}])["images"]
                mp4 = vids / f"{g}.mp4"
                write_video(shots, mp4, 30)
                top = [p for p, _ in predictor.topk(video_to_frames(mp4), 5)]
            except Exception as e:  # a failed sign counts as a miss so n stays comparable
                top, errors[g] = [], f"{type(e).__name__}: {e}"
                print(g, "ERROR", errors[g], flush=True)
            else:
                print(g, top[:3], flush=True)
            rows.append((g, top))
            preds[g] = top
            save()
        browser.close()
    print(score(rows), "errors:", len(errors))

if __name__ == "__main__":
    main()
