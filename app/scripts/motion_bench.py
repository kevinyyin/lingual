"""Measure the avatar's motion on fixed test items, for before/after comparison of the arm solver.

Drives ?mode=bench in the running app (web :5173, backend :8000) with Playwright; writes reports/motion/NAME.json and .md.
Metrics (web/src/avatar/metrics.ts): spike = an arm bone turning > 25 deg between 60 fps frames; backward elbow = flexion
< -5 deg; twist out = forearm twist > 90 deg; penetration in metres; contact error = hand probe to the mapped touched place (mean over frames, and the per-item minimum).
CONTINUOUS plays all SENTENCES back to back with no reset (excluded from TOTAL).

Hand columns (handModel.ts): handOutFrames/handOutJoints = frames / joint-frames with a finger joint > 1 deg past its
anatomical range, maxHandOutDeg the worst excess; maxFingerSelf / maxHandHand / maxFingerBody = deepest overlap (m) between
fingers of one hand, between the hands, and of a finger joint in the body; heldHands = hand-frames rejected as mirrored.

Run: uv run --with playwright python scripts/motion_bench.py --label NAME [--model NAME; default toon, Male_Adult_11 or mpfb] [--capture]
     [--no-hand-limits: before, the raw finger copy] [--all-signs: every clip in web/public/signs on its own]
(first time: uv run --with playwright playwright install chromium). --capture also writes data/eval/motion/NAME/<item>.mp4.
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:5173"
SIGNS = "GRANDMOTHER MOTHER FATHER MY PLEASE HELP MEET KING REMEMBER WHERE HELLO THANK-YOU".split()
SENTENCES = ["MY GRANDMOTHER TEACH I COOK", "HELLO NICE MEET YOU", "YOU HELP I FIND MY KEY", "TOMORROW I GO STORE BUY APPLE",
             "LIVE OR DIE WHICH", "I SORRY I UNDERSTAND NOT", "TRAIN LEAVE WHAT TIME", "YOUR CAR NEW", "FATHER MOTHER WORK WHERE",
             "NOW I NEED SLEEP"]
CONTINUOUS = "CONTINUOUS (10 sentences)"  # all SENTENCES back to back on one clock, no reset: the live stage's carried state
FPS = 60
COLUMNS = ["frames", "p95FrameMs", "spikes", "maxStepDeg", "p99StepDeg", "backwardElbow", "twistOut", "maxPenetration",
           "maxArmPenetration", "minArmGap", "contactErrorMean", "contactErrorMin",
           "handOutFrames", "handOutJoints", "maxHandOutDeg", "maxFingerSelf", "maxHandHand", "maxFingerBody", "heldHands",
           "atLimit", "edgeSpeed", "edgeJerk", "innerSpeed", "innerJerk"]


def items(page) -> dict[str, object]:
    out: dict[str, object] = {g: [g] for g in SIGNS}
    out.update({s: s.split() for s in SENTENCES})
    # ASL-LEX codes no contact for GRANDMOTHER, so measure it against the chin: the clip's own nose, 0.07 m down
    nose = page.evaluate("""async () => {
      const c = await (await fetch("/signs/GRANDMOTHER.json")).json();
      const m = [0, 1, 2].map((k) => c.frames.reduce((s, f) => s + f.pose[0][k], 0) / c.frames.length);
      return m;
    }""")
    out["GRANDMOTHER@chin"] = {"glosses": ["GRANDMOTHER"], "at": {"hand": "R", "place": [nose[0], nose[1] + 0.07, nose[2]]}}
    out[CONTINUOUS] = {"sentences": [s.split() for s in SENTENCES]}
    return out


def totals(rows: dict[str, dict]) -> dict:
    r = list(rows.values())
    contact = [x["contactErrorMean"] for x in r if x["contactErrorMean"] is not None]
    return {
        "frames": sum(x["frames"] for x in r), "p95FrameMs": max(x["p95FrameMs"] for x in r),
        "spikes": sum(x["spikes"] for x in r), "maxStepDeg": max(x["maxStepDeg"] for x in r),
        "p99StepDeg": max(x["p99StepDeg"] for x in r), "backwardElbow": sum(x["backwardElbow"] for x in r),
        "twistOut": sum(x["twistOut"] for x in r), "maxPenetration": max(x["maxPenetration"] for x in r),
        "maxArmPenetration": max(x["maxArmPenetration"] for x in r), "minArmGap": min(x["minArmGap"] for x in r),
        "contactErrorMean": sum(contact) / len(contact) if contact else None,
        "contactErrorMin": min((x["contactErrorMin"] for x in r if x.get("contactErrorMin") is not None), default=None),
        **{k: sum(x[k] for x in r) for k in ("handOutFrames", "handOutJoints", "heldHands", "atLimit")},
        **{k: sum(x[k] for x in r) / len(r) for k in ("edgeSpeed", "edgeJerk", "innerSpeed", "innerJerk")},
        **{k: max(x[k] for x in r) for k in ("maxHandOutDeg", "maxFingerSelf", "maxHandHand", "maxFingerBody")},
    }


def cell(v) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def table(rows: dict[str, dict]) -> str:
    lines = ["| item | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
    lines += [f"| {k} | " + " | ".join(cell(v[c]) for c in COLUMNS) + " |" for k, v in rows.items()]
    return "\n".join(lines) + "\n"


def write_video(images: list[str], dst: Path, fps: int = 30) -> None:
    frames = [cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR) for u in images]
    h, w = frames[0].shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def main() -> None:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default=None, help="e.g. Male_Adult_11 (default: toon; mpfb for MPFB)")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--no-hand-limits", action="store_true", help="raw finger copy, without the anatomical hand (before)")
    ap.add_argument("--all-signs", action="store_true", help="every built clip on its own instead of the fixed items")
    ap.add_argument("--channel", default=None, help="msedge: the installed Edge (where Windows blocks Playwright's own Chromium)")
    a = ap.parse_args()
    opts = {"handLimits": not a.no_hand_limits}
    url = f"{BASE}/?mode=bench" + (f"&model={a.model}" if a.model else "")
    run = "([i, o]) => window.bench.run(i, o)"
    results: dict[str, dict] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, channel=a.channel,
                                     args=["--window-size=700,800", *(["--use-angle=d3d11"] if a.channel == "msedge" else [])])
        page = browser.new_page(viewport={"width": 640, "height": 720})
        page.goto(url)
        page.wait_for_function("() => window.bench", timeout=120000)
        todo = {p.stem: [p.stem] for p in sorted((ROOT / "web" / "public" / "signs").glob("*.json")) if not p.stem.startswith("_")}             if a.all_signs else items(page)
        for name, item in todo.items():
            results[name] = page.evaluate(run, [item, {"fps": FPS, "capture": False, **opts}])
            print(name, {c: results[name][c] for c in ("frames", "spikes", "maxStepDeg")}, flush=True)
            if a.capture:  # 30 fps for the side-by-sides; a separate pass, so the metrics above stay at 60
                shots = page.evaluate(run, [item, {"fps": 30, "capture": True, **opts}])["images"]
                write_video(shots, ROOT / "data" / "eval" / "motion" / a.label / (name.replace("@", "_").replace(" ", "_") + ".mp4"))
        browser.close()
    out = ROOT / "reports" / "motion"
    out.mkdir(parents=True, exist_ok=True)
    total = totals({k: v for k, v in results.items() if k != CONTINUOUS})  # it replays the sentences: kept out of TOTAL
    (out / f"{a.label}.json").write_text(json.dumps({"label": a.label, "model": a.model or "toon", "items": results, "totals": total}, indent=2) + "\n")
    (out / f"{a.label}.md").write_text(f"# Motion bench: {a.label} (model: {a.model or 'toon'})\n\n"
                                       + table({**results, "TOTAL": total}))
    print(f"wrote {out / (a.label + '.md')}")


if __name__ == "__main__":
    main()
