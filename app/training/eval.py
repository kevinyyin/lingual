"""Measure the MVP against the spec's targets. Each subcommand prints a table and PASS/FAIL.

  uv run python -m training.eval signs [--holdout-signer sam]   sign recognition top-1/top-3
  uv run python -m training.eval grid                          question type across the emotion grid
  uv run python -m training.eval spell --holdout-signer sam    fingerspelled names fully correct
  uv run python -m training.eval latency [--last 30]           turn delays from logs/turns.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OURS = ROOT / "data" / "ours"
LOG = ROOT / "logs" / "turns.jsonl"
TARGET_MS = 2000


def latency_stats(lines: list[dict]) -> dict[str, list[float]]:
    """Per direction, ms from the end of the person's turn (or the picker tap) to output starting."""
    out: dict[str, list[float]] = {"sign2speech": [], "speech2sign": []}
    for ln in lines:
        st = ln["stamps"]
        if ln["direction"] == "sign2speech" and "first_audio" in st:
            start = st["picked"] if (ln.get("result") or {}).get("picks") else st["hands_rest"]
            out["sign2speech"].append(st["first_audio"] - start)
        elif ln["direction"] == "speech2sign" and "avatar_start" in st:
            out["speech2sign"].append(st["avatar_start"] - st["speech_end"])
    return out


def grid_accuracy(rows: list[tuple[str, str, str]]) -> tuple[float, dict[str, float]]:
    """rows: (true question type, emotion, predicted type) -> overall accuracy, accuracy per emotion."""
    by: dict[str, list[bool]] = defaultdict(list)
    for truth, emotion, pred in rows:
        by[emotion].append(truth == pred)
    allv = [v for vs in by.values() for v in vs]
    return float(np.mean(allv)) if allv else float("nan"), {e: float(np.mean(v)) for e, v in by.items()}


def spell_accuracy(pairs: list[tuple[str, str]]) -> float:
    return float(np.mean([truth == pred for truth, pred in pairs])) if pairs else float("nan")


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def cmd_latency(last: int) -> None:
    lines = [json.loads(l) for l in LOG.read_text().splitlines()][-last:] if LOG.exists() else []
    for direction, xs in latency_stats(lines).items():
        if not xs:
            print(f"{direction}: no turns")
            continue
        p50, p90, mx = np.percentile(xs, 50), np.percentile(xs, 90), max(xs)
        print(f"{direction}: n={len(xs)} p50={p50:.0f} p90={p90:.0f} max={mx:.0f} ms  {_verdict(mx <= TARGET_MS)} (max <= {TARGET_MS})")
    errors = [l for l in lines if l.get("error")]
    if errors:
        print(f"{len(errors)} turns logged an error, e.g. {errors[-1]['error']}")


def cmd_grid() -> None:
    from server.grammar import read_grammar
    from server.session import Baseline, compute_baseline
    from training.landmarks import load_npz

    rows = []
    for signer_dir in sorted(OURS.glob("*/_grid")):
        takes = {p: load_npz(p) for p in sorted(signer_dir.glob("*.npz"))}
        # Baseline brows: the signer's neutral-face statements stand in for calibration.
        neutral = [f for p, fs in takes.items() if p.name.startswith("statement_neutral") for f in fs] or [f for fs in takes.values() for f in fs]
        baseline: Baseline = compute_baseline(neutral)
        for p, frames in takes.items():
            qtype, emotion = p.stem.split("_")[:2]
            g = read_grammar(frames, baseline)
            rows.append((qtype, emotion, g["question"] or "statement"))
    overall, by = grid_accuracy(rows)
    print(f"question type accuracy {overall:.3f} over {len(rows)} takes  {_verdict(overall >= 0.85)} (>= 0.85)")
    for e, acc in sorted(by.items()):
        print(f"  {e:8s} {acc:.3f}")


def cmd_signs(holdout: str) -> None:
    from server.classifier import DEFAULT_MODEL, SignPredictor
    from training.train_signs import LM, load_dir, topk_acc

    pred = SignPredictor(DEFAULT_MODEL)
    X, y = load_dir(LM / "test", pred.labels)
    t1, t3 = topk_acc(pred.model, X, y)
    print(f"ASL Citizen test: top-1 {t1:.3f} top-3 {t3:.3f}  {_verdict(t1 >= 0.80)} (top-1 >= 0.80)")
    if holdout:
        Xh, yh = load_dir(OURS / holdout, pred.labels)
        h1, h3 = topk_acc(pred.model, Xh, yh)
        print(f"holdout {holdout}: top-1 {h1:.3f} top-3 {h3:.3f}  {_verdict(h1 >= 0.70 and h3 >= 0.90)} (0.70 / 0.90)")


def cmd_spell(holdout: str) -> None:
    from server.fingerspell import SpellReader
    from server.session import Segment
    from training.landmarks import load_npz

    reader = SpellReader()
    pairs = []
    for p in sorted((OURS / holdout / "_names").glob("*.npz")):
        words = reader.recognize(Segment(frames=load_npz(p), kind="spell"))
        pairs.append((p.stem.split("_")[0], words[0][0] if words else ""))
    acc = spell_accuracy(pairs)
    print(f"names fully correct {acc:.3f} over {len(pairs)} takes  {_verdict(acc >= 0.60)} (>= 0.60)")
    for truth, got in pairs:
        if truth != got:
            print(f"  {truth} -> {got or '(nothing)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["signs", "grid", "spell", "latency"])
    ap.add_argument("--holdout-signer", default="")
    ap.add_argument("--last", type=int, default=10_000, help="latency: only the last N turns")
    a = ap.parse_args()
    {"signs": lambda: cmd_signs(a.holdout_signer), "grid": cmd_grid,
     "spell": lambda: cmd_spell(a.holdout_signer), "latency": lambda: cmd_latency(a.last)}[a.what]()


if __name__ == "__main__":
    main()
