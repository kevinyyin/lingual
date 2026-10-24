"""Measure earbud output delay on this machine (optional hardware test).

Plays a sharp click to an output device while recording from an input device (a mic
held against the earbud), cross-correlates to find when the click was heard, repeats.

    uv sync --extra hw
    uv run python scripts/hw_bt_latency.py --list
    uv run python scripts/hw_bt_latency.py --out <id> --in <id> [--trials 20] [--label "AirPods Pro"]

The measured delay is the full round trip: OS output buffering + earbud/codec delay +
acoustic path + mic + OS input buffering. To isolate the earbud's contribution, measure
twice with the SAME mic and settings: once with a wired reference (wired earbuds or the
laptop speaker held to the mic) and once with the Bluetooth earbuds; the difference is
what Bluetooth adds. Add that to the model numbers for a realistic end-to-end figure.
Results are appended to results/hw_latency.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def click(sr: int) -> np.ndarray:
    """5 ms rising chirp: click-like, but correlates far more robustly than an impulse."""
    t = np.arange(int(0.005 * sr)) / sr
    c = np.sin(2 * np.pi * (1000 * t + 0.5 * (8000 - 1000) / t[-1] * t**2))
    return (c * np.hanning(len(c))).astype(np.float32) * 0.8


def measure_once(sd, sr: int, out_dev, in_dev, lead_s: float = 0.3, window_s: float = 1.2) -> tuple[float, float]:
    sig = np.zeros(int(window_s * sr), np.float32)
    c = click(sr)
    start = int(lead_s * sr)
    sig[start:start + len(c)] = c
    rec = sd.playrec(sig, samplerate=sr, channels=1, dtype="float32", device=(in_dev, out_dev), blocking=True)[:, 0]
    corr = np.correlate(rec, c, mode="valid")
    k = int(np.argmax(np.abs(corr)))
    snr = float(np.abs(corr[k]) / (np.median(np.abs(corr)) + 1e-9))
    return (k - start) / sr * 1000.0, snr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    ap.add_argument("--out", dest="out_dev", help="output device id or name substring")
    ap.add_argument("--in", dest="in_dev", help="input device id or name substring")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--label", default="", help="e.g. 'wired reference' or 'Bluetooth: Pixel Buds'")
    a = ap.parse_args()
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice not installed: run `uv sync --extra hw`")
    if a.list or a.out_dev is None or a.in_dev is None:
        print(sd.query_devices())
        if not a.list:
            print("\nPass --out and --in (ids or name substrings).")
        return
    conv = lambda d: int(d) if str(d).isdigit() else d  # noqa: E731
    out_dev, in_dev = conv(a.out_dev), conv(a.in_dev)
    delays, snrs = [], []
    for i in range(a.trials):
        d, snr = measure_once(sd, a.sr, out_dev, in_dev)
        ok = snr > 8
        print(f"trial {i + 1:2d}: {d:7.1f} ms  (correlation SNR {snr:5.1f}{'' if ok else ', REJECTED: click not heard clearly'})")
        if ok:
            delays.append(d)
            snrs.append(snr)
        time.sleep(0.3)
    if not delays:
        sys.exit("No clean detections. Hold the mic closer to the earbud and raise the volume.")
    res = {
        "label": a.label, "out_device": str(sd.query_devices(out_dev)["name"]), "in_device": str(sd.query_devices(in_dev)["name"]),
        "sample_rate": a.sr, "trials": a.trials, "accepted": len(delays),
        "median_ms": round(statistics.median(delays), 1), "p90_ms": round(float(np.percentile(delays, 90)), 1),
        "min_ms": round(min(delays), 1), "max_ms": round(max(delays), 1),
    }
    print(json.dumps(res, indent=1))
    out = ROOT / "results" / "hw_latency.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
