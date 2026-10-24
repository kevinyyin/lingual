"""S8: end-to-end latency budget from measured stage times (no new hardware). Writes results/s8_budget.json.

sign -> speech:  last sign ends -> hands-at-rest detected -> gloss sequence final -> LLM English -> Cartesia first audio
speech -> sign:  speech ends -> Ink final transcript -> LLM gloss -> first avatar motion

Stage sums use medians and p90s added stage by stage (p90 of a sum is not the sum of p90s; the summed p90
is a conservative upper estimate). Sources are named per stage.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from stats import pstats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def jl(name):
    return [json.loads(x) for x in (R / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def main(rest_hold_ms: float = 300.0):
    stages_s2s, stages_s2g = [], []
    # per-frame landmark latency: MediaPipe Holistic per frame on this CPU (ASL Citizen extraction)
    ext = json.load(open(R / "s2_extract_log.json")) if (R / "s2_extract_log.json").exists() else []
    import glob
    per_frame = []
    for f in sorted(glob.glob(str(ROOT / "cache/s2/citizen/*.npz")))[:400]:
        with np.load(f) as z:
            if float(z["n_frames"]) > 0:
                per_frame.append(float(z["extract_ms"]) / float(z["n_frames"]))
    pf = pstats(per_frame)
    stages_s2s.append({"stage": "landmarks for the last frame (MediaPipe Holistic, CPU)", "median": pf["median"], "p90": pf["p90"],
                       "n": pf["n"], "source": "S2 extraction, ms per frame over 400 ASL Citizen videos while 7 extraction workers shared the CPU; a single unloaded process measured ~40 ms/frame on RWTH"})
    stages_s2s.append({"stage": "hands-at-rest detected (hold time)", "median": rest_hold_ms, "p90": rest_hold_ms, "n": None,
                       "source": "design parameter (300 ms of low hand speed); not measured on real end-of-turn behaviour"})
    cont = json.load(open(R / "s2_continuous_rwth.json")) if (R / "s2_continuous_rwth.json").exists() else None
    if cont:
        t = cont["thresholds"]["0.5"]
        stages_s2s.append({"stage": "gloss sequence final (windowed classifier over the utterance)", "median": t["compute_ms_median"],
                           "p90": t["compute_ms_p90"], "n": cont["n_utterances"], "source": "S2 continuous test (RWTH, CPU)"})
    g2e = [r["latency_ms"] for r in jl("s5_g2e_v2.jsonl") if r.get("latency_ms")]
    s = pstats(g2e)
    stages_s2s.append({"stage": "LLM gloss -> English (claude-haiku-5-5)", "median": s["median"], "p90": s["p90"], "n": s["n"],
                       "source": "S5 v2, non-streaming request, sequential"})
    tts = json.load(open(R / "s4_summary.json"))["latency"]["ttfb_ms"]
    stages_s2s.append({"stage": "Cartesia Sonic 3.6 first audio byte", "median": tts["median"], "p90": tts["p90"], "n": tts["n"],
                       "source": "S4, WebSocket, warm connection"})
    ink = [r for r in jl("s8_ink.jsonl") if r.get("end_ms") is not None]
    s = pstats([r["end_ms"] for r in ink])
    se = pstats([r["eager_end_ms"] for r in ink if r.get("eager_end_ms") is not None and r["eager_end_ms"] > -200])
    stages_s2g.append({"stage": "Ink 2 turn.end (final transcript)", "median": s["median"], "p90": s["p90"], "n": s["n"],
                       "source": "S8 Ink test, 30 barista lines, real-time streaming"})
    e2g = [r["latency_ms"] for r in jl("s5_e2g_v2.jsonl") if r.get("latency_ms")]
    s = pstats(e2g)
    stages_s2g.append({"stage": "LLM English -> gloss + NMM (incl. one repair retry when invalid)", "median": s["median"],
                       "p90": s["p90"], "n": s["n"], "source": "S5 v2"})
    av = json.load(open(R / "s6_avatar.json"))["first_motion_ms"]
    stages_s2g.append({"stage": "first avatar motion after gloss JSON", "median": av["warm_median"], "p90": av["p90"], "n": av["n"],
                       "source": "S6, Edge + Intel Arc GPU; lexicon preloaded (cold first run: %.0f ms)" % av["cold_first_run"]})

    def total(st):
        return {"median_sum": round(sum(x["median"] for x in st if x["median"] is not None), 1),
                "p90_sum": round(sum(x["p90"] for x in st if x["p90"] is not None), 1),
                "biggest": max(st, key=lambda x: x["median"] or 0)["stage"]}
    out = {"sign_to_speech": {"stages": stages_s2s, **total(stages_s2s)},
           "speech_to_sign": {"stages": stages_s2g, **total(stages_s2g),
                              "eager_end_variant": {"ink_eager_end_ms": se,
                                                    "note": "start the LLM on turn.eager_end and discard on turn.resume"}},
           "not_included": ["camera capture and network transport (S7 deferred)", "audio playback buffering on the phone/glasses",
                            "sign duration itself: the avatar takes 7-11 s to sign these sentences at citation speed"]}
    json.dump(out, open(R / "s8_budget.json", "w"), indent=1)
    for k in ("sign_to_speech", "speech_to_sign"):
        print(k, out[k]["median_sum"], out[k]["p90_sum"], "biggest:", out[k]["biggest"])
        for s_ in out[k]["stages"]:
            print("  ", s_["stage"], s_["median"], s_["p90"])


if __name__ == "__main__":
    main()
