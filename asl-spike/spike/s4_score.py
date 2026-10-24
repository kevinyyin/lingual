"""S4 scoring: SpeechBrain wav2vec2 IEMOCAP emotion classifier (neu/ang/hap/sad) on every S4 clip, then
detection rates per condition/voice/emotion, latency stats, and the mismatch test. Writes
results/s4_scores.jsonl and results/s4_summary.json."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, os.path.dirname(__file__))
from stats import pstats, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LAB = {"neu": "neutral", "ang": "angry", "hap": "happy", "sad": "sad"}


def load_clf():
    from speechbrain.inference.interfaces import foreign_class
    from speechbrain.utils.fetching import LocalStrategy
    return foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py",
                         classname="CustomEncoderWav2vec2Classifier", savedir=str(ROOT / "cache/models/sb_emotion"),
                         local_strategy=LocalStrategy.COPY)


def main() -> None:
    torch.manual_seed(0)
    runs = [json.loads(x) for x in (RES / "s4_tts_runs.jsonl").read_text().splitlines() if x.strip()]
    clf = load_clf()
    ind2lab = clf.hparams.label_encoder.ind2lab
    scored = []
    for r in runs:
        p = ROOT / "cache" / "s4" / f"{r['id']}.wav"
        if r.get("error") or not p.exists():
            scored.append(dict(r, pred=None))
            continue
        x, sr = sf.read(p, dtype="float32")
        x = soxr.resample(x, sr, 16000)
        with torch.no_grad():
            out_prob, score, index, text_lab = clf.classify_batch(torch.tensor(x)[None])
        probs = torch.softmax(out_prob[0], -1).numpy() if out_prob.ndim == 2 else out_prob[0].numpy()
        scored.append(dict(r, pred=LAB[text_lab[0]], probs={LAB[ind2lab[i]]: round(float(probs[i]), 4)
                                                             for i in range(len(probs))}))
    # second, independent classifier (HuBERT-large fine-tuned on IEMOCAP, SUPERB ER task) as a cross-check
    from transformers import pipeline
    hub = pipeline("audio-classification", model="superb/hubert-large-superb-er", device=-1)
    for s in scored:
        p = ROOT / "cache" / "s4" / f"{s['id']}.wav"
        if s.get("pred") is None:
            s["pred2"] = None
            continue
        x, sr = sf.read(p, dtype="float32")
        out = hub({"raw": soxr.resample(x, sr, 16000), "sampling_rate": 16000}, top_k=4)
        s["probs2"] = {LAB[o["label"]]: round(o["score"], 4) for o in out}
        s["pred2"] = LAB[out[0]["label"]]
    with open(RES / "s4_scores.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps({k: v for k, v in s.items() if not k.endswith("_ns")}) + "\n")

    summ: dict = {"n_clips": len(scored), "n_failed": sum(1 for s in scored if s.get("error")),
                  "classifier": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP (neu/ang/hap/sad), 16 kHz"}
    # latency
    ok = [s for s in scored if not s.get("error") and s.get("ttfb_ms")]
    summ["latency"] = {"ttfb_ms": pstats([s["ttfb_ms"] for s in ok]), "total_ms": pstats([s["total_ms"] for s in ok]),
                       "by_condition": {c: {"ttfb_ms": pstats([s["ttfb_ms"] for s in ok if s["cond"] == c]),
                                            "total_ms": pstats([s["total_ms"] for s in ok if s["cond"] == c])}
                                        for c in sorted({s["cond"] for s in ok})}}
    # detection: b and c score against intended; a (no emotion) scored against each of the 4 intended labels
    det: dict = {}
    for cond in ("a", "b", "c", "d", "e", "f"):
        rows = []
        for s in scored:
            if s["cond"] != cond or s.get("pred") is None:
                continue
            if cond == "a":
                for e in ("neutral", "happy", "sad", "angry"):
                    rows.append((s["voice"], e, s["pred"] == e))
            else:
                rows.append((s["voice"], s["intended"], s["pred"] == s["intended"]))
        k = sum(h for *_, h in rows)
        det[cond] = {"hit_rate": k / len(rows) if rows else None, "n": len(rows), "ci": wilson(k, len(rows)),
                     "by_voice": {v: _rate([h for vv, _, h in rows if vv == v]) for v in sorted({r[0] for r in rows})},
                     "by_emotion": {e: _rate([h for _, ee, h in rows if ee == e]) for e in ("neutral", "happy", "sad", "angry")},
                     "by_voice_emotion": {f"{v}/{e}": _rate([h for vv, ee, h in rows if vv == v and ee == e])
                                          for v in sorted({r[0] for r in rows}) for e in ("neutral", "happy", "sad", "angry")}}
    summ["detection"] = det
    det2 = {}
    for cond in ("a", "b", "c", "d", "e", "f"):
        rows = []
        for s in scored:
            if s["cond"] != cond or s.get("pred2") is None:
                continue
            for e in (("neutral", "happy", "sad", "angry") if cond == "a" else (s["intended"],)):
                rows.append((s["voice"], e, s["pred2"] == e))
        k = sum(h for *_, h in rows)
        det2[cond] = {"hit_rate": k / len(rows) if rows else None, "n": len(rows), "ci": wilson(k, len(rows)),
                      "by_emotion": {e: _rate([h for _, ee, h in rows if ee == e]) for e in ("neutral", "happy", "sad", "angry")},
                      "by_voice": {v: _rate([h for vv, _, h in rows if vv == v]) for v in sorted({r[0] for r in rows})}}
    summ["detection_hubert"] = det2
    summ["classifier2"] = "superb/hubert-large-superb-er (neu/hap/ang/sad), 16 kHz"
    for cond in ("b", "c", "d", "e", "f"):
        cm = defaultdict(lambda: defaultdict(int))
        for s in scored:
            if s["cond"] == cond and s.get("pred2"):
                cm[s["intended"]][s["pred2"]] += 1
        summ[f"confusion_hubert_{cond}"] = {k: dict(v) for k, v in cm.items()}
    # confusion for b and c
    for cond in ("b", "c", "d", "e", "f"):
        cm = defaultdict(lambda: defaultdict(int))
        for s in scored:
            if s["cond"] == cond and s.get("pred"):
                cm[s["intended"]][s["pred"]] += 1
        summ[f"confusion_{cond}"] = {k: dict(v) for k, v in cm.items()}
    pa = defaultdict(int)
    for s in scored:
        if s["cond"] == "a" and s.get("pred"):
            pa[s["pred"]] += 1
    summ["no_emotion_pred_distribution"] = dict(pa)
    # mismatch
    mm = {}
    for cond in ("mm_contra", "mm_match"):
        rows = [s for s in scored if s["cond"] == cond and s.get("pred")]
        mm[cond] = {"hit_rate": _rate([s["pred"] == s["intended"] for s in rows]),
                    "hit_rate_hubert": _rate([s["pred2"] == s["intended"] for s in rows]),
                    "pred_is_text_emotion": _rate([s["pred"] == s["text_emotion"] for s in rows]),
                    "items": [{"text": s["text"], "voice": s["voice"], "tag": s["emotion"], "pred": s["pred"]} for s in rows]}
    summ["mismatch"] = mm
    json.dump(summ, open(RES / "s4_summary.json", "w"), indent=1)
    print(json.dumps({k: summ[k] for k in ("n_clips", "n_failed")}, indent=0))
    for c in ("a", "b", "c", "d", "e", "f"):
        print(c, round(det[c]["hit_rate"], 3), det[c]["by_emotion"], det[c]["by_voice"])
    print("confusion b", summ["confusion_b"]); print("confusion c", summ["confusion_c"]); print("confusion d", summ["confusion_d"])
    print("no-emotion preds", summ["no_emotion_pred_distribution"])
    print("mismatch", {k: (v["hit_rate"], v["pred_is_text_emotion"]) for k, v in mm.items()})
    for c in ("a", "b", "c", "d", "e", "f"):
        print("hubert", c, round(det2[c]["hit_rate"], 3), det2[c]["by_emotion"])
    print("hubert confusion b", summ["confusion_hubert_b"]); print("hubert confusion d", summ["confusion_hubert_d"])
    print("latency", summ["latency"]["ttfb_ms"], summ["latency"]["total_ms"])


def _rate(hs: list[bool]) -> dict:
    k, n = sum(hs), len(hs)
    return {"rate": round(k / n, 3) if n else None, "k": k, "n": n}


if __name__ == "__main__":
    main()
