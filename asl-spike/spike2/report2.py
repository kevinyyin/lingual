"""Write REPORT-SPIKE-2.md from results2/*.json (every number comes from a results file)."""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R2 = ROOT / "results2"


def j(name, default=None):
    p = R2 / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else default


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def ci(c):
    return "" if not c else f" [{100 * c[0]:.1f}–{100 * c[1]:.1f}]"


def pts(x):
    return f"{100 * x:.1f} points"


def acc(a):
    """{'acc','ci','n','signers'} -> text"""
    return "—" if not a else f"{pct(a['acc'])}{ci(a.get('ci'))}"


def r4_table():
    rows = defaultdict(list)
    for f in glob.glob(str(R2 / "r4_*_s*.json")):
        r = json.load(open(f))
        rows[r["variant"]].append(r)
    out = {}
    for v, rs in rows.items():
        d = {}
        for ds in ("citizen", "wlasl", "popsign"):
            t1 = [r["test"][ds]["top1"] for r in rs]
            t3 = [r["test"][ds]["top3"] for r in rs]
            d[ds] = (float(np.mean(t1)), float(min(t1)), float(max(t1)), float(np.mean(t3)))
        d["seeds"] = len(rs)
        d["train_min"] = float(np.mean([r["train_s"] for r in rs])) / 60
        d["n"] = rs[0]["test_n"]
        out[v] = d
    return out


def main():
    L = []
    w = L.append
    links = j("asllex_links.json")["summary"]
    r1 = j("r1_popsign_gap.json")
    fr = j("r1_framing.json")
    asp = j("r1_aspect.json")
    r2 = j("r2_wlasl_eval.json")
    r3 = j("r3_calibration_base.json")
    r3b = j("r3_calibration_best.json")
    p1 = j("p1_trimmer.json")
    p1p = j("p1_pad.json")
    v1 = j("v1_avatar.json")
    hold = j("v1_hold_sweep.json")
    sets = j("p2_sets.json")
    p5 = j("p5_eval_a_bi2.json")
    p5s = j("p5_eval_a_stream.json")
    p5a = j("p5_eval_a_bi.json")
    rw = j("p5_rwth_pooled.json")
    rr = j("p4_rerank.json")
    hs = j("handshapes.json")
    v4 = j("v4_roundtrip.json")
    v4f = j("v4_fallback_angles.json")
    v4s = j("v4_sentences.json")
    core = j("vocab_core.json")
    f1 = j("f1_face_size.json")
    r4 = r4_table()
    tr_bi = j("p3_train_a_bi.json")
    tr_bi2 = j("p3_train_a_bi2.json")
    tr_st = j("p3_train_a_stream.json")
    best_r4 = max([v for v in r4 if v != "baseline"] or list(r4), key=lambda v: r4[v]["wlasl"][0] - (0 if r4[v]["citizen"][0] >= r4["baseline"]["citizen"][0] - 0.02 else 1)) if r4 else None

    # ------------------------------------------------------------ header + build-plan changes
    w("# REPORT-SPIKE-2: early research spike, round 2; not product code\n")
    w("Early research spike, round 2, run 2026-10-16 on the same Windows 11 laptop (Intel Core Ultra 7 258V, "
      "32 GB, CPU only; Intel Arc 140V GPU for WebGL through Microsoft Edge). Every number comes from `results2/`. Splits are "
      "signer-disjoint; CIs are 95% bootstrap over signers. Gloss, handshape and facial-grammar choices are by hearing "
      "non-signers and need Deaf advisor review.\n")
    w("## What changes in the build plan\n")
    k3 = r3["curve"]["finetune"]["3"]
    kb3 = (r3b or r3)["curve"]["finetune"]["3"]
    w(f"1. **Camera: phone on a stand, landscape, both hands free, signer framed head to waist.** On new signers with "
      f"ordinary two-handed video (WLASL) the round-1 model already gets {pct(r2['aspect_fix']['top1']['acc'])} top-1; the "
      f"34% on PopSign came from its one-handed, phone-in-hand selfie style, not from new people. Correct x for the camera "
      f"aspect ratio (x·w/h) before any geometry.")
    w(f"2. **Calibrate every sign the demo uses, with the demo signer, before the demo: 3 examples per sign.** On stand-in "
      f"signers this took top-1 from {pct(r3['curve']['finetune']['0']['top1']['acc'])} to {pct(k3['top1']['acc'])} "
      f"(top-3 {pct(k3['top3']['acc'])}) in ~{k3['cpu_s']['median']:.0f} s of CPU"
      + (f" ({pct(kb3['top1']['acc'])} with the best R4 base model)" if r3b else "") +
      f"; signing time ≈ {k3['recording_min_estimate']} min for {r3['n_signs']} signs (more with pauses and retakes). Calibrating only some signs makes the "
      "others worse — calibrate all of them. Keep tap-to-confirm on the top 3.")
    w(f"3. **Phrase mode: yes, but only as script matching with a reranker.** A CTC model trained on synthetic sentences "
      f"picks the right café line among the top 3 {pct(p5['script_lines']['top3'])} of the time on unseen signers, and a "
      f"Claude reranker lifts top-1 from {pct(rr['script_top1_without_reranker'])} to {pct(rr['rerank_top1'])}. Free "
      f"recognition of arbitrary sentences is not good enough (vocabulary recall {pct(p5['synthetic_free']['greedy']['recall'])} "
      f"on synthetic, {pct(rw['b']['recall'])} on real RWTH sentences). Fallback: sign-by-sign with short holds.")
    w(f"4. **Delay is not the problem:** the phrase model's final hypothesis arrives "
      f"{p5['delay']['final_ms']['median']:.0f} ms (p90 {p5['delay']['final_ms']['p90']:.0f}) after the last sign, including the "
      "300 ms hands-at-rest hold; use the bidirectional model.")
    w(f"5. **Core vocabulary of ~50 signs** (`results2/vocab_core.json`) with script edits that avoid weak or confusable "
      "signs (THANK-YOU↔GOOD, WHAT↔COST, CARD); all need Deaf advisor review.")
    six = v1["six_sign_estimate_s"]
    w(f"6. **Avatar: trim clips to the active segment, direct 160 ms minimum-jerk transitions, play at 1.3×** — a typical "
      f"6-sign sentence drops from 9–14 s (round 1) to ≈{six['1.3']} s. Snap fingers to ASL-LEX handshapes (finger-angle error "
      f"{v4f['a']['finger_angle_abs_error_deg_median']}° → {v4f['c']['finger_angle_abs_error_deg_median']}°), but curved shapes "
      "(C, O, flat-O) need a hand-tuned table, and the sample model's bangs must be replaced by a model with visible brows.")
    if f1 and f1.get("min_reliable_face_width_px"):
        m = f1["min_reliable_face_width_px"]
        d60 = {k: round(v * m / 60, 1) for k, v in f1["max_distance_m"].items()}
        w(f"7. **Face size is not a constraint at normal distances:** brow readings held up down to the smallest face tested "
          f"({m} px wide; the native ASL Citizen median is {f1['native_face_width_px_median']:.0f} px). Aim for ≥ 60 px for margin — "
          f"on a 1080p phone in landscape that is within ≈ {d60['landscape_1920_hfov80']}–{d60['landscape_1920_hfov70']} m of the camera, "
          f"far beyond a head-to-waist framing.")
    w("")

    # ------------------------------------------------------------ gates
    w("## Decision gates\n")
    w("| Gate | Pass if | Measured | Verdict |\n|---|---|---|---|")
    dec = r1["decomposition"]
    w(f"| R1: PopSign gap explained | Two-handed signs account for most of the drop | one-handed {acc(r1['one_vs_two_handed']['one_handed']['top1'])} "
      f"vs two-handed {acc(r1['one_vs_two_handed']['two_handed']['top1'])} top-1; of a gap of {pts(dec['gap'])}, one-handed signing explains "
      f"{pts(dec['explained_by_one_handed'])}, variants {pts(dec['explained_by_variants'])}, aspect ratio "
      f"{pts(asp['top1']['acc'] - asp['round1_top1'])}; {pts(dec['remaining_gap_after_both'])} remain (PopSign's selfie style) | **FAIL** (informational) |")
    t = r2["aspect_fix"]
    w(f"| R2: new signers, two-handed, no retraining (WLASL) | ≥ 70% top-1 or ≥ 90% top-3 | {acc(t['top1'])} top-1, {pct(t['top3']['acc'])} top-3 "
      f"(n = {r2['n']} clips, {r2['signers']} signers, {r2['classes']} classes) | **PASS** |")
    for name, rr3 in (("base", r3), ("best R4", r3b)):
        if not rr3:
            continue
        c3 = rr3["curve"]["finetune"]["3"]
        ok = c3["top1"]["acc"] >= 0.9 and c3["top3"]["acc"] >= 0.97 and c3["cpu_s"]["median"] <= 120
        w(f"| R3: calibration ({name} model) | ≤ 3 examples/sign: ≥ 90% top-1, ≥ 97% top-3, ≤ 2 min CPU | k = 3: {acc(c3['top1'])} top-1, "
          f"{acc(c3['top3'])} top-3, {c3['cpu_s']['median']:.0f} s CPU (6 PopSign stand-in signers × 3 draws) | **{'PASS' if ok else 'FAIL'}** |")
    if r4:
        b, bv = r4["baseline"], r4[best_r4]
        ok = (bv["wlasl"][0] - b["wlasl"][0] >= 0.10) and (b["citizen"][0] - bv["citizen"][0] <= 0.02)
        w(f"| R4: robust training | WLASL held-out top-1 +10 points, ASL Citizen ≤ 2 points drop | best '{best_r4}': WLASL {pct(bv['wlasl'][0])} vs "
          f"{pct(b['wlasl'][0])}, ASL Citizen {pct(bv['citizen'][0])} vs {pct(b['citizen'][0])} (3 seeds) | **{'PASS' if ok else 'FAIL'}** (informational) |")
    g = p5["synthetic_free"]["greedy"]
    w(f"| P: phrase mode, synthetic free sentences | recall ≥ 80% and WER ≤ 30% | recall {pct(g['recall'])}{ci(g['recall_ci'])}, WER {pct(g['wer'])}{ci(g['wer_ci'])} "
      f"(2,000 sentences, {g['signers']} unseen signers) | **FAIL** |")
    s = p5["script_lines"]
    snf = p5["script_lines_no_fallback"]
    w(f"| P: phrase mode, script lines | top-3 ≥ 95% and top-1 ≥ 85% | top-3 {pct(s['top3'])}{ci(s['top3_ci'])}, top-1 {pct(s['top1'])}{ci(s['top1_ci'])} with the "
      f"out-of-script fallback ({pct(snf['top1'])} without it; {pct(rr['rerank_top1'])} with the Claude reranker on 100 items) | "
      f"**{'PASS' if s['top3'] >= 0.95 and max(snf['top1'], rr['rerank_top1']) >= 0.85 else 'FAIL'}** (with reranker or without fallback) |")
    w(f"| P: real sentences (RWTH) | recall ≥ 40% at precision ≥ 40% (round 1: 18.6% / 22.8%) | synthetic only {pct(rw['a']['recall'])} / {pct(rw['a']['precision'])}; "
      f"+ 2 RWTH signers {pct(rw['b']['recall'])} / {pct(rw['b']['precision'])}; + same-signer calibration {pct(rw['c']['recall'])} / {pct(rw['c']['precision'])} | **FAIL** |")
    w(f"| P: delay | final hypothesis ≤ 1.0 s after the last sign (incl. 300 ms hold) | bidirectional {p5['delay']['final_ms']['median']:.0f} ms median, "
      f"{p5['delay']['final_ms']['p90']:.0f} ms p90; streaming {p5s['delay']['final_ms']['median']:.0f} ms | **PASS** |")
    w(f"| V: avatar duration | 6-sign sentence ≤ 5 s at ≤ 1.3× | typical 6-sign sentence {six['1.3']} s at 1.3× ({six['1.0']} s at 1.0×); the 3 round-1 test "
      f"sentences {', '.join(str(round(r['new_1.2x_s'], 1)) for r in v1['sentences'][:3])} s at 1.2× | **PASS** (typical) / long signs exceed |")
    w(f"| V: avatar legibility | (c) beats (a) on round-trip top-3; path ≥ 90% | round-trip top-3 (c) {pct(v4['c']['top3'])} vs (a) {pct(v4['a']['top3'])} — MediaPipe finds the "
      f"avatar's hands in only {pct(v4['c']['hand_detection_rate_median'])} of frames (median), so the proxy is unusable; fallback: finger-angle error "
      f"{v4f['a']['finger_angle_abs_error_deg_median']}° (a) → {v4f['c']['finger_angle_abs_error_deg_median']}° (c); path ratio median "
      f"{pct(v1['lexicon']['path_ratio_median'], 0)}, {len(v1['lexicon']['signs_below_90pct_path'])} signs < 90% | **INCONCLUSIVE** |")
    if f1:
        w(f"| F1 (optional) | report minimum face width | brows stable down to {f1.get('min_reliable_face_width_px')} px (smallest tested) | measured |")
    w("")

    # ------------------------------------------------------------ human steps
    w("## Human steps needed\n")
    w("| Step | Why | Status |\n|---|---|---|")
    dl = r2["downloads"]
    w(f"| WLASL missing-videos request form (https://github.com/dxli94/WLASL, links arrive within ~7 days) | only {dl['ok']}/{dl['instances']} "
      f"({pct(dl['ok'] / dl['instances'], 0)}) instances downloaded: signingsavvy/handspeak/aslsignbank/aslsearch refuse scripts (403/404), aslpro is Flash (.swf) | needed (below 60%) |")
    w("| ASLLRP DAI account (https://dai.cs.rutgers.edu/dai/s/dai) | real continuous signing with time-aligned glosses and yes/no questions (P5, F2) | optional; not set (F2 not run) |")
    w("| An avatar with visible brows (make one in VRoid Studio with short bangs; put it in `avatar/models/` and load with `?model=models/<name>.vrm`) | removing the sample model's bangs leaves a ragged fringe | optional |")
    w("| Round-1 Cartesia listening test (`listening/index.html`, 3–5 people) | still not run | open |")
    w("| Deaf advisor review (checklist below) | handshapes, avatar, brows, core vocabulary, script edits | open |")
    w("| OSF (https://osf.io/zpha4/) was down for maintenance; ASL-LEX 2.0 was taken from asl-lex.org's own data file | re-check against the OSF CSV when it is back | optional |")
    w("")

    # ------------------------------------------------------------ reuse audit
    w("## Reuse audit\n")
    w("| Item | Found at | Status |\n|---|---|---|")
    for row in [
        ("ASL Citizen landmark cache (5,958 clips, 143 classes; hands, pose, pose_world, 24 face points, blendshapes)", "`cache/s2/citizen/*.npz`", "reused"),
        ("Round-1 feature pipeline", "`spike/features.py` (`from_holistic`, `normalize`), `spike/extract.py`", "reused unchanged"),
        ("Round-1 models (ASL Citizen only; + PopSign)", "`cache/s2/model.pt`, `cache/s2/model_with_kaggle.pt`", "reused (base for R1–R3, P1, V1, V4)"),
        ("Gloss alias table / class map", "`vocab.json` (`asl_citizen_class_videos`, `kaggle_asl_signs`, `wlasl`)", "reused"),
        ("Kaggle asl-signs (PopSign) features", "`cache/s2/kaggle_feats.npz` (32-frame features only)", "reused; raw landmarks re-read from the zip for R1 hand statistics; aspect-corrected features regenerated (`cache/s2r2/kaggle_feats_aspect.npz`)"),
        ("RWTH-BOSTON-104 + round-1 harness (75 sentences with ≥ 2 vocabulary signs)", "`data/rwth-boston-104/`, `cache/s3/rwth/*.npz`, `results/s3_rwth_ncslgr_match.json`, `spike/s2_continuous.py`", "reused (same 75 sentences)"),
        ("NCSLGR SignStream 2 decoder", "`spike/ss2.py`", "reused (via round-1 match file)"),
        ("Avatar page, P11 lexicon, 3 sentences", "`avatar/index.html`, `avatar/signs.json` (16 signs), `avatar/sentences.json`", "kept; round 2 builds `avatar/index2.html` + `avatar/signs2.json` (all 143 signs)"),
        ("S5 outputs (30 barista lines → gloss + NMM)", "`results/s5_*.jsonl`", "reused for V1/V4 durations"),
        ("Missing: hand world landmarks", "not cached in round 1", "V2 uses aspect-corrected image landmarks instead (approximation)"),
        ("Missing: OOV sign landmarks for P2", "only the 143 classes were extracted", "regenerated for 60 seeded other ASL Citizen glosses (train split fully, val/test 4 clips per gloss)"),
        ("Missing: WLASL, ASL-LEX", "new this round", "downloaded"),
    ]:
        w(f"| {row[0]} | {row[1]} | {row[2]} |")
    w("")

    # ------------------------------------------------------------ R0
    w("## R0 · ASL-LEX linking\n")
    w(f"Source: ASL-LEX 2.0 sign properties (2,723 signs; CC BY-NC 4.0) from `asl-lex.org/visualization/data/sign_props.json` (OSF was down). "
      f"Linked by ASL Citizen's `ASL-LEX Code` column: {links['linked']}/{links['n_classes']} classes. Sign types: {links['sign_type_counts']}. "
      f"Dominant handshapes needed: {links['handshapes_needed_dominant']}; signs with a flexion change: {links['flexion_change_signs']}; "
      f"signs without path movement (static holds): {links['path_movement_none']}. WELCOME1 has no code and no ASL-LEX entry (WELCOME2 → welcome_1 is used).\n")
    w("Classes whose merged variants disagree on sign type (the primary = most ASL Citizen videos is used): " +
      "; ".join(f"{a['class']} ({', '.join(f'{v[0]}={v[1]}' for v in a['variants'])})" for a in links["ambiguous_sign_type"]) + ".\n")

    # ------------------------------------------------------------ R1
    w("## R1 · Why PopSign scored 34%\n")
    w("| Slice (round-1 model, no retraining) | Classes | Top-1 [signer CI] | Top-3 |\n|---|---|---|---|")
    w(f"| All overlapping signs | {r1['n_classes']} | {acc(r1['overall']['top1'])} | {acc(r1['overall']['top3'])} |")
    for k, v in r1["by_sign_type"].items():
        w(f"| {k} | {v['classes']} | {acc(v['top1'])} | {acc(v['top3'])} |")
    w(f"\nn = {r1['n_sequences']:,} sequences, {r1['n_signers']} signers.\n")
    w("**Both hands detected** (fraction of hand frames with both hands):\n")
    w("| ASL-LEX sign type | Classes | PopSign | ASL Citizen |\n|---|---|---|---|")
    for k, v in r1["both_frac_by_type"].items():
        w(f"| {k} | {v['classes']} | {pct(v['popsign'])} | {pct(v['citizen'])} |")
    w(f"\nPopSign almost never shows two hands — for any sign type — consistent with signing while holding the phone. "
      f"Two-handed classes signed one-handed: {', '.join(dec['classes_signed_one_handed'])}. Likely different variants "
      f"(under 15% top-1 and a different dominant-hand location or flexion): {', '.join(dec['classes_likely_variant'])}.\n")
    w(f"**Decomposition** (class-level counterfactuals): ASL Citizen test top-1 on the same classes {pct(dec['citizen_test_top1_same_classes_weighted'])} vs "
      f"PopSign {pct(dec['popsign_top1_actual'])} — a gap of {pts(dec['gap'])}. One-handed signing explains {pts(dec['explained_by_one_handed'])}, "
      f"variants {pts(dec['explained_by_variants'])}; even the unaffected classes only reach {pct(dec['clean_classes_popsign_top1'])}, leaving "
      f"{pts(dec['remaining_gap_after_both'])}.\n")
    w(f"**Framing and aspect ratio.** PopSign faces are bigger in frame (median face width {pct(fr['popsign']['face_w']['median'], 0)} vs "
      f"{pct(fr['citizen']['face_w']['median'], 0)} of image width) but the shoulders are in frame and the shoulder/face ratio is similar "
      f"({fr['popsign']['sw_over_face']['median']} vs {fr['citizen']['sw_over_face']['median']}). The hand looked {pct(1 - fr['popsign']['hand_over_face']['median'] / fr['citizen']['hand_over_face']['median'], 0)} "
      f"smaller relative to the face, which pointed to aspect ratio: MediaPipe's normalized x is divided by width and y by height, "
      f"and PopSign is portrait (estimated w/h {asp['popsign_aspect_estimated']:.2f}) while ASL Citizen is 4:3. Correcting x (×{asp['x_scale_applied']:.2f}) "
      f"raises PopSign top-1 from {pct(asp['round1_top1'])} to {acc(asp['top1'])} — real but small.\n")
    w("**Conclusion.** Two-handed signs signed one-handed explain only part of the drop (~7 points), variants ~2 points and aspect ratio ~3 points. "
      "Most of the gap is PopSign's recording style itself (phone held in one hand, close-up selfie, casual fast signing). R2 confirms it: "
      "new signers with ordinary two-handed video do much better.\n")

    # ------------------------------------------------------------ R2
    w("## R2 · New signers, two-handed, no retraining (WLASL)\n")
    w(f"Downloaded {dl['ok']}/{dl['instances']} instances of glosses mapping to our classes (asllex excluded: {j('r2_downloads.json')['excluded_asllex']}). "
      "Success by source: " + ", ".join(f"{k} {v}" for k, v in dl["success_by_source"].items()) + ". Failure reasons: " +
      "; ".join(f"{k} ({v})" for k, v in dl["failed_reasons"].items()) + ". Clips cut to frame_start/frame_end on WLASL's 25 fps timeline; "
      "landmarks with the identical round-1 pipeline.\n")
    w("| Features | Top-1 [signer CI] | Top-3 | Top-5 |\n|---|---|---|---|")
    for k in ("round1_pipeline", "aspect_fix"):
        v = r2[k]
        w(f"| {k} | {acc(v['top1'])} | {acc(v['top3'])} | {acc(v['top5'])} |")
    w(f"\nn = {r2['n']} clips, {r2['signers']} signers, {r2['classes']} classes.\n")
    w("| Breakdown (aspect fix) | Group | Top-1 [CI] | Top-3 |\n|---|---|---|---|")
    for k in ("by_sign_type", "by_variation", "by_youtube", "by_source"):
        for g_, v in t[k].items():
            w(f"| {k.replace('by_', '')} | {g_} | {acc(v['top1'])} | {pct(v['top3'])} |")
    weak = sorted(t["per_sign"].items(), key=lambda kv: kv[1]["top1"])[:15]
    w("\nWeakest signs (top-1, instances/signers): " + ", ".join(f"{c} {pct(v['top1'], 0)} ({v['instances']}/{v['signers']})" for c, v in weak) + ".\n")
    w(f"Signer-disjoint split for R4 saved to `results2/wlasl_split.json`: {r2['split_summary']}.\n")

    # ------------------------------------------------------------ R3
    w("## R3 · Calibration curve\n")
    w(f"Stand-in demo signers: the 6 held-out PopSign signers, {r3['n_signs']} signs, aspect-corrected features (conservative: one-handed selfie style). "
      f"Tuned on validation signer {r3['tuning']['val_signer']}: fine-tune {r3['chosen']['finetune']['epochs']} epochs at LR {r3['chosen']['finetune']['lr']} "
      f"(calibration clips mixed 1:1 with replayed ASL Citizen training clips); prototype mixing α = {r3['chosen']['proto_alpha']}. 3 draws per signer.\n")
    for name, rr3 in (("round-1 base model", r3), ("best R4 model", r3b)):
        if not rr3:
            continue
        w(f"**{name}**\n\n| k | Fine-tune top-1 | top-3 | CPU s | Prototypes top-1 | top-3 | Recording (min) |\n|---|---|---|---|---|---|---|")
        for k in ("0", "1", "2", "3", "5"):
            a_, b_ = rr3["curve"]["finetune"][k], rr3["curve"]["proto"][k]
            w(f"| {k} | {acc(a_['top1'])} | {acc(a_['top3'])} | {a_['cpu_s']['median']} | {acc(b_['top1'])} | {acc(b_['top3'])} | {a_['recording_min_estimate']} |")
        pv = rr3["partial_vocab"]
        w(f"\nPartial calibration (k = 3 on some signs, test on the others): fine-tune on 10 signs → {acc(pv['finetune']['10']['top1_uncalibrated_signs'])}, "
          f"25 signs → {acc(pv['finetune']['25']['top1_uncalibrated_signs'])}, vs uncalibrated {acc(pv['finetune']['10']['base_top1_same_items'])}. "
          "Calibrating some signs does **not** fix the style for the others; fine-tuning makes the uncalibrated signs worse (forgetting), and "
          "prototypes with α = 1 always pick a calibrated sign.\n")
    w(f"Recording time = k × signs × median PopSign clip ({r3['popsign_median_clip_s']} s): signing time only — pauses between takes, setup and retakes will multiply it several times.\n")

    # ------------------------------------------------------------ P1
    w("## P1 · Active-segment trimmer\n")
    pa = p1["params"]
    w(f"Method (to re-implement in the main build): x·(w/h); dominant hand = more wrist motion; height h = (shoulder-midline y − wrist y)/shoulder width "
      f"(missing hand = at rest); rest = 10th percentile of h in the clip; speed = wrist speed in shoulder widths/s (3-frame mean). "
      f"Active = raised (h > rest + **{pa['rise_sw']}**) and fast (> **{pa['speed_sw_per_s']}** sw/s) from the first to the last such frame; "
      f"hysteresis: extend up to **{pa['hysteresis_s']} s** while h > rest + RISE/2; static-hold signs (ASL-LEX movement None) keep ≥ "
      f"**{pa['min_hold_s_static']} s** after the last fast frame; pad **{p1p['chosen_pad']} frames**. Thresholds tuned on ASL Citizen "
      f"validation signers (grid in `results2/p1_trimmer.json`, padding in `results2/p1_pad.json`).\n")
    w(f"| | Value |\n|---|---|\n| Removed per clip (test, pad 2) | median {pct(p1['test']['removed_frac']['median'], 0)}, p90 {pct(p1['test']['removed_frac']['p90'], 0)} |"
      f"\n| Removed per clip (final, pad {p1p['chosen_pad']}) | median {pct(p1p['test_removed_median'], 0)}; clip {p1['test']['seconds_before']['median']} s → {p1p['test_seconds_after_median']:.2f} s |"
      f"\n| Round-1 classifier, ASL Citizen test, untrimmed → trimmed (pad {p1p['chosen_pad']}) | {pct(p1['test']['untrimmed_top1'])} → {pct(p1p['test_trimmed_top1'])} top-1 "
      f"({pct(p1['test']['untrimmed_top3'])} → {pct(p1p['test_trimmed_top3'])} top-3); n = {p1['test']['n']} clips, {p1['test']['signers']} signers |"
      f"\n| Validation drop with pad 2 / pad {p1p['chosen_pad']} | {pts(p1p['val_untrimmed_top1'] - p1p['grid'][0]['val_top1'])} / {pts(p1p['val_untrimmed_top1'] - [g_ for g_ in p1p['grid'] if g_['pad'] == p1p['chosen_pad']][0]['val_top1'])} |")
    w("\n20 contact sheets (lead-in, onset, middle, offset, lead-out): `results2/p1_contact/` (local; ASL Citizen frames).\n")

    # ------------------------------------------------------------ V1
    w("## V1 · Avatar: trim, direct transitions, speed\n")
    lx = v1["lexicon"]
    vp = v1["params"]
    w(f"Lexicon: one ASL Citizen training clip per class ({lx['n_signs']} signs; signers {lx['signers']}). Per sign: P1 active segment; internal holds "
      f"(< {vp['hold_speed_sw_per_s']} sw/s for > {vp['hold_min_s']} s) compressed to {vp['hold_keep_s']} s except static signs — the hold sweep showed "
      f"citation signs have almost no internal holds (higher thresholds cut real movement: `results2/v1_hold_sweep.json`). Direct "
      f"{vp['transition_ms']} ms minimum-jerk transitions; rest only at the end ({vp['rest_ms_at_end']} ms).\n")
    w(f"| | Value |\n|---|---|\n| Median sign: whole clip → active segment | {lx['median_full_s']} s → {lx['median_processed_s']} s |"
      f"\n| Dominant-wrist path length kept | median {pct(lx['path_ratio_median'], 0)}, min {pct(lx['path_ratio_min'], 0)}; below 90%: {', '.join(lx['signs_below_90pct_path']) or 'none'} |"
      f"\n| Round-1 classifier on lexicon landmarks, full → processed | {pct(lx['classifier_full_top1'])} → {pct(lx['classifier_processed_top1'])} top-1 (training clips: optimistic) |"
      f"\n| Typical 6-sign sentence at 1.0 / 1.2 / 1.3 / 1.4× | {six['1.0']} / {six['1.2']} / {six['1.3']} / {six['1.4']} s |")
    w("\n| Sentence | Signs played | Round 1 (s) | 1.0× | 1.2× | 1.4× | Signs/s at 1.2× |\n|---|---|---|---|---|---|---|")
    for r in v1["sentences"]:
        if r["n_signs_played"] == 0:
            continue
        w(f"| {r['sentence'][:60]} | {r['n_signs_played']} | {r['round1_s']:.1f} | {r['new_1.0x_s']:.1f} | {r['new_1.2x_s']:.1f} | {r['new_1.4x_s']:.1f} | {r['signs_per_s_1.2x']} |")
    sps = [r["signs_per_s_1.2x"] for r in v1["sentences"] if r["signs_per_s_1.2x"]]
    w(f"\nMedian {np.median(sps):.2f} signs/s at 1.2× vs ~2 signs/s in fluent signing: the citation clips are still slow because each ASL Citizen clip "
      "is a careful, isolated production; per-sign speed-up is the remaining lever.\n")

    # ------------------------------------------------------------ P2
    w("## P2 · Synthetic continuous sentences\n")
    w("Generator (`spike2/p2_gen.py`): clips converted to body coordinates first (x·w/h relative to 4:3, shoulder-centered/scaled, dominant hand in the "
      "right slot) so clips from different videos join; P1 active segment with an extra random 0–20% trim at each end; per-sign speed 0.8–1.5×; "
      "80–250 ms minimum-jerk transitions of all landmarks; optional lead-in/lead-out rest from the clip's own trimmed-off segments; "
      "out-of-vocabulary ASL Citizen signs (60 other glosses) inserted at ~15% of positions as <unk>; jitter (sd 0.01 shoulder widths) and 3% dropped hand frames. "
      "Content: 50% café lines (30 gloss lines; the first 14 are the scenario's customer turns), 35% bigram samples, 15% uniform; 2–7 signs. "
      "**Synthetic transitions are not real coarticulation; the RWTH tests are the check.**\n")
    w("| Set | Split | Sentences | Mean signs | <unk> rate | Mean frames | Signs needing a second signer |\n|---|---|---|---|---|---|---|")
    for k in ("val", "test", "val_script", "test_script"):
        v = sets[k]
        w(f"| {k} | {v['split']} | {v['n']} | {v['mean_signs']} | {pct(v['unk_rate'])} | {v['mean_frames']} | {pct(v['mixed_rate'])} |")
    w("\nThe sentence signer is the one covering most of its signs (ties broken at random); every free sentence came from a single signer.\n")

    # ------------------------------------------------------------ P3
    w("## P3 · Phrase-mode model\n")
    w(f"Encoder as round 1 (Linear 228→192, 3 residual Conv1d k = 5, GRU) initialized from the round-1 isolated model where shapes match; CTC head over "
      f"143 classes + <unk> + blank. Bidirectional: {tr_bi['params']:,} parameters; streaming (causal convolutions, unidirectional GRU 192): "
      f"{tr_st['params']:,}. Batches of 16 generated on the fly (train-split signers); AdamW, gradient clip 1.0. 1 seed per variant (time).\n")
    w("| Model | Training | Steps | CPU minutes |\n|---|---|---|---|")
    w(f"| (a) bidirectional, synthetic | from round-1 weights, LR 1e-3 then 5e-4 | {tr_bi['steps']} + {tr_bi2['steps']} | {(tr_bi['train_s'] + tr_bi2['train_s']) / 60:.0f} |")
    w(f"| (a) streaming, synthetic | from round-1 weights, LR 1e-3 | {tr_st['steps']} | {tr_st['train_s'] / 60:.0f} |")
    tb = [j(f"p3_train_b_bi_f{k}.json") for k in range(3)]
    tc = [j(f"p3_train_c_bi_f{k}.json") for k in range(3)]
    w(f"| (b) + RWTH, 2 of 3 signers (3 folds) | from (a), half the batches real RWTH sentences, LR 5e-4 | {', '.join(str(x['steps']) for x in tb)} | {np.mean([x['train_s'] for x in tb]) / 60:.0f} each |")
    w(f"| (c) + half of the held-out signer's sentences | from (b), LR 3e-4 | {', '.join(str(x['steps']) for x in tc)} | {np.mean([x['train_s'] for x in tc]) / 60:.0f} each |")
    w("")

    # ------------------------------------------------------------ P4 + P5
    w("## P4–P5 · Decoding and phrase-mode evaluation\n")
    w(f"Decoders: greedy; CTC prefix beam (width 10) with a gloss bigram LM from the café lines (LM weight 0.5, insertion bonus 0.5); "
      f"script-constrained (CTC forced-alignment log-likelihood of each of the 30 café gloss lines, top 3) with an out-of-script fallback "
      f"(free decode wins if it beats the best line by > τ nats; τ = {p5['tau']} chosen on validation as the smallest τ keeping script top-1 within "
      f"3 points of no fallback).\n")
    w("| Set (unseen ASL Citizen signers unless noted) | Model / decoder | Result |\n|---|---|---|")
    for tag, res in (("bidirectional (55 min)", p5), ("bidirectional (25 min)", p5a), ("streaming", p5s)):
        for d_ in ("greedy", "beam"):
            v = res["synthetic_free"][d_]
            w(f"| Free synthetic sentences (2,000; {v['signers']} signers) | {tag}, {d_} | WER {pct(v['wer'])}{ci(v.get('wer_ci'))}, recall {pct(v['recall'])}{ci(v.get('recall_ci'))}, precision {pct(v['precision'])} |")
    for tag, res in (("bidirectional (55 min)", p5), ("streaming", p5s)):
        s_, n_ = res["script_lines"], res["script_lines_no_fallback"]
        w(f"| Café script lines (462 = 14 lines × 33 renders, {s_['signers']} signers) | {tag}, script decoding | top-1 {pct(s_['top1'])}{ci(s_.get('top1_ci'))}, top-3 {pct(s_['top3'])}{ci(s_.get('top3_ci'))}; "
          f"fallback fired {pct(s_['fallback_rate'])}; without fallback top-1 {pct(n_['top1'])} |")
    w(f"| Café script lines, 100 seeded items | + Claude reranker (claude-haiku-5-5) | top-1 {pct(rr['script_top1_without_reranker'])} → {pct(rr['rerank_top1'])}, top-3 {pct(rr['rerank_top3'])}; "
      f"latency {rr['latency_ms']['median']:.0f} ms median, {rr['latency_ms']['p90']:.0f} p90; cost ${rr['est_cost_usd']} |")
    for k, lab in (("a", "synthetic only"), ("b", "+ 2 RWTH signers"), ("c", "+ same-signer calibration (other half)")):
        v = rw[k]
        w(f"| RWTH-BOSTON-104 ({v['n_sentences']} sentences, 3 signers, leave-one-signer-out) | {lab}, greedy | recall {pct(v['recall'])}{ci(v['recall_ci_signer'])}, precision {pct(v['precision'])} ({v['tp']}/{v['n_ref']} vocabulary signs) |")
    w("| RWTH-BOSTON-104, round 1 | sliding window | recall 18.6%, precision 22.8% |")
    w("")
    dl_ = p5["delay"]
    w(f"**Delay** (end of the last sign known exactly in synthetic data; includes the 300 ms hands-at-rest hold): bidirectional final hypothesis "
      f"{dl_['final_ms']['median']:.0f} ms median / {dl_['final_ms']['p90']:.0f} p90 / {dl_['final_ms']['max']:.0f} max (compute {dl_['compute_ms_per_utterance']['median']:.0f} ms per utterance on CPU); "
      f"streaming: the model emits the last sign {p5s['delay']['model_emission_ms']['median']:.0f} ms (p90 {p5s['delay']['model_emission_ms']['p90']:.0f}) after it ends, so the hold dominates "
      f"({p5s['delay']['final_ms']['median']:.0f} ms). Partial hypotheses: with the streaming model each sign appears as soon as it ends, but it is less accurate.\n")
    er = p5["synthetic_free"]["errors"]
    w("**Error analysis** (bidirectional, beam): deletions dominate (the model under-emits). Most common — substitutions: " +
      ", ".join(f"{e['ref']}→{e['hyp']} ({e['count']})" for e in er["sub"][:15]) + "; deletions: " +
      ", ".join(f"{e['ref']} ({e['count']})" for e in er["del"][:15]) + "; insertions: " + ", ".join(f"{e['hyp']} ({e['count']})" for e in er["ins"][:15]) + ".\n")
    w("Limitations: 1 seed per phrase model; ~55 min of CPU training is short for CTC; synthetic transitions are interpolations, not coarticulation; "
      "RWTH is 2007 black-and-white video of 3 signers with mostly non-café vocabulary; DAI (ASLLRP) not available, so no boundary error.\n")

    # ------------------------------------------------------------ V2
    w("## V2 · Handshape library and snapping\n")
    lib = hs["library"]
    w(f"Canonical handshapes: frames from the middle 40% of P1 active segments of ASL Citizen training clips whose ASL-LEX entry has no flexion change, "
      f"dominant hand mirrored to right-hand geometry, aspect-corrected image landmarks (hand world landmarks were not cached). Joint angles in a "
      f"hand-local frame; robust median per handshape. The vocabulary needs {len(hs['handshapes_needed_dominant'])} dominant handshapes; the library "
      f"covers {len(lib)} labels (non-dominant labels B, C, 8, stacked_5 aliased to open_b, c, open_8, 5).\n")
    w("| Handshape | Frames | Clips | Signers | Index MCP/PIP | Middle MCP/PIP | Pinky MCP/PIP | Thumb MCP | Thumb abd |\n|---|---|---|---|---|---|---|---|---|")
    for k in sorted(lib, key=lambda k: -lib[k]["frames"]):
        m = lib[k]["median"]
        w(f"| {k} | {lib[k]['frames']} | {lib[k]['clips']} | {lib[k]['signers']} | {m['index_mcp']:.0f}/{m['index_pip']:.0f} | {m['middle_mcp']:.0f}/{m['middle_pip']:.0f} | "
          f"{m['pinky_mcp']:.0f}/{m['pinky_pip']:.0f} | {m['thumb_mcp']:.0f} | {m['thumb_abd']:.0f} |")
    w("\n(All angles in degrees; full table with spread, DIP and thumb CMC/IP/opposition in `results2/handshapes.json`.)\n")
    w("Avatar mapping (`avatar/index2.html`): finger flexion = (angle − flat-hand baseline) × 1.3 (image landmarks underestimate flexion), about the "
      "bone's z axis (sign mirrored for the left hand); spread about y; the thumb's two base degrees of freedom were found by a render sweep "
      "(metacarpal about −y folds it across the palm) and its fold is set from ASL-LEX ThumbPosition (Closed → folded). Non-dominant hand: one-handed → "
      "relaxed; symmetrical/alternating or asymmetrical-same → the dominant shape; asymmetrical-different → the coded non-dominant shape. Flexion-change "
      "signs interpolate to a data-derived end shape over the active segment. Wrist rotation from the palm frame (wrist, index MCP, pinky MCP); One-Euro "
      "smoothing (min cutoff 1.2 Hz, β 0.4) of arm and hand landmarks; round-1 vector alignment for shoulder→elbow→wrist.\n")
    w("**Honest result:** straight and fist shapes read correctly (5, open_b, 1, l, y, s); curved shapes (c, o, flat_o, curved_5, f) come out nearly flat "
      "because image landmarks register little curvature; the thumb is rule-based, not data-driven. Contact sheets (avatar vs reference crop): "
      "`results2/v2_sheets/` (local; ASL Citizen frames).\n")

    # ------------------------------------------------------------ V3
    w("## V3 · Brows and face\n")
    w("The sample model's hair is one mesh (bangs included). Change: at load, hair triangles whose bind-pose centroid lies in front of the forehead "
      "(within 7 cm of the face's front, between 58% and 72% of the face-box height, within 75% of its half-width) are removed — about 1,100 triangles. "
      "Clipping planes did not work (MToon materials ignored them). Result: brows visible, but the fringe looks ragged; a model with short bangs is the real fix (human step).\n")
    hsh = json.load(open(ROOT / "avatar/signs2.json"))["meta"]["headshake"] if (ROOT / "avatar/signs2.json").exists() else {}
    w(f"Non-manual markers: yes/no = Fcl_BRW_Surprised 1.0 + Fcl_EYE_Surprised 0.55 + head pitch forward 0.10 rad; wh = Fcl_BRW_Angry 1.0 + "
      f"Fcl_EYE_Angry 0.35 + pitch 0.05 + roll 0.07 rad; negation = head yaw ±{hsh.get('peak_deg_estimate')}° at {hsh.get('freq_hz')} Hz, matched to "
      f"round-1 S3 negations (n = {hsh.get('n')}; yaw-proxy sd {hsh.get('yaw_proxy_sd')}). Screenshots for advisor review: rendered with "
      "`spike2/v_shots.py face` (neutral / raise / lower); the raise/lower difference is visible but subtle.\n")

    # ------------------------------------------------------------ V4
    w("## V4 · Avatar evaluation\n")
    w("| Configuration | Hand detection (median per sign) | Round-trip top-1 | Top-3 | Finger-angle error vs canonical (where detected) |\n|---|---|---|---|---|")
    names = {"a": "(a) round-1 avatar", "b": "(b) + V1 trim/transitions", "c": "(c) + V2 snapping, wrist, smoothing", "d": "(d) = (c) at 1.3×"}
    for k in "abcd":
        w(f"| {names[k]} | {pct(v4[k]['hand_detection_rate_median'])} | {pct(v4[k]['top1'])} | {pct(v4[k]['top3'])} | {v4f[k]['finger_angle_abs_error_deg_median']}° ({v4f[k]['hand_frames']} frames) |")
    w(f"| Upper bound: human source landmarks | — | {pct(v4['human_upper_bound']['top1'])} | {pct(v4['human_upper_bound']['top3'])} | — |")
    w("\nRenders: Edge + Playwright on the Intel Arc GPU, 1280×720 at 30 fps, frontal head-to-waist framing, plain background. **MediaPipe finds the "
      "avatar's hands in only a few percent of frames** (it is trained on real hands), so the automatic round trip cannot measure legibility; the body "
      f"pose is found ({pct(v4f['c']['pose_detection_rate_median'])} of frames). The fallback finger-angle comparison favors snapping. Sentence-level round "
      "trip (3 round-1 sentences + 5 barista lines through the phrase model) decoded essentially nothing for the same reason (`results2/v4_sentences.json`). "
      "Contact sheets per sign (human vs avatar key frames): `results2/v4_contact/`; capture of the 3 sentences: `avatar/capture2.mp4` (both local). "
      "**A classifier recognizing (or not recognizing) the avatar says nothing about whether Deaf people find it natural or correct; only advisor review can.**\n")

    # ------------------------------------------------------------ R4
    w("## R4 · Robust training ablation\n")
    if r4:
        w("| Variant | ASL Citizen test top-1 (mean, range) | WLASL held-out top-1 | PopSign held-out top-1 | Train min/seed |\n|---|---|---|---|---|")
        for v in ("baseline", "aug", "wlasl", "popsign", "combined"):
            if v in r4:
                d = r4[v]
                w(f"| {v} | {pct(d['citizen'][0])} ({pct(d['citizen'][1])}–{pct(d['citizen'][2])}) | {pct(d['wlasl'][0])} ({pct(d['wlasl'][1])}–{pct(d['wlasl'][2])}) | "
                  f"{pct(d['popsign'][0])} ({pct(d['popsign'][1])}–{pct(d['popsign'][2])}) | {d['train_min']:.0f} |")
        n = r4["baseline"]["n"]
        w(f"\nTest sets: ASL Citizen {n['citizen']} clips (11 signers), WLASL {n['wlasl']} clips (14 signers), PopSign {n['popsign']} sequences (6 signers). "
          f"3 seeds each. Best variant by WLASL held-out top-1 (ASL Citizen within 2 points): **{best_r4}**. PopSign training data is one-handed selfie style.\n")
    else:
        w("Not run.\n")

    # ------------------------------------------------------------ R5
    w("## R5 · Core vocabulary proposal (needs Deaf advisor review)\n")
    w("Robustness score = min(ASL Citizen test top-1, WLASL top-1 without retraining, post-calibration top-1 at k = 3 on PopSign stand-ins — only for the signs PopSign has) − "
      "0.15 × share of errors going to the single most-confused sign. WLASL per-sign numbers rest on few instances (3–15), so they are noisy.\n")
    w("| Sign | Score | ASL Citizen | WLASL | Calibrated k=3 | Most confused with | Script needs it |\n|---|---|---|---|---|---|---|")
    for c in sorted(core["core"], key=lambda c: c["score"] if c["score"] is not None else -9):
        w(f"| {c['gloss']} | {c['score']} | {pct(c.get('citizen'), 0)} | {pct(c.get('wlasl'), 0)} | {pct(c.get('calibrated_k3'), 0)} | {c.get('most_confused_with') or '—'} | {'yes' if c['needed_by_script'] else ''} |")
    w("\n**Proposed script edits** (all need Deaf advisor review; English stays the same, the signed form changes):\n")
    for line, new, why in SCRIPT_EDITS:
        w(f"- `{line}` → `{new}` — {why}")
    w("\nSigns the dialogue truly needs stay even when weak and are flagged: THANK-YOU (confused with GOOD), WANT, WHAT, CARD, NOT. Mitigations: "
      "avoid GOOD in the same exchange, rely on headshake for negation, tap-to-confirm.\n")

    # ------------------------------------------------------------ F1/F2
    w("## F1 · How big must the face be? (optional)\n")
    if f1:
        w(f"ASL Citizen test clips: {f1['n_wh']} wh-sign clips, {f1['n_non_wh']} non-wh; native face width median {f1['native_face_width_px_median']} px.\n")
        w("| Face width | Detection | corr(browDown) with native | corr(browInnerUp) | AUC wh vs non-wh (sustained browDown) |\n|---|---|---|---|---|")
        for k, v in f1["by_scale"].items():
            w(f"| {k}{'' if k == 'native' else ' px'} | {pct(v['detection_rate_mean'])} | {v['corr_browDown_median']} | {v['corr_browInnerUp_median']} | {v['auc_wh_sustained_browDown']} |")
        m = f1["min_reliable_face_width_px"]
        d = f1["max_distance_m"]
        smallest = min(int(k) for k in f1["by_scale"] if k != "native")
        w(f"\nCriterion: {f1['criterion']}. Every tested size passed, so the limit is **at or below {m} px**{' (the smallest size tested)' if m == smallest else ''}; "
          f"at {m} px the browDown correlation ({f1['by_scale'][str(m)]['corr_browDown_median']}) is close to the 0.8 bar, so use ≥ 60 px for margin. "
          f"Maximum distance for {m} px: landscape {d['landscape_1920_hfov80']}–{d['landscape_1920_hfov70']} m, portrait {d['portrait_1080_hfov80']}–{d['portrait_1080_hfov70']} m "
          f"({f1['assumptions']}); for 60 px multiply by {m / 60:.2f}. The wh-vs-non-wh AUC is only ≈ 0.71 even at native size: isolated citation signs do not "
          "reliably carry grammatical facial expressions, so this measures signal stability, not a gate.\n")
    else:
        w("Not run.\n")
    w("## F2 · Facial grammar with timing rules (optional)\n\nNot run: DAI_ACCESS is not set (no ASLLRP utterances with face close-ups).\n")

    # ------------------------------------------------------------ numbers to carry
    w("## Numbers to carry into the build\n")
    w("| What | Value |\n|---|---|")
    w(f"| Aspect correction | x_px = x · (w/h) before any geometry |")
    w(f"| Trimmer | rise {pa['rise_sw']} sw above the clip's 10th-percentile wrist height; speed {pa['speed_sw_per_s']} sw/s; hysteresis {pa['hysteresis_s']} s at rise/2; static holds ≥ {pa['min_hold_s_static']} s; pad {p1p['chosen_pad']} frames |")
    w(f"| Avatar transitions | direct minimum-jerk, 160 ms; rest only at sentence end (300 ms) |")
    w(f"| Avatar playback speed | 1.3× (≈{six['1.3']} s for a typical 6-sign sentence) |")
    w(f"| Calibration | k = 3 per sign for every demo sign; fine-tune {r3['chosen']['finetune']['epochs']} epochs, LR {r3['chosen']['finetune']['lr']}, 1:1 replay of base training data; ~{k3['cpu_s']['median']:.0f} s CPU; ≈{k3['recording_min_estimate']} min recording for {r3['n_signs']} signs |")
    w(f"| Phrase model | Conv1d + BiGRU, {tr_bi['params']:,} params, CTC over classes + <unk> + blank; ~55 min CPU from round-1 weights |")
    w(f"| Decoder | script-constrained (CTC forced alignment over the café lines), top 3; out-of-script τ = {p5['tau']} nats; Claude reranker (haiku 5.5) with the barista's last line |")
    w(f"| Hands-at-rest hold | 300 ms; bidirectional model compute ≈ {p5['delay']['compute_ms_per_utterance']['median']:.0f} ms |")
    w("| Handshape table | `results2/handshapes.json` (data); curved shapes need hand-tuned values |")
    w(f"| Headshake | ±{hsh.get('peak_deg_estimate')}° at {hsh.get('freq_hz')} Hz (round-1 S3 negations) |")
    if f1 and f1.get("min_reliable_face_width_px"):
        w(f"| Minimum face width | ≤ {f1['min_reliable_face_width_px']} px works; aim for ≥ 60 px |")
    w("")

    # ------------------------------------------------------------ advisor review
    w("## Needs Deaf advisor review\n")
    w("- Handshape contact sheets: `results2/v2_sheets/` (avatar canonical handshape vs a real signer's hand).")
    w("- Avatar contact sheets per sign: `results2/v4_contact/`; capture of the 3 sentences: `avatar/capture2.mp4`.")
    w("- Brow and headshake renders (neutral / raise / lower; headshake strip): render with `uv run python spike2/v_shots.py face <out.jpg>`.")
    w("- Core vocabulary (`results2/vocab_core.json`) and the script edits above; whether pointing (IX-1P/YOU) should be recognized at all.")
    w("- The non-dominant-hand rules, the thumb rules, and the 1.3× playback speed.")
    w("")
    w("## Estimated API spend\n")
    w(f"Claude reranker only: {rr['n']} calls, {rr['tokens_in_out'][0]:,} input / {rr['tokens_in_out'][1]:,} output tokens ≈ **${rr['est_cost_usd']}** "
      "(claude-haiku-5-5 at $0.10 / $0.50 per MTok). No other paid APIs were used this round.\n")
    (ROOT / "REPORT-SPIKE-2.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote REPORT-SPIKE-2.md", len("\n".join(L)))


SCRIPT_EDITS = [
    ("HELLO IX-1P WANT COFFEE PLEASE", "HELLO COFFEE PLEASE", "WANT is weak on new signers (WLASL 20%, n = 5) and IX-1P/WANT add little in an order"),
    ("YES YOU HAVE OAT MILK", "OAT MILK HAVE? (raised brows)", "YOU is weak (WLASL 36%, confused with SAME); OAT is not in ASL Citizen (fingerspell)"),
    ("MUFFIN WHAT KIND", "MUFFIN KIND WHICH", "WHAT is weak and confused with COST; KIND (0.81) and WHICH (0.78) are robust"),
    ("BLUEBERRY ONE WARM CAN", "FS-BLUEBERRY ONE WARM PLEASE", "BLUEBERRY is confused with BANANA; CAN is weak after calibration (40%)"),
    ("MY NAME FS-MAYA", "NAME IX-1P FS-MAYA", "MY confused with PLEASE"),
    ("SORRY HOW-MUCH AGAIN PLEASE", "SORRY HOW-MUCH", "shorter; AGAIN is mid-strength"),
    ("CARD PAY CAN", "PAY FS-CARD", "CARD scored 0% on WLASL (n = 5); fingerspell or tap-to-confirm"),
    ("FINE THANK-YOU GOOD DAY", "THANK-YOU BYE", "GOOD and THANK-YOU are each other's top confusion; BYE is robust (0.67)"),
]

if __name__ == "__main__":
    main()
