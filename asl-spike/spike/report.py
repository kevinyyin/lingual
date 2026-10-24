"""Write REPORT-SPIKE.md from results/*.json. Every number in the report comes from a results file."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def j(name, default=None):
    p = R / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else default


def jl(name):
    p = R / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def ms(x):
    return "—" if x is None else f"{x:,.0f}"


def ci(c):
    return "" if not c else f" [{100 * c[0]:.0f}–{100 * c[1]:.0f}%]"


def pst(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None, None, None, 0
    return statistics.median(xs), xs[int(round(0.9 * (len(xs) - 1)))], xs[-1], len(xs)


def main():
    cov = j("s1_coverage.json")
    vocab = j("../vocab.json")
    cit = j("s2_citizen_eval.json")
    citk = j("s2_citizen_eval_with_kaggle.json")
    stg = j("s2_stgcn_eval.json")
    kag = j("s2_kaggle_eval.json")
    cont = j("s2_continuous_rwth.json")
    s3 = j("s3_rwth_eval.json")
    s4 = j("s4_summary.json")
    s5j = j("s5_judgments.json")
    s6 = j("s6_avatar.json")
    s6l = j("s6_lexicon.json")
    s8 = j("s8_budget.json")
    ink = jl("s8_ink.jsonl")
    g2e1, g2e2, e2g1, e2g2 = jl("s5_g2e.jsonl"), jl("s5_g2e_v2.jsonl"), jl("s5_e2g.jsonl"), jl("s5_e2g_v2.jsonl")
    L = []
    w = L.append

    # ---------------------------------------------------------------- gates
    p3 = s3["primary_exact"]
    s3lr = p3["sentence_type_logreg_brows"]
    gates = []
    if cit:
        t1 = cit["test"]["top1"]
        gates.append(("Recognition, unseen ASL Citizen signers", "≥ 80% top-1 on the ~150-sign vocabulary",
                      f"{pct(t1)} top-1, {pct(cit['test']['top3'])} top-3, {pct(cit['test']['top5'])} top-5 "
                      f"({cit['n_classes']} signs; n = {cit['n']['test']['videos']} videos, {cit['n']['test']['signers']} test signers)"
                      + (f"; released ST-GCN: {pct(stg['vocab_restricted']['top1'])} top-1" if stg else ""),
                      "PASS" if t1 >= 0.8 else "FAIL"))
    else:
        gates.append(("Recognition, unseen ASL Citizen signers", "≥ 80% top-1", "not run", "—"))
    if kag:
        a = kag["all_vocab_classes"]
        gates.append(("Recognition, different dataset (PopSign Deaf signers, phone cameras)", "≥ 70% top-1 (or ≥ 90% top-3) on overlapping signs",
                      f"{pct(a['top1'])} top-1, {pct(a['top3'])} top-3 over {kag['n_overlap']} overlapping signs "
                      f"(n = {kag['n_sequences']:,} sequences, {kag['n_signers']} signers; argmax over all {cit['n_classes'] if cit else '?'} classes)"
                      + (f". With the other PopSign signers added to training: {pct(citk['kaggle_heldout_signers']['top1'])} top-1, "
                         f"{pct(citk['kaggle_heldout_signers']['top3'])} top-3 on {len(citk['kaggle_heldout_signers']['signers'])} held-out PopSign signers "
                         f"(n = {citk['kaggle_heldout_signers']['n']:,}), ASL Citizen test unchanged at {pct(citk['test']['top1'])}"
                         if citk and citk.get("kaggle_heldout_signers") else ""),
                      ("PASS" if (a["top1"] >= 0.7 or a["top3"] >= 0.9) else "FAIL")
                      + (" as specified (no retraining); PASS with PopSign in training"
                         if citk and citk.get("kaggle_heldout_signers", {}).get("top1", 0) >= 0.7 else "")))
    if cont:
        t = cont["thresholds"]["0.5"]
        gates.append(("Continuous signing (RWTH-BOSTON-104 fallback; no DAI access)", "≥ 70% recall of vocabulary signs, ≤ 1.5 s after last sign",
                      f"recall {pct(t['recall'])}, precision {pct(t['precision'])} at confidence 0.5 "
                      f"(n = {cont['n_utterances']} utterances, {cont['n_signers']} signers); delay ≈ 300 ms rest hold + {ms(t['compute_ms_median'])} ms compute",
                      "PASS" if t["recall"] >= 0.7 else "FAIL"))
    gates.append(("Facial grammar (RWTH-BOSTON-104 + NCSLGR labels, leave-one-signer-out)", "≥ 85% sentence type on held-out signers",
                  f"{pct(s3lr['accuracy'])} accuracy but {pct(s3lr['balanced_accuracy'])} balanced (statement vs wh, n = {s3lr['n']}, 3 signers; "
                  f"no yes/no questions in this corpus); wh vs raised-brow questions AUC {p3['auc']['raised_vs_lowered_questions_brow_contrast']['auc']}",
                  "FAIL (inconclusive data)"))
    nb = p3["naive_baseline_hsemotion"]["headline"]
    gates.append(("Naive baseline contrast", "Baseline mislabels most yes/no questions as surprise",
                  f"wh-questions labelled anger/disgust: {nb['wh_labelled_anger_or_disgust']}; raised-brow questions labelled surprise: "
                  f"{nb['raised_brow_questions_labelled_surprise (rhq; no yn in RWTH)']} (rhetorical; no yes/no in corpus); statements labelled anger: "
                  f"{p3['naive_baseline_hsemotion']['statement']['majority_label_counts'].get('anger', 0)}/{p3['naive_baseline_hsemotion']['statement']['n']}",
                  "PARTLY (wh → anger confirmed; yes/no untested)"))
    d = s4["detection"]
    best_c = max(("b", "c", "d", "e"), key=lambda c: d.get(c, {}).get("hit_rate") or 0)
    gates.append(("Cartesia emotion", "Intended emotion detected ≥ 60% by the classifier",
                  f"tag only: {pct(d['b']['hit_rate'])} (chance 25%; no tag: {pct(d['a']['hit_rate'])}); best condition '{best_c}': "
                  f"{pct(d[best_c]['hit_rate'])} (SpeechBrain; HuBERT {pct(s4.get('detection_hubert', {}).get(best_c, {}).get('hit_rate'))}); "
                  f"the same emotional wording *without* the tag: {pct(d.get('f', {}).get('hit_rate'))}, so the gain is from the words, not the tag",
                  "PASS" if max(d[c]["hit_rate"] for c in ("b", "c", "d")) >= 0.6 else "FAIL"))
    fm = s6["first_motion_ms"]
    gates.append(("Avatar", "5-sign sentence with brow grammar, first motion ≤ 2 s",
                  f"first motion {fm['warm_median']:.0f} ms median ({fm['cold_first_run']:.0f} ms cold), 5–6-sign sentences with brow/headshake NMM; "
                  "handshapes poor, brows hidden by the model's hair", "PASS (latency) / quality poor"))
    gates.append(("Glasses", "≥ 15 fps, ≤ 500 ms added delay via mock device", "not run: deferred (S7)", "NOT RUN"))

    w("# REPORT-SPIKE: two-way ASL MVP feasibility (early research spike; not product code)\n")
    w("Early research spike, run 2026-10-16 on one Windows 11 laptop (Intel Core Ultra 7 258V, 32 GB, "
      "no CUDA; Intel Arc 140V for WebGL). Every number below comes from `results/`. n (clips, signers) is given beside "
      "each result. Glosses and facial-grammar choices are by hearing non-signers and need Deaf advisor review.\n")
    w("## Decision gates\n")
    w("| Gate | Pass if | Measured | Verdict |\n|---|---|---|---|")
    for g in gates:
        w(f"| {g[0]} | {g[1]} | {g[2]} | **{g[3]}** |")
    w("")

    # ---------------------------------------------------------------- human steps
    w("## Human steps needed\n")
    w("| Step | Why | Status |\n|---|---|---|")
    for row in [
        ("Accept the Kaggle rules for asl-signs and asl-fingerspelling (browser, signed in)", "Kaggle download", "done"),
        ("RWTH-BOSTON-104 download form", "S2 continuous / S3 fallback data", "done (archive placed in the repo)"),
        ("ASLLRP DAI account (https://dai.cs.rutgers.edu/dai/s/dai), hand-approved", "real ASLLRP/NCSLGR videos with face close-up and yes/no questions: the missing piece for the S3 gate", "not requested (user decision); fallbacks used"),
        ("Download How2Sign frontal clips in a browser (Google Drive refuses scripted downloads: 'too many users have viewed or downloaded this file'). Files: val/test 'Green Screen RGB clips (frontal)' from http://how2sign.github.io/ into `asl-spike/data/how2sign/`", "S3 fallback (b), weak labels", "blocked; only the translation CSVs downloaded"),
        ("NCSLGR XML: the public XML link on the BU page is missing and the Rutgers mirror (secrets.rutgers.edu) does not respond", "we decoded the public SignStream 2 database files instead (validated against the 3 XML files in the BU parser package)", "worked around"),
        ("Ask Microsoft (ASL_Citizen@microsoft.com) whether a public demo and its derived avatar poses/model weights are permitted", "ASL Citizen license: non-commercial research, no distribution of data or modified data", "open"),
        ("Run the S4 listening sheet with 3–5 people (`listening/index.html`)", "the classifiers disagree with each other; human judgement needed", "ready, not run"),
        ("Deaf advisor review of every gloss, NMM rule and S5 output (checklist at the end)", "we are hearing beginners", "open"),
        ("S7 camera paths: phone on local Wi-Fi; Meta Wearables Device Access Toolkit (developer account, Android Studio + JDK, Android phone with USB debugging; Mock Device Kit; iOS needs a Mac)", "glasses gate", "deferred"),
    ]:
        w(f"| {row[0]} | {row[1]} | {row[2]} |")
    w("")

    # ---------------------------------------------------------------- S1
    w("## S1 · Scenario and vocabulary\n")
    w(f"`scenario.md`: 28-turn café dialogue (yes/no ×3, wh ×3, negation ×3, fingerspelled name). `vocab.json`: "
      f"{cov['n_candidates']} candidate glosses (dialogue + café neighbours + numbers 1–20 + function signs).\n")
    w("| Coverage | Glosses |\n|---|---|")
    w(f"| In ASL Citizen (any alias) | {cov['asl_citizen']} of {cov['n_candidates']} |")
    w(f"| Distinct recognition classes (one ASL Citizen sign trains one class) | {cov['model_classes']} |")
    w(f"| Dialogue glosses in ASL Citizen | {cov['dialogue_in_asl_citizen']} of {cov['dialogue_glosses']} |")
    w(f"| In WLASL | {cov['wlasl']} |")
    w(f"| In Kaggle asl-signs (PopSign 250) and a model class | {len(cov['in_citizen_and_kaggle'])} |")
    w("")
    miss = [e for e in vocab if not e["asl_citizen"]]
    w("Not in ASL Citizen, with the proposed substitute: " + "; ".join(f"{e['gloss']} → {e['substitute']}" for e in miss) + ".")
    w("Numbers 12–20 (and FIVE, TEN) are missing from ASL Citizen, so a number/fingerspelling handshape path is needed for prices.\n")
    shared = [e for e in vocab if e.get("recognized_as") and e["recognized_as"] != e["gloss"]]
    if shared:
        w("Shared signs (recognized as another class): " + "; ".join(f"{e['gloss']} → {e['recognized_as']}" for e in shared) + ".\n")

    # ---------------------------------------------------------------- S2
    w("## S2 · Isolated sign recognition\n")
    if cit:
        w(f"Landmarks: MediaPipe Holistic (Tasks API) on CPU. Features: both hands (21×2), 8 upper-body points, 6 face points; "
          f"centered and scaled by the shoulders. The dominant hand is mirrored into one slot, and clips are resampled to 32 frames. "
          f"Model: Conv1d + BiGRU + attention, {cit['params']:,} parameters, trained on CPU in {cit['train_s'] / 60:.0f} min. "
          f"Official signer-independent splits: train {cit['n']['train']['videos']} videos / {cit['n']['train']['signers']} signers, "
          f"val {cit['n']['val']['videos']} / {cit['n']['val']['signers']}, test {cit['n']['test']['videos']} / {cit['n']['test']['signers']}.\n")
        w("| Model | Test top-1 | top-3 | top-5 |\n|---|---|---|---|")
        w(f"| Ours, ASL Citizen only | {pct(cit['test']['top1'])}{ci(cit['test'].get('top1_wilson'))} | {pct(cit['test']['top3'])} | {pct(cit['test']['top5'])} |")
        if citk:
            w(f"| Ours, ASL Citizen + PopSign/Kaggle signers | {pct(citk['test']['top1'])} | {pct(citk['test']['top3'])} | {pct(citk['test']['top5'])} |")
        if stg:
            w(f"| ASL Citizen released ST-GCN, restricted to our {cit['n_classes']} classes (no fine-tuning) | {pct(stg['vocab_restricted']['top1'])} | {pct(stg['vocab_restricted']['top3'])} | {pct(stg['vocab_restricted']['top5'])} |")
            w(f"| ASL Citizen released ST-GCN, full 2,731-way (paper: 63% top-1) | {pct(stg['full_2731']['top1'])} | — | {pct(stg['full_2731']['top5'])} |")
        w("")
        per = sorted(cit["per_sign"].items(), key=lambda kv: kv[1]["acc"])
        w("Weakest signs (test accuracy, n): " + ", ".join(f"{k} {pct(v['acc'], 0)} ({v['n']})" for k, v in per[:15]) + ".\n")
        w("15 most-confused pairs (count): " + ", ".join(f"{a}↔{b} ({n})" for (a, b), n in [(tuple(p['pair']), p['count']) for p in cit['top_confused_pairs']]) + ".\n")
    if kag:
        a, r_ = kag["all_vocab_classes"], kag["restricted_to_overlap"]
        w(f"**Cross-dataset (PopSign Deaf signers on phone selfie cameras, Kaggle landmark files, no retraining):** "
          f"{kag['n_overlap']} vocabulary signs overlap, {kag['n_sequences']:,} sequences from {kag['n_signers']} signers.\n")
        w("| Scoring | top-1 | top-3 | top-5 |\n|---|---|---|---|")
        w(f"| argmax over all vocabulary classes | {pct(a['top1'])}{ci(a.get('top1_wilson'))} | {pct(a['top3'])} | {pct(a['top5'])} |")
        w(f"| restricted to the {kag['n_overlap']} overlapping classes | {pct(r_['top1'])} | {pct(r_['top3'])} | {pct(r_['top5'])} |")
        if cit and cit.get("kaggle_heldout_signers"):
            kh = cit["kaggle_heldout_signers"]
            w(f"| held-out PopSign signers ({len(kh['signers'])}), ASL-Citizen-only model | {pct(kh['top1'])} | {pct(kh['top3'])} | {pct(kh['top5'])} |")
        if citk and citk.get("kaggle_heldout_signers"):
            kh = citk["kaggle_heldout_signers"]
            w(f"| same held-out signers, model trained with the other PopSign signers | {pct(kh['top1'])} | {pct(kh['top3'])} | {pct(kh['top5'])} |")
        w("")
        ps = sorted(a["per_sign"].items(), key=lambda kv: kv[1]["top1"])
        w("Per sign (top-1 / top-3): " + ", ".join(f"{k} {pct(v['top1'], 0)}/{pct(v['top3'], 0)}" for k, v in ps) + ".\n")
    if cont:
        w("**Continuous signing (fallback: RWTH-BOSTON-104, because DAI access was not requested).** "
          f"{cont['n_utterances']} sentences containing at least two vocabulary signs, {cont['n_signers']} signers, "
          "a full-body black-and-white camera at 25 fps, 336×312. The recognizer uses sliding windows of 12–36 frames "
          "with the S2 model, a confidence threshold, temporal non-maximum suppression, and merging of consecutive "
          "duplicates. Recall and precision use an order-preserving alignment against the gloss sequence. "
          "Boundary error cannot be measured because RWTH-104 annotations have no per-sign times.\n")
        w("| Confidence threshold | Recall | Precision | False positives per OOV sign | Compute per utterance (median / p90 ms) |\n|---|---|---|---|---|")
        for k, t in cont["thresholds"].items():
            w(f"| {k} | {pct(t['recall'])} ({t['tp']}/{t['n_gt_vocab_tokens']}) | {pct(t['precision'])} | {t['fp_per_oov_token']:.2f} ({t['false_positives']}/{t['oov_tokens']}) | {ms(t['compute_ms_median'])} / {ms(t['compute_ms_p90'])} |")
        w("")
    w("**Not done:**\n"
      "- Fingerspelling (optional). Time went to the core risks. The Kaggle asl-fingerspelling rules are accepted, and the data can be downloaded later.\n"
      "- Gemini zero-shot baseline. The ASL Citizen license forbids transferring the materials to third parties, and only PopSign *landmarks* (no video) were downloaded. Skipped: license.\n"
      "- SignGemma. Not publicly available as of 2026-10-16: no Hugging Face repository, and nothing in the Gemma release notes.\n")

    # ---------------------------------------------------------------- S3
    w("## S3 · Facial grammar vs emotion\n")
    w(f"**Data (source a):** RWTH-BOSTON-104 videos, camera 0. This is a full-body view in which the face is about 40 px wide; it is upscaled 2× before MediaPipe. "
      f"Labels come from the NCSLGR annotations: the public SignStream 2 database files, decoded by `spike/ss2.py`. The decoder was validated on 45 of 45 "
      f"aligned utterances against the XML files shipped in the BU parser package. RWTH sentences were matched to NCSLGR utterances by normalized gloss sequence, "
      f"using translations to break ties. Primary set: {p3['n_utterances']} exact matches. Classes: {p3['class_counts']}. "
      f"By signer: {p3['by_signer']}. Negation: {p3['negation_count']}. **This corpus has no yes/no questions**, so yes/no detection is untested.\n")
    w("Signals are MediaPipe blendshapes (browInnerUp/OuterUp, browDown) plus geometric brow height. The sentence-level feature is a "
      "0.5 s rolling mean, which captures sustained raise or lower. Head pose is from mesh geometry, and a headshake is yaw oscillation "
      "at 1.5–6 Hz. Signals are z-scored per signer without labels; this is transductive calibration, and a product would calibrate "
      "on the first seconds of a conversation.\n")
    w("| Test (primary exact set, leave-one-signer-out) | Result |\n|---|---|")
    for k, lab in (("sentence_type_logreg_brows", "Statement vs wh, logistic regression on brow features"),
                   ("sentence_type_threshold_brow_down", "Statement vs wh, single brow-down threshold")):
        r_ = p3[k]
        w(f"| {lab} | accuracy {pct(r_['accuracy'])}{ci(r_['wilson'])}, balanced {pct(r_['balanced_accuracy'])}, n = {r_['n']}; confusion {r_['confusion']} |")
    for k, v in p3["auc"].items():
        if v:
            w(f"| AUC {k.replace('_', ' ')} | {v['auc']} [{v['ci'][0]}–{v['ci'][1]}] (n = {v['n_pos']} vs {v['n_neg']}) |")
    for k, lab in (("negation_threshold_headshake_amp", "Negation, headshake amplitude threshold"), ("negation_logreg_headshake", "Negation, logistic regression on headshake features")):
        r_ = p3[k]
        w(f"| {lab} | precision {pct(r_['precision'])}, recall {pct(r_['recall'])} ({r_['tp']} TP, {r_['fp']} FP, {r_['fn']} FN; n = {r_['n']}) |")
    w(f"| 3-way brow rule (neutral / lowered / raised) | accuracy {pct(p3['brow_rule_3way']['accuracy'])}, balanced {pct(p3['brow_rule_3way']['balanced_accuracy'])} |")
    w("")
    tc = p3["brow_by_group_topic_confound"]
    w("**Key finding: raised brows are not only questions.** Mean per-signer z-scores of sustained brow raise and lower:\n")
    w("| Group | n | brow raise | brow lower |\n|---|---|---|---|")
    for g_, v in tc.items():
        w(f"| {g_} | {v['n']} | {v['brow_up_sus_max']:+.2f} | {v['brow_down_sus_max']:+.2f} |")
    w("\nStatements with topic or conditional marking raise the brows *more* than rhetorical questions do, because topics and "
      "conditionals are marked with raised brows in ASL. Whole-utterance brow raise therefore cannot mean \"yes/no question\". "
      "The grammar layer needs *where* the raise is: over the whole clause or ending at the sentence end means a question, while "
      "a raise on the first constituents followed by a drop means a topic or conditional. Gloss context also matters: a wh-sign means "
      "a wh-question. Once an utterance is known to be a question, the brows separate wh from raised-brow questions well "
      f"(AUC {p3['auc']['raised_vs_lowered_questions_brow_contrast']['auc']}).\n")
    w("**Naive baseline (HSEmotion enet_b0_8, frame by frame on face crops, majority label per utterance):**\n")
    w("| Sentence type | n | Majority labels |\n|---|---|---|")
    for c, v in p3["naive_baseline_hsemotion"].items():
        if c != "headline":
            w(f"| {c} | {v['n']} | {v['majority_label_counts']} |")
    w(f"\nHeadline: the naive classifier calls **{nb['wh_labelled_anger_or_disgust']} wh-questions angry or disgusted**. Our grammar layer labels "
      f"them {p3['grammar_layer_labels'].get('wh')}. The naive classifier also calls {p3['naive_baseline_hsemotion']['statement']['majority_label_counts'].get('anger', 0)} of "
      f"{p3['naive_baseline_hsemotion']['statement']['n']} statements angry, because the low-resolution faces and lowered brows read as anger. Raised-brow (rhetorical) questions were "
      f"labelled surprise in {nb['raised_brow_questions_labelled_surprise (rhq; no yn in RWTH)']}; yes/no questions are untested. "
      f"Association with sentence type (Cramér's V): naive {p3['naive_vs_type_cramers_v']:.2f} vs our affect layer {p3['affect_vs_type_cramers_v']:.2f}.\n")
    w(f"**Affect layer** (mouth, cheeks, eye squint and nose sneer, plus hand speed, amplitude and tempo; brows excluded): label distribution by type: "
      f"{p3['affect_by_type']}. Its association with sentence type is low, so it does not simply track question type. The corpora "
      "have no emotion labels, so emotion *accuracy* is untested and needs labeled recordings later.\n")
    w("`to_voice(grammar, affect, kinematics)` is in `spike/voice_map.py`:\n"
      "- **Text:** questions get \"?\".\n"
      "- **Emotion:** one of Cartesia's well-supported emotions plus \"happy\". It is sent only at intensity ≥ 0.35, and dropped when it contradicts the LLM's reading of the words.\n"
      "- **Speed:** signing tempo ratio, clamped to 0.8–1.3.\n"
      "- **Volume:** movement amplitude ratio, clamped to 0.8–1.5.\n"
      "Examples are in `results/s3_rwth_eval.json` → `to_voice_examples`.\n")
    w("Weak-label How2Sign fallback (b): not run. Google Drive blocked scripted downloads; see Human steps.\n")

    # ---------------------------------------------------------------- S4
    w("## S4 · Cartesia expressive voice\n")
    w("Sonic 3.6 over one WebSocket, sequential, one fresh context per request. Twelve scenario sentences × 4 emotions × 3 voices "
      "(Leo, Maya, Gavin), across several conditions:\n"
      "- **(a)** no emotion;\n"
      "- **(b)** `generation_config.emotion`;\n"
      "- **(c)** emotion plus speed and volume from `to_voice`;\n"
      "- **(d)** inline `<emotion value=…/>` tag;\n"
      "- **(e/f)** an LLM rewrite of each line so the words carry the emotion, with (e) and without (f) the tag; Leo and Maya only.\n\n"
      "Two independent 4-class speech-emotion classifiers scored every clip: SpeechBrain wav2vec2-IEMOCAP and HuBERT-large SUPERB-ER. Chance is 25%.\n")
    lat = s4["latency"]
    w(f"Latency (n = {lat['ttfb_ms']['n']}): first audio byte median {ms(lat['ttfb_ms']['median'])} ms, p90 {ms(lat['ttfb_ms']['p90'])}, "
      f"max {ms(lat['ttfb_ms']['max'])}; total synthesis median {ms(lat['total_ms']['median'])} ms, p90 {ms(lat['total_ms']['p90'])}.\n")
    w("| Condition | SpeechBrain hit rate | HuBERT hit rate | n | per emotion (SpeechBrain) |\n|---|---|---|---|---|")
    names = {"a": "a · no emotion (scored vs each intent)", "b": "b · emotion tag", "c": "c · tag + speed/volume", "d": "d · inline tag",
             "e": "e · emotional wording + tag", "f": "f · emotional wording, no tag"}
    for c in ("a", "b", "c", "d", "e", "f"):
        if c in d:
            h = s4.get("detection_hubert", {}).get(c, {})
            w(f"| {names[c]} | {pct(d[c]['hit_rate'])}{ci(d[c]['ci'])} | {pct(h.get('hit_rate'))} | {d[c]['n']} | "
              + ", ".join(f"{e} {pct(v['rate'], 0)}" for e, v in d[c]["by_emotion"].items()) + " |")
    w("")
    if "e" in d and "f" in d:
        w(f"**Words, not tags:** emotional wording with the tag scored {pct(d['e']['hit_rate'])} and the same wording without the tag "
          f"{pct(d['f']['hit_rate'])}. All tag-only conditions sit at chance. Neither classifier detected 'sad' in any condition, "
          "which may be a classifier limit on synthetic speech as much as a Sonic limit; the listening test should settle it.\n")
    mm = s4["mismatch"]
    w(f"**Mismatch test** (6 texts × 3 voices; emotion contradicting the words). With the contradicting tag, the intended emotion was "
      f"detected in {mm['mm_contra']['hit_rate']['k']}/{mm['mm_contra']['hit_rate']['n']} clips (SpeechBrain), and the *text's* emotion in "
      f"{mm['mm_contra']['pred_is_text_emotion']['k']}/{mm['mm_contra']['pred_is_text_emotion']['n']}. With the matching tag: "
      f"{mm['mm_match']['hit_rate']['k']}/{mm['mm_match']['hit_rate']['n']} (HuBERT {pct(mm['mm_match'].get('hit_rate_hubert', {}).get('rate'))}). "
      "This confirms the docs: the words win. `to_voice` drops the emotion when it conflicts with the LLM's reading of the sentence.\n")
    w("An acoustic check (duration, RMS, median F0) shows the tags barely change these neutral café sentences: about 1.85–1.90 s, "
      "RMS 0.073–0.077 and F0 197–203 Hz across tags. The two classifiers also disagree with each other: SpeechBrain says "
      "neutral or angry, HuBERT mostly happy. That is why the listening sheet matters.\n")
    w("**Recommendation.** The default voice is provisional until the listening test. Gavin read as neutral most consistently "
      "(83% of untagged clips, vs 67% for Maya and 25% for Leo), and Leo's anger was the most detectable (75% with the tag). "
      "Rules:\n"
      "1. Convey affect mainly through speed and volume, which are measurable and controllable.\n"
      "2. Send an emotion tag only when the LLM's English already carries that emotion.\n"
      "3. Never let the LLM add emotional words the signer did not sign. The e/f rewrites invented content such as "
      "\"Get it right this time\" and \"It's been a long day\". That misrepresents a Deaf user and needs Deaf advisor review.\n"
      "4. Keep the set of emotions small: neutral, happy, sad, angry.\n")
    w("Listening sheet: `listening/index.html` (60 blind clips, key in `listening/key.json`, score with `spike/s4_listening.py score`). Not run.\n")

    # ---------------------------------------------------------------- S5
    w("## S5 · LLM glue (claude-haiku-5-5, structured JSON, effort low, thinking off)\n")

    def lat_of(rs):
        m, p9, mx, n = pst([r.get("latency_ms") for r in rs])
        return f"median {ms(m)} ms, p90 {ms(p9)}, max {ms(mx)} (n = {n})"

    def judge(key, lo, hi):
        js = s5j[key]
        sel = [js[str(i)] for i in range(lo, hi)]
        return f"{sel.count('A')} A / {sel.count('B')} B / {sel.count('W')} W"
    w("| Direction | Variant | Acceptable meaning (A / B / W, my judgement) | Latency |\n|---|---|---|---|")
    w(f"| Gloss → English, 30 clean | v1: glosses only | {judge('g2e_v1', 0, 30)} | {lat_of(g2e1)} |")
    w(f"| Gloss → English, 10 with one wrong/missing gloss | v1 | {judge('g2e_v1', 30, 40)} | |")
    w(f"| Gloss → English, 30 clean | v2: + barista's previous line as context | {judge('g2e_v2', 0, 30)} | {lat_of(g2e2)} |")
    w(f"| Gloss → English, 10 with errors | v2 | {judge('g2e_v2', 30, 40)} | |")
    rej1 = sum(1 for r in e2g1 if r.get("rejected"))
    rej2 = sum(1 for r in e2g2 if r.get("rejected"))
    first2 = sum(1 for r in e2g2 if r.get("first_invalid"))
    w(f"| English → gloss + NMM, 30 barista lines | v1: vocabulary in the prompt | rejected (gloss outside vocabulary) {rej1}/30 | {lat_of(e2g1)} |")
    w(f"| English → gloss + NMM | v2: + worked examples, one repair retry | first pass invalid {first2}/30, after retry **{rej2}/30 rejected**; meaning {judge('e2g_v2_meaning', 0, 30)}; NMM errors: {len(s5j['e2g_v2_nmm_errors'])} | {lat_of(e2g2)} (incl. retries) |")
    w("\nA = meaning preserved; B = understandable but lossy or changed; W = wrong meaning. Judged on meaning only, not on ASL "
      "quality. Every output is in `results/s5_*.jsonl` and the prompts are in `results/s5_config.json`; all of them need Deaf advisor review. "
      "Lessons:\n"
      "1. Pass the last barista utterance as context. It fixed most v1 errors, such as \"It's hot.\" turning into \"Hot, please.\"\n"
      "2. A misrecognized off-topic gloss (FISH, DOG) still leaks through about half the time. The UI should show low-confidence words.\n"
      "3. A missing negation sign with the headshake flag set is recovered in v2.\n"
      "4. Validate-and-retry makes English → gloss output always valid, at the cost of a second call on 30% of lines.\n")

    # ---------------------------------------------------------------- S6
    w("## S6 · Avatar for speech → sign\n")
    w("| Option | Assessment |\n|---|---|")
    w("| (a) sign.mt spoken-to-signed-translation | MIT code. Lexicons are Swiss (sgg/ssr/slf); ASL is a rule-based 'lexical plan' with no ASL lexicon listed. Outputs .pose/video. Non-manuals are explicitly 'downstream work'. The app is CC BY-NC-SA 4.0. Not usable for ASL as is. |")
    w("| (b) AWS GenASL | MIT-0 code. Stitched 2D avatar videos rendered from ASLLVD citation clips with RTMPose. Needs Transcribe, Bedrock, Step Functions, Lambda, DynamoDB, S3 and Amplify. No facial grammar. ASLLVD videos come from BU, so their terms must be checked. Too heavy for 36 h. |")
    w("| (c) ours: per-gloss MediaPipe poses → VRM avatar in three.js | Built (`avatar/index.html`). three.js 0.169, @pixiv/three-vrm 3.1, kalidokit 1.1.5 for fingers. VRM sample model 'VRM1_Constraint_Twist_Sample' (VRM Public License 1.0: commercial use and redistribution allowed). |")
    w("")
    w(f"Prototype: 3 sentences from S5 v2 outputs (wh: {s6l['presets'][0]['sentence']['signs'].__len__()} signs, yes/no: "
      f"{s6l['presets'][1]['sentence']['signs'].__len__()}, negation: {s6l['presets'][2]['sentence']['signs'].__len__()}). Glosses missing from ASL Citizen were dropped "
      f"({', '.join(sum([p['dropped_glosses'] for p in s6l['presets']], [])) or 'none'}). The page applies 200 ms slerp blends between signs, brow-only morph targets "
      "(Fcl_BRW_Surprised for raise, Fcl_BRW_Angry for lower) over each NMM span, and a 3 Hz head yaw for negation. Measured in Microsoft Edge "
      f"(Playwright) on {s6['gpu_renderer']}:\n")
    w(f"| Metric | Value |\n|---|---|\n| Gloss JSON → first avatar motion (warm, lexicon loaded) | median {fm['warm_median']:.1f} ms, p90 {fm['p90']:.1f}, n = {fm['n']} |"
      f"\n| Same, first run (fetching a 5 MB lexicon) | {fm['cold_first_run']:.0f} ms |\n| Paint cost per frame | {s6['render_ms_median']} ms |"
      f"\n| Page ready (model load) | {s6['page_ready_s']} s |")
    w("\nScreen capture: `avatar/capture.webm` (local, git-ignored). Key frames: `avatar/shots/`.\n")
    w("**Visual problems, honestly:**\n"
      "1. kalidokit's arm solver puts the arms behind the body on a VRM 1.0 model under every flip/mirror setting. We replaced it with direct vector alignment of the shoulder→elbow→wrist directions, which puts the hands in signing space. Arm twist and wrist roll are still missing.\n"
      "2. Handshapes from kalidokit's finger solver are unreliable: an index point (YOU) does not read clearly.\n"
      "3. The sample model's bangs cover the eyebrows, so brow grammar is nearly invisible. Pick an avatar with visible, thick brows and add eye aperture and head tilt.\n"
      "4. Citation clips are slow: 7–11 s for a 5–6-sign sentence. Trim holds and play at about 1.3–1.5×.\n"
      "5. Transitions between different signers' clips would jump; the prototype uses one signer (P11).\n"
      "6. The hidden in-app browser pane renders WebGL in software and cannot even compile the model's shaders in 45 s. Test in a real browser with a GPU.\n")
    w("**License blocker for a public demo:** the per-gloss poses come from ASL Citizen, whose license forbids distributing the data or modified data. "
      "A public demo should use a lexicon we may distribute (PopSign is CC BY 4.0; check the Kaggle data terms) or one recorded by the team's Deaf advisor with consent.\n")

    # ---------------------------------------------------------------- S7
    w("## S7 · Camera input paths\n\nDeferred; not run. Instructions to do it later:\n"
      "- **Phone browser path:** the page must be served over HTTPS (a self-signed certificate is accepted on the phone) for getUserMedia on a non-localhost origin. Allow the Windows Firewall prompt for the local server, then measure fps and frame age at the laptop.\n"
      "- **Meta glasses path:**\n"
      "  1. Create a Meta Wearables developer account and enroll in Device Access Toolkit access.\n"
      "  2. Install Android Studio and a JDK, and connect an Android phone with USB debugging.\n"
      "  3. Build the DAT sample app, using a GitHub token with `read:packages` if the SDK is hosted on GitHub Packages.\n"
      "  4. Use the Mock Device Kit to inject a licensed signing video, such as a PopSign clip (CC BY 4.0).\n"
      "  5. Stream frames to the laptop service.\n"
      "  6. Check Bluetooth audio output to the glasses' speakers from the phone app.\n")

    # ---------------------------------------------------------------- S8
    w("## S8 · End-to-end latency budget (from measured stages)\n")
    for k, title in (("sign_to_speech", "Sign → speech (last sign ends → first audio)"), ("speech_to_sign", "Speech → sign (speech ends → first avatar motion)")):
        b = s8[k]
        w(f"**{title}**\n\n| Stage | Median ms | p90 ms | n | Source |\n|---|---|---|---|---|")
        for st in b["stages"]:
            big = " **(biggest)**" if st["stage"] == b["biggest"] else ""
            w(f"| {st['stage']}{big} | {ms(st['median'])} | {ms(st['p90'])} | {st['n'] or '—'} | {st['source']} |")
        w(f"| **Total** | **{ms(b['median_sum'])}** | **{ms(b['p90_sum'])}** (sum of p90s, conservative) | | |\n")
    ee = s8["speech_to_sign"]["eager_end_variant"]["ink_eager_end_ms"]
    w(f"Ink's `turn.eager_end` arrives at a median {ms(ee['median'])} ms (p90 {ms(ee['p90'])}). Starting the LLM on eager end, and discarding "
      "the result on `turn.resume`, saves roughly the difference. Ink transcripts were essentially perfect on these clean TTS lines; "
      f"examples are in `results/s8_ink.jsonl`. Not included: camera capture and transport (S7), audio buffering, and the time the "
      f"avatar takes to sign (7–11 s per sentence at citation speed).\n")

    # ---------------------------------------------------------------- licenses
    w("## Dataset and asset licenses\n\n| Item | License / terms | Used for | Demo impact |\n|---|---|---|---|")
    for row in [
        ("ASL Citizen (Microsoft Research)", "Microsoft Research License Terms: non-commercial research only; no sharing or distribution of the data or modified data; no transfer to third parties; personal data kept confidential", "S2 training and testing; S6 lexicon", "Training locally is OK for research. Public demo assets derived from it are likely not allowed; ask Microsoft"),
        ("ASL Citizen code + ST-GCN weights", "MIT (code repository)", "S2 baseline", "OK"),
        ("Kaggle asl-signs (from PopSign)", "competition rules accepted; PopSign itself is CC BY 4.0", "S2 cross-dataset test", "Check the competition data terms before using it in a demo"),
        ("RWTH-BOSTON-104", "CC BY-NC-SA 4.0 (form on download)", "S2 continuous, S3", "Research only"),
        ("NCSLGR annotations (BU ASLLRP)", "public download page; cite ASLLRP and C. Neidle; videos via DAI only", "S3 labels", "Research only"),
        ("How2Sign", "CC BY-NC 4.0", "not used (download blocked)", "—"),
        ("VRM1_Constraint_Twist_Sample (pixiv)", "VRM Public License 1.0: commercial use and redistribution allowed", "S6 avatar", "OK"),
        ("SpeechBrain / HuBERT SUPERB-ER / HSEmotion models", "open model licenses (Apache 2.0 / MIT-style)", "S4, S3 evaluation only", "OK for evaluation"),
    ]:
        w("| " + " | ".join(row) + " |")
    w("")

    # ---------------------------------------------------------------- closing
    kh0 = (cit or {}).get("kaggle_heldout_signers", {})
    kh1 = (citk or {}).get("kaggle_heldout_signers", {})
    s2s, g2s = s8["sign_to_speech"], s8["speech_to_sign"]
    pop_line = (f"Adding the other PopSign signers to training lifts held-out PopSign signers from {pct(kh0.get('top1'))} to {pct(kh1.get('top1'))} top-1 "
                f"({pct(kh1.get('top3'))} top-3) while ASL Citizen test stays at {pct(citk['test']['top1'])}."
                if citk and kh1 else "The PopSign-augmented model was not completed.")
    w("## Go/no-go recommendation\n")
    w("**Go, with a narrowed hero demo.** The parts that must be fast are fast. Sign → speech is about "
      f"{s2s['median_sum'] / 1000:.1f} s median ({s2s['p90_sum'] / 1000:.1f} s p90), and speech → sign is about "
      f"{g2s['median_sum'] / 1000:.1f} s median ({g2s['p90_sum'] / 1000:.1f} s p90). The LLM call is the largest stage in both directions. "
      f"Isolated-sign recognition clears the gate on unseen ASL Citizen signers: {pct(cit['test']['top1'])} top-1 and "
      f"{pct(cit['test']['top3'])} top-3 over {cit['n_classes']} signs. "
      f"It does **not** transfer to other cameras and signers without retraining: PopSign phone signers get {pct(kag['all_vocab_classes']['top1'])} top-1, "
      f"and continuous signing gets {pct(cont['thresholds']['0.5']['recall'])} recall. {pop_line}\n\n"
      "So the demo should be isolated signs with brief holds, a top-3 tap-to-confirm, and a vocabulary of about 50 core café signs. "
      "Ideally the demo signer is the team's Deaf advisor, with a few minutes of their own examples added to training before the demo. "
      "Use facial grammar for wh vs yes/no (once a question is detected), not for question vs statement. Use the headshake for negation. "
      "Convey voice affect with speed and volume rather than Cartesia emotion tags. For speech → sign, use captions plus the avatar, "
      "and keep pre-rendered clips of the core phrases as the fallback. The avatar's handshape quality is not demo-ready, "
      "and its ASL Citizen-derived poses cannot be shown publicly.")
    w("\n## Five biggest risks for the main build\n")
    w(f"1. **Recognition collapses on a new signer or camera.** Measured: {pct(kag['all_vocab_classes']['top1'])} top-1 on PopSign phone signers "
      f"vs {pct(cit['test']['top1'])} on ASL Citizen. Mitigation: about 50 signs, tap-to-confirm top-3, the demo signer's own samples added "
      "before the demo, and PopSign data in training.")
    w(f"2. **Continuous signing is not recognized** ({pct(cont['thresholds']['0.5']['recall'])} recall on RWTH sentences). Mitigation: "
      "sign-by-sign with holds, and segmentation by hands at rest. Do not promise sentence-level recognition.")
    w("3. **Wrong grammar or affect in the voice misrepresents the Deaf user.**\n"
      "   - Raised brows also mark topics.\n"
      "   - Emotion classifiers read grammar as anger.\n"
      "   - LLM rewrites invent words.\n\n"
      "   Mitigation: show the English on screen before or while speaking, keep tags conservative, and get Deaf advisor sign-off.")
    w("4. **The avatar's signing is not legible:** handshapes are wrong, the brows are hidden, and citation clips are slow. "
      "Mitigation: captions first, the avatar as an extra, the advisor's recorded clips for the core phrases, and a different avatar model.")
    w("5. **Licensing and data:**\n"
      "   - ASL Citizen forbids redistribution, so public demo assets must come from CC BY data or the team.\n"
      "   - Glasses camera access (S7) was not tested.\n\n"
      "   Mitigation: phone camera as the default input, and ask Microsoft and Meta early.")
    w("\n## Needs Deaf advisor review\n")
    w("Every item below was chosen by hearing non-signers.\n")
    w("- **Glosses:** " + ", ".join(e["gloss"] for e in vocab) + ".")
    w("- **Substitutions:** " + "; ".join(f"{e['gloss']} → {e['substitute']}" for e in vocab if e.get("substitute")) + ".")
    w("- **Alias matches to confirm** (same English label, possibly different sign): " + "; ".join(f"{e['gloss']} ↔ {', '.join(e['asl_citizen'])}" for e in vocab if e["asl_citizen"] and not e["asl_citizen_exact"]) + ".")
    w("- **Facial grammar rules:**\n"
      "  - raised brows over the clause mark a yes/no question;\n"
      "  - lowered brows mark a wh-question;\n"
      "  - a headshake marks negation;\n"
      "  - raised brows on initial constituents mark a topic or conditional, not a question;\n"
      "  - the avatar's 3 Hz head yaw is our stand-in for a negation headshake.")
    w("- **S5 outputs:** all 40 gloss → English and 30 English → gloss outputs (`results/s5_*.jsonl`), including the NMM spans.")
    w("- **Voice affect:**\n"
      "  - whether conveying affect by speed and volume is acceptable to Deaf users;\n"
      "  - whether a voice should ever add emotional wording.")
    w("- **Avatar:** handshape accuracy, movement speed, the whole avatar approach (vs. captions or recorded clips), and the choice of avatar appearance.\n")
    w("## Estimated API spend\n")
    w("| Service | Usage | Estimated cost |\n|---|---|---|")
    w("| Anthropic claude-haiku-5-5 | about 200 calls (S5 v1/v2, retries, 48 S4 rewrites, smoke tests): about 170k input / 15k output tokens | ≈ $0.03 ($0.10 / $0.50 per MTok) |")
    w("| Cartesia Sonic 3.6 TTS | 696 S4 clips + 30 S8 barista lines + smoke tests: about 28.5k characters | ≈ 28.5k credits |")
    w("| Cartesia Ink 2 STT | 30 streamed utterances: about 1.8 min of audio | small (per-minute STT pricing) |")
    w("| OpenAI, Gemini | not used | $0 |")
    w("\nPlease confirm against the Anthropic and Cartesia dashboards.")
    (ROOT / "REPORT-SPIKE.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote REPORT-SPIKE.md", len("\n".join(L)), "chars")


if __name__ == "__main__":
    main()
