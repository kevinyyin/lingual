"""Build listening/index.html: a blind forced-choice listening test for S4 (local only, not run by us).

60 clips sampled with a fixed seed: 3 voices x 4 intended emotions x conditions b (emotion tag),
c (tag + speed/volume), d (inline tag), plus no-emotion controls (a). Each listener hears clips in a
shuffled order and picks neutral / happy / sad / angry. The page never shows the condition or intended
emotion; the key is in listening/key.json. Answers can be copied as CSV and scored with
`uv run python spike/s4_listening.py score answers.csv`.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / "listening"
SEED = 7

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice emotion listening test</title>
<style>
:root { --bg:#f7f8fa; --fg:#161a20; --mut:#5d6672; --acc:#2557a7; --line:#d9dee5; --card:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#111418; --fg:#e9edf2; --mut:#9aa4af; --acc:#86aef0; --line:#2a313a; --card:#181c22; } }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { max-width: 640px; margin: 0 auto; padding: 24px 16px 60px; display:grid; gap:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; display:grid; gap:12px; }
.row { display:flex; flex-wrap:wrap; gap:8px; }
button { font:600 15px system-ui; padding:10px 14px; border-radius:6px; border:1px solid var(--acc); background:transparent; color:var(--acc); cursor:pointer; }
button.on { background:var(--acc); color:var(--card); }
.mut { color:var(--mut); font-size:14px; }
textarea { width:100%; min-height:120px; font:12px ui-monospace, monospace; }
</style></head><body><main>
<h1 style="margin:0;font-size:24px">Which emotion do you hear?</h1>
<p class="mut">Early research spike. Play each clip and pick the emotion the voice sounds like. Judge the voice, not the words.
Listen as many times as you like. Your answers stay in this browser until you copy them.</p>
<div class="card"><label>Your initials <input id="who" maxlength="8" style="font:inherit;padding:6px"></label></div>
<div class="card" id="trial"></div>
<div class="card"><div class="row"><button id="copy">Copy my answers (CSV)</button><span class="mut" id="msg"></span></div>
<textarea id="out" readonly></textarea></div>
</main>
<script>
const CLIPS = __CLIPS__;
const EMO = ["neutral", "happy", "sad", "angry"];
let st = {}; try { st = JSON.parse(localStorage.getItem("s4ans") || "{}"); } catch (e) {}
const who = document.getElementById("who");
try { who.value = localStorage.getItem("s4who") || ""; } catch (e) {}
who.oninput = () => { try { localStorage.setItem("s4who", who.value); } catch (e) {} };
// per-listener order: shuffle seeded by initials so two listeners hear different orders
function order() { let s = 0; for (const c of (who.value || "x")) s = (s * 31 + c.charCodeAt(0)) >>> 0;
  const a = CLIPS.map((_, i) => i); for (let i = a.length - 1; i > 0; i--) { s = (s * 1103515245 + 12345) >>> 0; const j = s % (i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; }
let pos = 0;
function render() {
  const ord = order(); const box = document.getElementById("trial");
  if (pos >= ord.length) { box.innerHTML = "<b>All done. Thank you!</b> Copy your answers below."; return; }
  const id = CLIPS[ord[pos]];
  box.innerHTML = `<div class="mut">Clip ${pos + 1} of ${ord.length}</div><audio controls preload="auto" src="audio/${id}.wav"></audio>
   <div class="row">${EMO.map(e => `<button data-e="${e}" class="${st[id] === e ? "on" : ""}">${e}</button>`).join("")}</div>
   <div class="row"><button id="prev">Back</button><button id="next">Next</button></div>`;
  box.querySelectorAll("[data-e]").forEach(b => b.onclick = () => { st[id] = b.dataset.e; try { localStorage.setItem("s4ans", JSON.stringify(st)); } catch (e) {} pos++; render(); });
  box.querySelector("#prev").onclick = () => { pos = Math.max(0, pos - 1); render(); };
  box.querySelector("#next").onclick = () => { pos++; render(); };
}
document.getElementById("copy").onclick = () => {
  const rows = ["listener,clip,answer", ...Object.entries(st).map(([k, v]) => `${who.value || "anon"},${k},${v}`)];
  const t = rows.join("\\n"); const out = document.getElementById("out"); out.value = t;
  navigator.clipboard.writeText(t).then(() => msg.textContent = "Copied.", () => { out.select(); msg.textContent = "Select-all and copy the text below."; });
};
render();
</script></body></html>"""


def build() -> None:
    runs = [json.loads(x) for x in (ROOT / "results/s4_tts_runs.jsonl").read_text().splitlines() if x.strip()]
    rng = random.Random(SEED)
    pick = []
    for cond, per in (("b", 1), ("c", 1), ("d", 1)):
        for v in ("Leo", "Maya", "Gavin"):
            for e in ("neutral", "happy", "sad", "angry"):
                cand = [r for r in runs if r["cond"] == cond and r["voice"] == v and r["intended"] == e and not r.get("error")]
                pick += rng.sample(cand, per)
    for v in ("Leo", "Maya", "Gavin"):
        pick += rng.sample([r for r in runs if r["cond"] == "a" and r["voice"] == v], 8)
    rng.shuffle(pick)
    (L / "audio").mkdir(parents=True, exist_ok=True)
    key, ids = {}, []
    for i, r in enumerate(pick):
        cid = f"c{i:02d}"
        shutil.copy(ROOT / "cache/s4" / f"{r['id']}.wav", L / "audio" / f"{cid}.wav")
        key[cid] = {"run_id": r["id"], "cond": r["cond"], "voice": r["voice"], "intended": r["intended"], "text": r["text"]}
        ids.append(cid)
    (L / "index.html").write_text(PAGE.replace("__CLIPS__", json.dumps(ids)), encoding="utf-8")
    json.dump(key, open(L / "key.json", "w"), indent=1)
    print("listening sheet:", len(ids), "clips")


def score(csv_path: str) -> None:
    key = json.load(open(L / "key.json"))
    rows = list(csv.DictReader(open(csv_path)))
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        k = key[r["clip"]]
        if k["intended"] is None:
            continue
        acc[k["cond"]][0] += r["answer"] == k["intended"]
        acc[k["cond"]][1] += 1
    print({c: f"{a}/{n}" for c, (a, n) in acc.items()})


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        build()
