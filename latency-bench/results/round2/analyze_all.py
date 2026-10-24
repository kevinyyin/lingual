"""Re-analyze every round-2 batch with Whisper (chrF). Transcriptions are cached on disk."""
import json, time
from bench.config import load_config
from bench.round2.analyze import analyze_batches
from bench.round2.common import R2
cfg = load_config()
batches = json.loads((R2 / "batches.json").read_text())
for name, ids in batches.items():
    t = time.time()
    analyze_batches(cfg, name, ids, asr=True, log=lambda *a: None)
    print(f"{name}: done in {time.time() - t:.0f} s", flush=True)
