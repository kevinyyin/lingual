"""`bench prepare`: synthesize missing base clips (flagged synthetic), build derived clips
(noisy / overlap / silence), pad every clip with room tone, run VAD, write the index."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np

from bench.audio import (
    load_wav, mix_at_snr, mix_overlap, normalize_peak, pad_with_room_tone, resample, room_tone, save_wav,
)
from bench.config import CORPUS_DIR, BenchConfig
from bench.corpus import Clip, DerivedSpec, PreparedClip, load_index, load_manifest, save_index, PREPARED_DIR
from bench.vad import source_timing

CARTESIA_API = "https://api.cartesia.ai"
CARTESIA_VERSION = "2026-08-14"
VOICE_CACHE = CORPUS_DIR / "cache" / "voices.json"


# ------------------------------------------------------------------ TTS backends
class TTSError(RuntimeError):
    pass


def _cartesia_voices(lang: str) -> list[str]:
    cache = json.loads(VOICE_CACHE.read_text(encoding="utf-8")) if VOICE_CACHE.exists() else {}
    if lang in cache:
        return cache[lang]
    key = os.environ.get("CARTESIA_API_KEY")
    if not key:
        raise TTSError("CARTESIA_API_KEY not set")
    r = httpx.get(f"{CARTESIA_API}/voices", params={"language": lang, "limit": 50},
                  headers={"X-API-Key": key, "Cartesia-Version": CARTESIA_VERSION}, timeout=30)
    r.raise_for_status()
    voices = [v["id"] for v in r.json().get("data", []) if v.get("language") == lang] or \
             [v["id"] for v in r.json().get("data", [])]
    if not voices:
        raise TTSError(f"no Cartesia voices for language {lang}")
    cache[lang] = voices[:8]
    VOICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    VOICE_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[lang]


def synth_cartesia(text: str, lang: str, out: Path, cfg: BenchConfig, voice_idx: int) -> str:
    key = os.environ.get("CARTESIA_API_KEY")
    if not key:
        raise TTSError("CARTESIA_API_KEY not set")
    voices = _cartesia_voices(lang)
    voice = voices[voice_idx % len(voices)]
    body = {
        "model_id": cfg.prepare.cartesia_model, "transcript": text, "voice": {"id": voice},
        "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
        "language": lang,
    }
    r = httpx.post(f"{CARTESIA_API}/tts/bytes", json=body,
                   headers={"X-API-Key": key, "Cartesia-Version": CARTESIA_VERSION}, timeout=120)
    if r.status_code != 200:
        raise TTSError(f"Cartesia TTS {r.status_code}: {r.text[:300]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return f"cartesia:{cfg.prepare.cartesia_model}:{voice}"


def synth_openai(text: str, lang: str, out: Path, cfg: BenchConfig, voice_idx: int) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise TTSError("OPENAI_API_KEY not set")
    voices = ["coral", "ash", "sage", "verse", "ballad", "alloy"]
    voice = cfg.prepare.openai_tts_voice if voice_idx == 0 else voices[voice_idx % len(voices)]
    body = {"model": cfg.prepare.openai_tts_model, "voice": voice, "input": text, "response_format": "wav",
            "instructions": f"Speak naturally and conversationally, as a native speaker of language code '{lang}'."}
    r = httpx.post("https://api.openai.com/v1/audio/speech", json=body,
                   headers={"Authorization": f"Bearer {key}"}, timeout=120)
    if r.status_code != 200:
        raise TTSError(f"OpenAI TTS {r.status_code}: {r.text[:300]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return f"openai:{cfg.prepare.openai_tts_model}:{voice}"


def os_voices() -> dict[str, list[str]]:
    """Locally installed OS voices by 2-letter language (Windows SAPI only for now)."""
    if sys.platform != "win32":
        return {}
    ps = ("Add-Type -AssemblyName System.Speech; "
          "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
          "% { $_.VoiceInfo.Culture.TwoLetterISOLanguageName + '|' + $_.VoiceInfo.Name }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    res: dict[str, list[str]] = {}
    for ln in out.splitlines():
        if "|" in ln:
            lang, name = ln.strip().split("|", 1)
            res.setdefault(lang, []).append(name)
    return res


def synth_os(text: str, lang: str, out: Path, cfg: BenchConfig | None, voice_idx: int) -> str:
    """Offline synthesis with the OS voice (Windows SAPI / macOS `say` / Linux espeak-ng)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        voices = os_voices().get(lang)
        if not voices:
            raise TTSError(f"no local Windows voice for language '{lang}'")
        voice = voices[voice_idx % len(voices)]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            txt = f.name
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SelectVoice('{voice}'); "
            "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
            f"$s.SetOutputToWaveFile('{out}', $fmt); "
            f"$s.Speak([IO.File]::ReadAllText('{txt}')); $s.Dispose()"
        )
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
        os.unlink(txt)
        if r.returncode != 0 or not out.exists():
            raise TTSError(f"SAPI synthesis failed: {r.stderr[:300]}")
        return f"os:sapi:{voice}"
    if sys.platform == "darwin":
        aiff = out.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        x, sr = load_wav(aiff)
        save_wav(out, x, sr)
        aiff.unlink()
        return "os:say"
    r = subprocess.run(["espeak-ng", "-v", lang, "-w", str(out), text], capture_output=True)
    if r.returncode != 0:
        raise TTSError("espeak-ng not available")
    return "os:espeak-ng"


SYNTHS = {"cartesia": synth_cartesia, "openai": synth_openai, "os": synth_os}


# ------------------------------------------------------------------ helpers
def trim_silence(x: np.ndarray, sr: int, thr_dbfs: float = -50.0, margin_s: float = 0.03) -> np.ndarray:
    thr = 10 ** (thr_dbfs / 20)
    idx = np.flatnonzero(np.abs(x) > thr)
    if idx.size == 0:
        return x
    m = int(margin_s * sr)
    return x[max(0, idx[0] - m): min(len(x), idx[-1] + m)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def finalize_clip(
    x: np.ndarray, sr: int, cfg: BenchConfig, *, cid: str, pair: str, category: str, source_lang: str,
    synthetic: bool, out_dir: Path, text: str | None, reference: str | None, meta: dict | None = None,
    pad: bool = True,
) -> PreparedClip:
    """Resample to the prepared rate, pad with room tone, run VAD, save, index entry."""
    psr = cfg.audio.prepared_sample_rate
    x = resample(x, sr, psr)
    seed = int(hashlib.md5(cid.encode()).hexdigest()[:6], 16)
    if pad:
        x = pad_with_room_tone(x, psr, cfg.audio.pad_pre_s, cfg.audio.pad_post_s, cfg.audio.room_tone_dbfs, seed)
    path = out_dir / f"{cid}.wav"
    save_wav(path, x, psr)
    x, _ = load_wav(path)  # VAD on exactly what will be streamed (post PCM16 quantization)
    timing = source_timing(x, psr, cfg.vad)
    try:
        rel = str(path.relative_to(CORPUS_DIR)).replace("\\", "/")
    except ValueError:
        rel = str(path)
    return PreparedClip(
        id=cid, pair=pair, category=category, source_lang=source_lang, synthetic=synthetic, file=rel,
        sample_rate=psr, duration_s=round(len(x) / psr, 3), timing=timing, text=text, reference=reference,
        sha256=_sha(path), meta=meta or {},
    )


def _babble(base: dict[str, np.ndarray], exclude: set[str], sr: int, n: int, seed: int, talkers: int = 6) -> np.ndarray:
    """Café-like babble: several unrelated talkers summed at random offsets."""
    rng = random.Random(seed)
    pool = [k for k in base if k not in exclude] or list(base)
    y = np.zeros(n, np.float32)
    for _ in range(talkers):
        src = base[rng.choice(pool)]
        reps = np.tile(src, int(np.ceil(n / max(1, len(src)))) + 1)
        off = rng.randrange(0, max(1, len(src)))
        y += reps[off: off + n] * rng.uniform(0.6, 1.0)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5


# ------------------------------------------------------------------ main entry
def prepare(cfg: BenchConfig, tts: str | None = None, only_pairs: list[str] | None = None, log=print) -> dict[str, PreparedClip]:
    manifest = load_manifest()
    tts = tts or cfg.prepare.tts
    synth = SYNTHS[tts]
    index = load_index()
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    base_audio: dict[str, np.ndarray] = {}
    psr = cfg.audio.prepared_sample_rate
    voice_counter: dict[str, int] = {}
    errors: list[str] = []

    def want(pair: str) -> bool:
        return not only_pairs or pair in only_pairs

    # base clips needed by wanted pairs directly, or as inputs to their derived clips
    needed = {c.id for c in manifest.clips if want(c.pair)}
    needed |= {i for d in manifest.derived if want(d.pair) for i in d.inputs}

    # 1) base clips: record/synthesize, trim, pad, VAD
    for clip in manifest.clips:
        audio_path = CORPUS_DIR / (clip.file or f"audio/{clip.id}.wav")
        synthetic = clip.synthetic
        voice = clip.voice
        if not audio_path.exists():
            if not clip.text:
                errors.append(f"{clip.id}: no audio file and no text to synthesize")
                continue
            if clip.id not in needed:
                continue
            lang = clip.source_lang
            if clip.voice_group:  # same voice for every sentence of a multi-sentence clip
                vi = int(hashlib.md5(clip.voice_group.encode()).hexdigest()[:4], 16)
            else:
                vi = voice_counter.get(lang, 0)
                voice_counter[lang] = vi + 1
            try:
                voice = synth(clip.text, lang, audio_path, cfg, vi)
                synthetic = True
                log(f"[tts] {clip.id} ({voice})")
            except Exception as e:
                errors.append(f"{clip.id}: synthesis failed: {e}")
                continue
        elif synthetic is False and audio_path.with_suffix(".synth.json").exists():
            synthetic = True  # audio produced by an earlier `prepare` run
            voice = json.loads(audio_path.with_suffix(".synth.json").read_text())["voice"]
        if synthetic and not audio_path.with_suffix(".synth.json").exists():
            audio_path.with_suffix(".synth.json").write_text(json.dumps({"voice": voice, "synthetic": True}))
        x, sr = load_wav(audio_path)
        x = normalize_peak(trim_silence(x, sr), -3.0)
        base_audio[clip.id] = resample(x, sr, psr)
        if not want(clip.pair):
            continue
        pc = finalize_clip(x, sr, cfg, cid=clip.id, pair=clip.pair, category=clip.category,
                           source_lang=clip.source_lang, synthetic=synthetic, out_dir=PREPARED_DIR,
                           text=clip.text, reference=clip.reference, meta={"voice": voice})
        if pc.timing.onset_sample is None and clip.category != "silence":
            errors.append(f"{clip.id}: VAD found no speech")
        index[clip.id] = pc

    # 2) derived clips
    clips_by_id = manifest.by_id()
    noise_files = sorted((CORPUS_DIR / "noise").glob("*.wav"))
    for d in manifest.derived:
        if not want(d.pair):
            continue
        try:
            pc = _derive(d, cfg, base_audio, clips_by_id, noise_files)
            index[d.id] = pc
        except Exception as e:
            errors.append(f"{d.id}: {e}")

    save_index(index)
    for e in errors:
        log(f"[warn] {e}")
    log(f"[prepare] {len(index)} prepared clips in {PREPARED_DIR}")
    return index


def _derive(d: DerivedSpec, cfg: BenchConfig, base: dict[str, np.ndarray], clips: dict[str, Clip],
            noise_files: list[Path]) -> PreparedClip:
    psr = cfg.audio.prepared_sample_rate
    seed = int(hashlib.md5(d.id.encode()).hexdigest()[:6], 16)
    srcs = [clips[i] for i in d.inputs]
    for i in d.inputs:
        if i not in base:
            raise RuntimeError(f"input clip {i} has no audio (prepare it first)")
    synthetic = any(c.synthetic or (CORPUS_DIR / (c.file or f"audio/{c.id}.wav")).with_suffix(".synth.json").exists()
                    for c in srcs)
    meta: dict = {"derived_from": d.inputs, "kind": d.kind}
    if d.kind == "noisy":
        speech = base[d.inputs[0]]
        if d.noise and d.noise != "auto" and (CORPUS_DIR / "noise" / d.noise).exists():
            noise, _ = load_wav(CORPUS_DIR / "noise" / d.noise, psr)
            meta["noise_source"] = d.noise
        elif noise_files and d.noise != "babble":
            nf = noise_files[seed % len(noise_files)]
            noise, _ = load_wav(nf, psr)
            meta["noise_source"] = nf.name
        else:
            # building-block sentences (category `component`, added in round 2) are excluded so
            # the babble, and therefore every noisy clip, stays identical to round 1
            pool = {k: v for k, v in base.items() if k in clips and clips[k].category != "component"}
            noise = _babble(pool, set(d.inputs), psr, len(speech) + psr, seed)
            meta["noise_source"] = "babble(6 corpus talkers)"
        x = mix_at_snr(speech, noise, cfg.audio.noise_snr_db, seed)
        meta["snr_db"] = cfg.audio.noise_snr_db
        src_lang, text = srcs[0].source_lang, srcs[0].text
    elif d.kind == "overlap":
        lo, hi = cfg.audio.overlap_s
        ov = d.overlap_s if d.overlap_s is not None else random.Random(seed).uniform(lo, hi)
        x, b_start = mix_overlap(base[d.inputs[0]], base[d.inputs[1]], psr, ov)
        meta.update({"overlap_s": round(ov, 3), "speaker_b_start_s": round(b_start / psr, 3)})
        src_lang = "+".join(c.source_lang for c in srcs)
        text = " || ".join(c.text or "" for c in srcs)
    elif d.kind == "concat":
        # sentences joined by exact room-tone gaps; spans recorded for premature-flush checks
        parts, spans, pos = [], [], 0
        for k, i in enumerate(d.inputs):
            y = base[i]
            spans.append((pos, pos + len(y)))
            parts.append(y)
            pos += len(y)
            if k < len(d.inputs) - 1:
                g = room_tone(int(round(d.gaps_s[k] * psr)), psr, cfg.audio.room_tone_dbfs, seed + k)
                parts.append(g)
                pos += len(g)
        x = np.concatenate(parts)
        pre = int(round(cfg.audio.pad_pre_s * psr))
        meta.update({"gaps_s": d.gaps_s, "sentence_spans": [(a + pre, b + pre) for a, b in spans]})
        src_lang = srcs[0].source_lang
        text = " ".join(c.text or "" for c in srcs)
        ref = d.reference or " ".join(c.reference or "" for c in srcs)
        return finalize_clip(x, psr, cfg, cid=d.id, pair=d.pair, category=d.category, source_lang=src_lang,
                             synthetic=synthetic, out_dir=PREPARED_DIR, text=text, reference=ref, meta=meta)
    elif d.kind == "insert_pause":
        # one sentence with a pause inserted at a word boundary found by Whisper word timings
        from bench.analysis.align import transcribe_words

        y = base[d.inputs[0]]
        words = transcribe_words(cfg, y, psr, srcs[0].source_lang)["words"]
        k = d.after_word if d.after_word is not None else len(words) // 2 - 1
        cut = int(round((words[k][2] + words[k + 1][1]) / 2 * psr))
        gap = room_tone(int(round((d.pause_ms or 500) / 1000 * psr)), psr, cfg.audio.room_tone_dbfs, seed)
        x = np.concatenate([y[:cut], gap, y[cut:]])
        pre = int(round(cfg.audio.pad_pre_s * psr))
        meta.update({"pause_ms": d.pause_ms, "after_word": words[k][0], "pause_at_s": round(cut / psr, 3),
                     "sentence_spans": [(pre, pre + len(x))]})
        return finalize_clip(x, psr, cfg, cid=d.id, pair=d.pair, category=d.category,
                             source_lang=srcs[0].source_lang, synthetic=synthetic, out_dir=PREPARED_DIR,
                             text=srcs[0].text, reference=d.reference or srcs[0].reference, meta=meta)
    elif d.kind == "alias":
        # same audio, different pair/category (e.g. English speech into an es->en session)
        x = base[d.inputs[0]]
        src_lang, text = srcs[0].source_lang, srcs[0].text
    elif d.kind == "silence":
        x = room_tone(int((d.duration_s or 10.0) * psr), psr, cfg.audio.room_tone_dbfs, seed)
        src_lang, text, synthetic = "none", None, True  # generated room tone
        return finalize_clip(x, psr, cfg, cid=d.id, pair=d.pair, category=d.category, source_lang=src_lang,
                             synthetic=synthetic, out_dir=PREPARED_DIR, text=text, reference=d.reference,
                             meta=meta, pad=False)
    else:
        raise RuntimeError(f"unknown derived kind {d.kind}")
    ref = d.reference if d.reference is not None else (srcs[0].reference if srcs and d.kind != "alias" else None)
    return finalize_clip(x, psr, cfg, cid=d.id, pair=d.pair, category=d.category, source_lang=src_lang,
                         synthetic=synthetic, out_dir=PREPARED_DIR, text=text, reference=ref, meta=meta)
