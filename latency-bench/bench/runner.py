"""Runs (provider x clip x repetition) in randomized order and records raw events.

One run:
  1. resample the prepared clip to the provider's input rate and cut 20 ms PCM16 chunks
  2. t_connect_start -> provider.connect() -> t_session_ready   (setup time, kept separate)
  3. paced_send(): chunks at exactly 1x real time on an absolute perf_counter_ns schedule
  4. provider.finish(): graceful close, drain every tail event until `closed`
  5. write results/raw/<run_id>.jsonl and results/audio/<run_id>.wav

Any exception is recorded in the run footer (status=failed) and the batch moves on.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import gc
import hashlib
import json
import random
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.audio import chunk_pcm16, load_wav, resample, room_tone_cached
from bench.clock import SendRecord, enable_high_res_timer, now_ns, paced_send
from bench.config import RESULTS_DIR, BenchConfig, config_raw_text
from bench.corpus import PreparedClip
from bench.events import (
    ConnectRecord, EventRecord, ProviderEvent, RunFooter, RunHeader, SendRecordModel, write_jsonl,
)
from bench.meta import machine_metadata, probe_host
from bench.providers import make_provider, provider_class
from bench.providers.base import ClipContext, Provider, UnsupportedPair

CONNECT_TIMEOUT_S = 30.0


@dataclass
class System:
    """A named experimental condition: a provider plus config overrides."""

    label: str
    provider: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    provider: str
    clip: PreparedClip
    repetition: int
    order_index: int = 0
    system: System | None = None


@dataclass
class RunPaths:
    raw_dir: Path
    audio_dir: Path


def default_paths() -> RunPaths:
    return RunPaths(RESULTS_DIR / "raw", RESULTS_DIR / "audio")


def config_sha() -> str:
    try:
        return hashlib.sha256(config_raw_text().encode()).hexdigest()[:12]
    except Exception:
        return ""


def _chunk_index(sample: int | None, sr: int, chunk_ms: int) -> int | None:
    if sample is None:
        return None
    return int(sample * 1000 // (sr * chunk_ms))


async def run_one(
    provider: Provider,
    job: Job,
    cfg: BenchConfig,
    batch_id: str,
    paths: RunPaths,
    machine: dict[str, Any],
    network: dict[str, Any],
) -> Path:
    clip = job.clip
    chunk_ms = int(provider.cfg.get("append_chunk_ms", cfg.audio.chunk_ms))
    send_alignment = "end" if chunk_ms > cfg.audio.chunk_ms else "start"
    sr = provider.input_sample_rate
    label = job.system.label if job.system else provider.name
    safe = "".join(ch if ch.isalnum() or ch in "-" else "-" for ch in label)
    run_id = f"{batch_id}_{job.order_index:04d}_{safe}_{clip.id}_r{job.repetition}"

    x, prep_sr = load_wav(clip.path)
    x, transform = apply_clip_transform(x, prep_sr, provider.cfg, cfg)
    timing = clip.timing.model_copy(update={"n_samples": len(x)})
    x = resample(x, prep_sr, sr)
    tail = room_tone_cached(sr, sr, cfg.audio.room_tone_dbfs, seed=7)
    if transform.get("trailing_audio") == "digital_silence":
        tail = np.zeros_like(tail)
    chunks = chunk_pcm16(x, sr, chunk_ms, tail_fill=tail)
    chunk_samples = sr * chunk_ms // 1000
    provider.set_clip_context(ClipContext(
        onset_chunk=_chunk_index(timing.onset_sample, timing.sample_rate, chunk_ms),
        offset_chunk=_chunk_index(timing.offset_sample, timing.sample_rate, chunk_ms),
        chunk_ms=chunk_ms,
    ))

    header = RunHeader(
        run_id=run_id, batch_id=batch_id, provider=provider.name, provider_config=provider.public_config(),
        clip_id=clip.id, pair=clip.pair, category=clip.category, synthetic=clip.synthetic,
        repetition=job.repetition, order_index=job.order_index,
        source_lang=clip.pair.split("-")[0], target_lang=clip.pair.split("-")[1],
        chunk_ms=chunk_ms, input_sample_rate=sr, chunk_samples=chunk_samples, n_chunks=len(chunks),
        source=timing, machine=machine, network=network, config_sha=config_sha(),
        condition=job.system.label if job.system else None, experiment=provider.cfg.get("_experiment"),
        send_alignment=send_alignment, clip_transform=transform,
    )

    events: list[ProviderEvent] = []
    ready_ns: list[int] = []

    async def consume() -> None:
        async for ev in provider.events():
            events.append(ev)
            if ev.kind == "session_ready" and not ready_ns:
                ready_ns.append(ev.t_recv_ns)

    consumer = asyncio.create_task(consume())
    status, error = "ok", None
    t_connect_start = now_ns()
    connect_rec = ConnectRecord(t_connect_start_ns=t_connect_start, t_session_ready_ns=None)
    send_rec = SendRecord()
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()  # no collector pauses while the stream is live
    try:
        await asyncio.wait_for(provider.connect(header.source_lang, header.target_lang), CONNECT_TIMEOUT_S)
        await asyncio.sleep(0)  # let the consumer record session_ready
        connect_rec.t_session_ready_ns = ready_ns[0] if ready_ns else now_ns()

        async def send(i: int, chunk: bytes, t_sent: int) -> None:
            await provider.send_audio(chunk, t_sent)

        await paced_send(chunks, chunk_ms * 1_000_000, send, record=send_rec)
        events.append(ProviderEvent("stage_mark", now_ns(), stage="runner.finish_called",
                                    meta={"audio_s_sent": round(len(chunks) * chunk_ms / 1000, 3)}))
        drain_max = float(provider.cfg.get("drain_max_s", provider.cfg.get("close_timeout_s", 30.0))) + 5.0
        await asyncio.wait_for(provider.finish(), drain_max)
        await asyncio.wait_for(consumer, 5.0)
        fatal = [e for e in events if e.kind == "error" and e.meta.get("fatal")]
        api = [e for e in events if e.kind == "error" and looks_like_api_failure(e.text)]
        if fatal:
            status, error = "failed", f"provider error: {fatal[0].text}"
        elif api:  # e.g. a quota error delivered as an in-session event on an otherwise open session
            status, error = "failed", f"provider API error: {api[0].text}"
    except UnsupportedPair as e:
        status, error = "unsupported", str(e)
    except Exception as e:  # isolate: record and move on
        status = "failed"
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
    finally:
        if gc_was_enabled:
            gc.enable()
        try:
            await asyncio.wait_for(provider.aclose(), 5.0)
        except Exception:
            pass
        if not consumer.done():
            consumer.cancel()
            try:
                await consumer
            except (asyncio.CancelledError, Exception):
                pass

    # ---- write output audio (arrival order) and the raw event log ----
    records: list[Any] = [header, connect_rec]
    if send_rec.sent_ns:
        records.append(SendRecordModel(
            planned_ns=send_rec.planned_ns, sent_ns=send_rec.sent_ns,
            max_lateness_ms=round(send_rec.max_lateness_ms, 3), drift_ms=round(send_rec.drift_ms, 3),
            p99_lateness_ms=round(send_rec.p99_lateness_ms, 3),
        ))
    out_sr: int | None = None
    pcm_parts: list[bytes] = []
    offset = 0
    for ev in events:
        rec = EventRecord(kind=ev.kind, t_recv_ns=ev.t_recv_ns, payload_size=ev.payload_size,
                          text=ev.text, stage=ev.stage, meta=ev.meta)
        if ev.kind == "audio_out" and ev.pcm:
            pcm = ev.pcm
            if out_sr is None:
                out_sr = ev.sample_rate
            if ev.sample_rate != out_sr:  # keep one wav rate; note the original
                y = resample(np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0, ev.sample_rate, out_sr)
                pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
                rec.meta = {**rec.meta, "original_sample_rate": ev.sample_rate}
            n = len(pcm) // 2
            rec.sample_rate, rec.audio_offset, rec.n_samples = out_sr, offset, n
            pcm_parts.append(pcm)
            offset += n
        records.append(rec)

    audio_file = None
    if pcm_parts and out_sr:
        from bench.audio import save_pcm16_wav

        audio_path = paths.audio_dir / f"{run_id}.wav"
        save_pcm16_wav(audio_path, b"".join(pcm_parts), out_sr)
        audio_file = audio_path.name
    records.append(RunFooter(status=status, error=error, t_end_ns=now_ns(),
                             output_audio_file=audio_file, output_sample_rate=out_sr))
    raw_path = paths.raw_dir / f"{run_id}.jsonl"
    write_jsonl(raw_path, records)
    return raw_path


_NOISE_BED: dict[int, np.ndarray] = {}


def noise_bed(sr: int, n: int, cfg: BenchConfig) -> np.ndarray:
    """Six-talker babble from the corpus (the same construction as the noisy category),
    normalized to -50 dBFS RMS, cached on disk for reproducibility."""
    from bench.audio import rms, save_wav
    from bench.config import CORPUS_DIR

    path = CORPUS_DIR / "cache" / "noise_bed_babble_m50.wav"
    if sr not in _NOISE_BED:
        if not path.exists():
            from bench.prepare import _babble, normalize_peak, trim_silence

            base = {}
            for f in sorted((CORPUS_DIR / "audio").glob("*.wav")):
                y, ysr = load_wav(f, cfg.audio.prepared_sample_rate)
                base[f.stem] = normalize_peak(trim_silence(y, ysr), -3.0)
            psr = cfg.audio.prepared_sample_rate
            bed = _babble(base, set(), psr, 60 * psr, seed=50)
            save_wav(path, bed / (rms(bed) + 1e-12) * 10 ** (-50 / 20), psr)
        y, _ = load_wav(path, sr)
        _NOISE_BED[sr] = y
    y = _NOISE_BED[sr]
    return np.tile(y, int(np.ceil(n / len(y))))[:n]


def apply_clip_transform(x: np.ndarray, sr: int, pcfg: dict[str, Any], cfg: BenchConfig) -> tuple[np.ndarray, dict]:
    """Optional per-condition change of the trailing audio after speech (round 2).

    pad_post_s replaces the prepared clip's standard post-speech padding with that many
    seconds of `trailing_audio`: room_tone_m60 (round-1 pink noise, -60 dBFS RMS),
    digital_silence (zeros) or noise_bed_m50 (six-talker babble at -50 dBFS RMS)."""
    pad = pcfg.get("pad_post_s")
    kind = pcfg.get("trailing_audio")
    if pad is None and kind is None:
        return x, {}
    pad = float(pad if pad is not None else cfg.audio.pad_post_s)
    kind = kind or "room_tone_m60"
    body = x[: len(x) - int(round(cfg.audio.pad_post_s * sr))]
    n = int(round(pad * sr))
    if kind == "room_tone_m60":
        trail = room_tone_cached(n, sr, cfg.audio.room_tone_dbfs, seed=11)
    elif kind == "digital_silence":
        trail = np.zeros(n, np.float32)
    elif kind == "noise_bed_m50":
        trail = noise_bed(sr, n, cfg)
    else:
        raise ValueError(f"unknown trailing_audio {kind}")
    return np.concatenate([body, trail.astype(np.float32)]), {"pad_post_s": pad, "trailing_audio": kind}


def _batch_id() -> str:
    # wall clock is used ONLY to name the batch, never in latency math
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def provider_hosts(cfg: BenchConfig, name: str) -> list[str]:
    hosts = list(provider_class(name).hosts)
    if name == "cascade":
        from urllib.parse import urlparse

        role = cfg.provider_cfg("cascade").get("llm", "translate")
        hosts.append(urlparse(cfg.llm_cfg(role).base_url).hostname or "")
    return [h for h in hosts if h]


def build_jobs(systems: list[System], clips: list[PreparedClip], runs: int, seed: int) -> list[Job]:
    jobs = [Job(s.provider, c, r, system=s) for s in systems for c in clips for r in range(runs)]
    random.Random(seed).shuffle(jobs)
    for i, j in enumerate(jobs):
        j.order_index = i
    return jobs


API_FAILURE_RE = re.compile(
    r"\b(401|402|403|429)\b|insufficient_quota|quota|billing|\bcredits?\b|payment|rate[ _]limit|unauthori[sz]ed"
    r"|invalid[ _]api[ _]key|limit exceeded|suspended|permission denied", re.I)


class BatchAborted(RuntimeError):
    """Raised when an API stops working (likely billing/quota); remaining jobs are recorded."""


def looks_like_api_failure(error: str | None) -> bool:
    return bool(API_FAILURE_RE.search(error or ""))


async def run_batch(
    cfg: BenchConfig,
    providers: list[str] | None,
    clips: list[PreparedClip],
    runs: int,
    paths: RunPaths | None = None,
    provider_overrides: dict[str, dict[str, Any]] | None = None,
    probe_network: bool = True,
    log=print,
    systems: list[System] | None = None,
    experiment: str | None = None,
    jobs_override: list[Job] | None = None,
    continues: str | None = None,
) -> dict[str, Any]:
    """Run every (system, clip, repetition) in randomized order.

    `providers` (round 1) is shorthand for one system per provider. Round 2 passes
    `systems`: several named conditions of the same provider in one randomized batch.
    Circuit breaker: a run whose error looks like an auth/billing/quota/rate-limit problem,
    or 3 consecutive failures of one provider, aborts the batch; the failed run is moved to
    results/round2/tossed/ and the remaining jobs are recorded for a resume."""
    enable_high_res_timer()
    paths = paths or default_paths()
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.audio_dir.mkdir(parents=True, exist_ok=True)
    batch_id = _batch_id()
    machine = machine_metadata(cfg.resolved_location)
    if systems is None:
        systems = [System(p, p, dict((provider_overrides or {}).get(p, {}))) for p in (providers or [])]

    def pcfg(system: System) -> dict[str, Any]:
        d = cfg.provider_cfg(system.provider)
        for k, v in system.overrides.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = {**d[k], **v}
            else:
                d[k] = v
        d["_bench"] = cfg  # providers that need other config sections (cascade LLM) read it here
        if experiment:
            d["_experiment"] = experiment
        return d

    skipped: dict[str, str] = {}
    active: list[System] = []
    for s_ in systems:
        reason = provider_class(s_.provider).check_available(pcfg(s_))
        if reason:
            skipped[s_.label] = reason
            log(f"[skip] {s_.label}: {reason}")
        else:
            active.append(s_)

    network: dict[str, Any] = {}
    if probe_network:
        for s_ in active:
            for host in provider_hosts(cfg, s_.provider):
                if host not in network:
                    log(f"[net] probing {host} ...")
                    network[host] = await asyncio.to_thread(probe_host, host, 443, cfg.network_probe.samples)

    if jobs_override is not None:
        jobs = [j for j in jobs_override if j.system and j.system.label in {a.label for a in active}]
    else:
        jobs = build_jobs(active, clips, runs, cfg.seed)
    batch_meta = {
        "batch_id": batch_id, "experiment": experiment, "continues": continues, "seed": cfg.seed, "runs": runs,
        "systems": [{"label": s_.label, "provider": s_.provider, "overrides": s_.overrides} for s_ in systems],
        "providers": sorted({s_.provider for s_ in systems}), "active": [s_.label for s_ in active],
        "skipped": skipped, "n_jobs": len(jobs), "machine": machine, "network": network,
        "config_sha": config_sha(), "concurrency": cfg.concurrency, "inter_run_delay_s": cfg.inter_run_delay_s,
        "job_order": [f"{j.system.label if j.system else j.provider}/{j.clip.id}/r{j.repetition}" for j in jobs],
    }
    meta_path = paths.raw_dir / f"batch_{batch_id}.json"
    meta_path.write_text(json.dumps(batch_meta, indent=1), encoding="utf-8")

    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    done = 0
    abort: list[str] = []
    finished: set[int] = set()
    consecutive: dict[str, int] = {}

    async def do(job: Job) -> None:
        nonlocal done
        if abort:
            return
        async with sem:
            if abort:
                return
            system = job.system or System(job.provider, job.provider)
            prov = make_provider(job.provider, pcfg(system))
            path = await run_one(prov, job, cfg, batch_id, paths, machine, network)
            done += 1
            footer = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
            err = footer.get("error")
            log(f"[{done}/{len(jobs)}] {system.label:24s} {job.clip.id:26s} r{job.repetition} -> {footer['status']}"
                + (f" ({err.splitlines()[0][:140]})" if err else ""))
            if footer["status"] == "failed":
                consecutive[job.provider] = consecutive.get(job.provider, 0) + 1
                if looks_like_api_failure(err) or consecutive[job.provider] >= 3:
                    tossed = RESULTS_DIR / "round2" / "tossed"
                    tossed.mkdir(parents=True, exist_ok=True)
                    path.replace(tossed / path.name)
                    if footer.get("output_audio_file"):
                        a = paths.audio_dir / footer["output_audio_file"]
                        if a.exists():
                            a.replace(tossed / a.name)
                    abort.append(f"{system.label}: {err.splitlines()[0][:300] if err else 'repeated failures'}")
                    return
            else:
                consecutive[job.provider] = 0
            finished.add(job.order_index)
            if cfg.inter_run_delay_s > 0:
                await asyncio.sleep(cfg.inter_run_delay_s)

    if cfg.concurrency <= 1:
        for job in jobs:
            await do(job)
            if abort:
                break
    else:
        await asyncio.gather(*(do(j) for j in jobs))
    if abort:
        remaining = [j for j in jobs if j.order_index not in finished]
        batch_meta["aborted"] = abort[0]
        batch_meta["remaining_jobs"] = [
            {"system": j.system.label if j.system else j.provider, "clip": j.clip.id, "rep": j.repetition,
             "order_index": j.order_index} for j in remaining]
        meta_path.write_text(json.dumps(batch_meta, indent=1), encoding="utf-8")
        log(f"[ABORTED] {abort[0]} -- {len(remaining)} jobs remain; fix the account, then resume this batch.")
        raise BatchAborted(abort[0])
    return batch_meta
