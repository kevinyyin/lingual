"""latency-bench CLI: selftest, prepare, run, analyze, report."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer

from bench.clock import run as clock_run
from bench.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Measure how far behind the speaker each live speech-to-speech translator is.")


def _split(v: Optional[str]) -> list[str]:
    return [s.strip() for s in v.split(",") if s.strip()] if v else []


@app.command()
def selftest() -> None:
    """Run the fixed-delay mock and verify the harness measures 300 ms correctly."""
    from bench.selftest import run_selftest

    ok = run_selftest(load_config())
    raise typer.Exit(0 if ok else 1)


@app.command()
def prepare(
    tts: Optional[str] = typer.Option(None, help="cartesia | openai | os (default from config)"),
    pairs: Optional[str] = typer.Option(None, help="Comma list of pairs to prepare (default all)."),
) -> None:
    """Synthesize missing clips (flagged synthetic), build derived clips, pad, run VAD."""
    from bench.prepare import prepare as do_prepare

    idx = do_prepare(load_config(), tts=tts, only_pairs=_split(pairs) or None)
    if not idx:
        raise typer.Exit(1)


@app.command()
def run(
    providers: str = typer.Option("mock", help="Comma list: mock,openai,gemini,cascade,hibiki"),
    pairs: str = typer.Option("es-en", help="Comma list of pairs, or 'all'"),
    categories: str = typer.Option("all", help="Comma list of categories, or 'all'"),
    runs: int = typer.Option(5, help="Repetitions per (provider, clip)"),
    clips: Optional[str] = typer.Option(None, help="Restrict to these clip ids"),
    synthetic: Optional[bool] = typer.Option(None, "--synthetic/--real", help="Only synthetic / only real clips"),
    set_: list[str] = typer.Option([], "--set", help="Override provider config, e.g. --set cascade.translate_mode=incremental"),
) -> None:
    """Stream every selected clip to every provider in randomized order and record raw events."""
    import yaml
    from bench.corpus import load_index
    from bench.runner import run_batch

    cfg = load_config()
    idx = load_index()
    if not idx:
        typer.echo("No prepared clips. Run `bench prepare` first.", err=True)
        raise typer.Exit(1)
    pl = cfg.pairs if pairs == "all" else _split(pairs)
    cl = cfg.categories if categories == "all" else _split(categories)
    ids = set(_split(clips))
    selected = [c for c in idx.values() if c.pair in pl and c.category in cl and (not ids or c.id in ids)
                and (synthetic is None or c.synthetic == synthetic)]
    if not selected:
        typer.echo("No clips match the selection.", err=True)
        raise typer.Exit(1)
    missing = sorted(set(cl) - {c.category for c in selected})
    if missing:
        typer.echo(f"note: no clips for categories {missing} in pairs {pl}")
    typer.echo(f"{len(selected)} clips x {runs} runs x providers {providers}")
    overrides: dict[str, dict] = {}
    for item in set_:
        key, _, val = item.partition("=")
        prov, _, path = key.partition(".")
        if not path or not _:
            typer.echo(f"bad --set '{item}' (expected provider.key=value)", err=True)
            raise typer.Exit(1)
        node = overrides.setdefault(prov, {})
        parts = path.split(".")
        base = cfg.provider_cfg(prov)
        if len(parts) > 1:  # nested key: merge into a copy of the configured sub-dict
            node.setdefault(parts[0], dict(base.get(parts[0], {})))
            sub = node[parts[0]]
            for p in parts[1:-1]:
                sub = sub.setdefault(p, {})
            sub[parts[-1]] = yaml.safe_load(val)
        else:
            node[parts[0]] = yaml.safe_load(val)
    if overrides:
        typer.echo(f"provider overrides: {overrides}")
    clock_run(run_batch(cfg, _split(providers), selected, runs, provider_overrides=overrides))


@app.command()
def analyze(
    batch: Optional[str] = typer.Option(None, help="Only analyze these batch ids (comma list); default all"),
    no_evs: bool = typer.Option(False, help="Skip LLM word alignment (EVS)"),
    no_asr: bool = typer.Option(False, help="Skip faster-whisper transcription (also disables EVS/chrF)"),
    whisper_model: Optional[str] = typer.Option(None, help="Override analysis.whisper_model"),
) -> None:
    """Compute per-run metrics from raw events; write results/analysis/runs.jsonl and summary.csv."""
    from bench.analysis.pipeline import analyze_all

    cfg = load_config()
    if whisper_model:
        cfg.analysis.whisper_model = whisper_model
    analyze_all(cfg, batches=_split(batch) or None, evs=not no_evs and not no_asr, asr=not no_asr)


@app.command()
def report(
    batch: Optional[str] = typer.Option(None, help="Restrict report to these batch ids"),
    rewrite_summary: bool = typer.Option(False, help="Also rewrite results/summary.csv from round-1 runs"),
) -> None:
    """Build results/report.html (self-contained) from analyzed runs (round 1 + round 2)."""
    from bench.report.build import build_report

    path = build_report(load_config(), batches=_split(batch) or None, rewrite_summary=rewrite_summary)
    typer.echo(f"wrote {path}")


r2 = typer.Typer(add_completion=False, no_args_is_help=True, help="Round-2 experiments.")
app.add_typer(r2, name="r2")


@r2.command("exp1")
def r2_exp1() -> None:
    """Experiment 1: re-score round-1 runs under catch-up playback policies (offline)."""
    from bench.round2 import exp1

    exp1.run(load_config())


def _batch_cost(batch_id: str, experiment: str, note: str) -> dict:
    import json as _json

    from bench.config import RESULTS_DIR
    from bench.round2.common import log_cost

    sessions = 0
    audio_s = wall_s = 0.0
    other: dict[str, float] = {}
    for f in sorted((RESULTS_DIR / "raw").glob(f"{batch_id}_*.jsonl")):
        prov = None
        for line in open(f, encoding="utf-8"):
            d = _json.loads(line)
            if d.get("type") == "header":
                prov = d["provider"]
            if d.get("type") == "event" and d.get("stage") == "runner.finish_called" and prov != "openai":
                other[prov] = other.get(prov, 0.0) + d["meta"].get("audio_s_sent", 0.0)
            if d.get("type") == "event" and d.get("stage") == "cost.session" and d["meta"].get("provider") == "openai":
                sessions += 1
                audio_s += d["meta"].get("audio_s", 0.0)
                wall_s += d["meta"].get("wall_s", 0.0)
    rec = log_cost(batch_id, experiment, note, sessions, audio_s / 60, wall_s / 60,
                   {"other_provider_audio_min": {k: round(v / 60, 2) for k, v in other.items()}})
    return rec


@r2.command("run")
def r2_run(
    name: str = typer.Argument(..., help="Plan name: exp2a, exp2b, exp2c, exp3, exp4, exp5a, exp5b, exp6"),
    chunk: Optional[int] = typer.Option(None), trailing: Optional[str] = typer.Option(None),
    runs: Optional[int] = typer.Option(None, help="Override repetitions (smoke tests)"),
    clips: Optional[str] = typer.Option(None, help="Restrict to clip ids (smoke tests)"),
    tag: Optional[str] = typer.Option(None, help="Override the experiment tag (e.g. smoke)"),
    best: Optional[str] = typer.Option(None, help="exp5b: winning single-context variant from exp5a"),
    base: Optional[str] = typer.Option(None, help="exp6: JSON overrides of the best exp5 configuration"),
    only: Optional[str] = typer.Option(None, help="Run only these system labels (comma list)"),
) -> None:
    """Run a round-2 plan in one randomized batch, then log its estimated cost."""
    from bench.round2.plans import plan as get_plan
    from bench.runner import BatchAborted, run_batch

    import json as _json

    kw = {k: v for k, v in {"chunk": chunk, "trailing": trailing, "best": best}.items() if v is not None}
    if base:
        kw["base"] = _json.loads(base)
    pl = get_plan(name, **kw)
    cfg = load_config()
    cfg = cfg.model_copy(update={"concurrency": pl.concurrency, "inter_run_delay_s": 1.0})
    if only:
        keep = set(_split(only))
        pl.systems = [x for x in pl.systems if x.label in keep]
    sel = pl.clips()
    if clips:
        ids = set(_split(clips))
        sel = [c for c in sel if c.id in ids]
    n = runs or pl.runs
    exp = tag or pl.experiment
    typer.echo(f"{exp}: {len(pl.systems)} systems x {len(sel)} clips x {n} reps = {len(pl.systems) * len(sel) * n} runs, "
               f"concurrency {pl.concurrency}. {pl.notes}")
    try:
        meta = clock_run(run_batch(cfg, None, sel, n, systems=pl.systems, experiment=exp))
    except BatchAborted as e:
        typer.echo(f"PAUSED: an API stopped working ({e}). Fix the account, then: bench r2 resume <batch_id>", err=True)
        raise typer.Exit(2)
    rec = _batch_cost(meta["batch_id"], exp, pl.notes)
    typer.echo(f"batch {meta['batch_id']} done. Estimated OpenAI spend: ${rec['openai_est_usd']:.2f} "
               f"({rec['openai_audio_min']:.1f} audio-min in {rec['openai_sessions']} sessions).")


@r2.command("resume")
def r2_resume(batch_id: str) -> None:
    """Re-run the jobs an aborted batch did not finish (new batch id, linked to the old one)."""
    import json as _json

    from bench.config import RESULTS_DIR
    from bench.corpus import load_index
    from bench.runner import BatchAborted, Job, System, run_batch

    meta = _json.loads((RESULTS_DIR / "raw" / f"batch_{batch_id}.json").read_text(encoding="utf-8"))
    systems = {s["label"]: System(s["label"], s["provider"], s["overrides"]) for s in meta["systems"]}
    idx = load_index()
    jobs = [Job(systems[j["system"]].provider, idx[j["clip"]], j["rep"], j["order_index"], systems[j["system"]])
            for j in meta.get("remaining_jobs", [])]
    cfg = load_config().model_copy(update={"concurrency": meta.get("concurrency", 1), "inter_run_delay_s": 1.0})
    try:
        m = clock_run(run_batch(cfg, None, [], 0, systems=list(systems.values()), experiment=meta.get("experiment"),
                                jobs_override=jobs, continues=batch_id))
    except BatchAborted as e:
        typer.echo(f"PAUSED again: {e}", err=True)
        raise typer.Exit(2)
    rec = _batch_cost(m["batch_id"], meta.get("experiment") or "", f"resume of {batch_id}")
    typer.echo(f"resume batch {m['batch_id']} done. Estimated OpenAI spend: ${rec['openai_est_usd']:.2f}")


@r2.command("analyze")
def r2_analyze(name: str, batch: str = typer.Option(..., help="Comma list of batch ids"),
               no_asr: bool = typer.Option(False, help="Skip Whisper (no chrF); fast, for stall decisions")) -> None:
    """Analyze round-2 batches into results/round2/analysis/<name>.jsonl."""
    from bench.round2.analyze import analyze_batches

    analyze_batches(load_config(), name, _split(batch), asr=not no_asr)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
