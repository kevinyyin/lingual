"""`bench report`: self-contained results/report.html (plotly embedded inline)."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
from plotly.subplots import make_subplots

from bench.analysis.metrics import STAGE_COMPONENTS, output_chunks, simulate_playback, source_times
from bench.analysis.pipeline import BEHAVIOR_CATEGORIES, EVS_CATEGORIES, load_rows, summarize, write_summary
from bench.config import RESULTS_DIR, BenchConfig, config_raw_text
from bench.events import read_run

REPORT_PATH = RESULTS_DIR / "report.html"

# Validated categorical palette (dataviz reference palette), assigned in FIXED order by
# provider identity - colour follows the entity, never its rank in a filtered view.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
PROVIDER_ORDER = ["openai", "gemini", "cascade[on_turn_end]", "cascade[incremental]", "hibiki", "mock"]
STAGE_LABELS = {
    "stt_finalize_ms": "STT finalization (speech end → EndOfTurn)",
    "translate_wait_ms": "Translate wait (EndOfTurn → first text to TTS)",
    "tts_first_byte_ms": "TTS first byte",
    "tts_stream_rest_ms": "TTS remaining stream",
}
CATEGORY_ORDER = ["short", "medium", "long", "verb_final", "noisy", "overlap", "same_language", "silence"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _color_map(labels: list[str]) -> dict[str, str]:
    ordered = [p for p in PROVIDER_ORDER if p in labels] + sorted(p for p in labels if p not in PROVIDER_ORDER)
    return {p: SERIES_LIGHT[i % len(SERIES_LIGHT)] for i, p in enumerate(ordered)}


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left", yref="container", y=1, yanchor="top",
                   pad=dict(t=8)),
        height=height, margin=dict(l=60, r=20, t=96, b=56),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK2),
        legend=dict(orientation="h", yref="container", y=1 - 34 / height, yanchor="top", x=0, xanchor="left",
                    font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def _fmt(v: Any, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return html.escape(str(v))


def _med(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def _p90(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.percentile(v, 90)) if v else None


# ------------------------------------------------------------------ sections
def summary_table(rows: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] in EVS_CATEGORIES:
            groups[(r["provider_label"], r["pair"], r["audio_source"])].append(r)
    if not groups:
        return "<p class='muted'>No speech-category runs.</p>"
    head = ("<tr><th>Provider</th><th>Pair</th><th>Audio</th><th class=n>Runs</th>"
            "<th class=n>Start lag<br>median</th><th class=n>p90</th>"
            "<th class=n>End lag (playback)<br>median</th><th class=n>p90</th>"
            "<th class=n>EVS ≈<br>median</th><th class=n>p90</th>"
            "<th class=n>Setup<br>median</th><th class=n>Failure<br>rate</th><th class=n>chrF<br>median</th></tr>")
    body = []
    for (prov, pair, src), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        ok = [r for r in rs if r["status"] == "ok"]
        att = [r for r in rs if r["status"] in ("ok", "failed")]
        fail = (len(att) - len(ok)) / len(att) if att else None
        if not att:
            body.append(f"<tr><td>{_fmt(prov)}</td><td>{pair}</td><td>{src}</td><td class=n>{len(rs)}</td>"
                        f"<td colspan=9 class=muted>unsupported for this pair</td></tr>")
            continue
        g = lambda k: [r.get(k) for r in ok]  # noqa: E731
        body.append(
            f"<tr><td>{_fmt(prov)}</td><td>{pair}</td><td>{src}</td><td class=n>{len(rs)}</td>"
            f"<td class=n>{_fmt(_med(g('start_lag_ms')))}</td><td class=n>{_fmt(_p90(g('start_lag_ms')))}</td>"
            f"<td class=n>{_fmt(_med(g('end_lag_playback_ms')))}</td><td class=n>{_fmt(_p90(g('end_lag_playback_ms')))}</td>"
            f"<td class=n>{_fmt(_med(g('evs_median_ms')))}</td><td class=n>{_fmt(_p90(g('evs_median_ms')))}</td>"
            f"<td class=n>{_fmt(_med(g('setup_ms')))}</td>"
            f"<td class=n>{_fmt(None if fail is None else fail * 100)}{'%' if fail is not None else ''}</td>"
            f"<td class=n>{_fmt(_med(g('chrf')), 1)}</td></tr>")
    return (f"<table class=data><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"
            "<p class=note>All times in ms. Speech categories only (short, medium, long, verb_final, noisy). "
            "EVS is approximate (ASR word timing + LLM word alignment). chrF is a quality sanity check, "
            "never combined with latency.</p>")


def box_plot(rows: list[dict], cmap: dict[str, str]) -> go.Figure | None:
    ok = [r for r in rows if r["status"] == "ok" and r.get("end_lag_playback_ms") is not None]
    if not ok:
        return None
    cats = [c for c in CATEGORY_ORDER if any(r["category"] == c for r in ok)]
    fig = make_subplots(rows=1, cols=len(cats), shared_yaxes=True, subplot_titles=cats, horizontal_spacing=0.02)
    shown = set()
    for ci, cat in enumerate(cats, start=1):
        for prov in cmap:
            ys = [r["end_lag_playback_ms"] for r in ok if r["category"] == cat and r["provider_label"] == prov]
            if not ys:
                continue
            fig.add_trace(go.Box(
                y=ys, name=prov, legendgroup=prov, showlegend=prov not in shown, marker_color=cmap[prov],
                line=dict(width=2), boxpoints="all", jitter=0.4, pointpos=0, marker=dict(size=8, opacity=0.55),
                hovertemplate=f"{prov}<br>{cat}<br>%{{y:.0f}} ms<extra></extra>",
            ), row=1, col=ci)
            shown.add(prov)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(title_text="End lag, playback (ms)", row=1, col=1, rangemode="tozero")
    for a in fig.layout.annotations:
        a.font = dict(size=12, color=INK2)
    return _layout(fig, "End lag (playback) per provider, by category", 460)


def stage_bars(rows: list[dict]) -> go.Figure | None:
    ok = [r for r in rows if r["status"] == "ok" and r["provider"] == "cascade"
          and all(r.get(k) is not None for k in STAGE_COMPONENTS) and r["category"] in EVS_CATEGORIES]
    if not ok:
        return None
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        groups[f"{r['provider_label']} · {r['pair']} · {r['category']}"].append(r)
    keys = sorted(groups, key=lambda k: (k.split(" · ")[0], k.split(" · ")[1], CATEGORY_ORDER.index(k.split(" · ")[2])))
    fig = go.Figure()
    for i, comp in enumerate(STAGE_COMPONENTS):
        vals = [float(np.median([r[comp] for r in groups[k]])) for k in keys]
        fig.add_trace(go.Bar(
            y=keys, x=vals, name=STAGE_LABELS[comp], orientation="h", marker_color=SERIES_LIGHT[i],
            marker_line=dict(color=SURFACE, width=2),
            hovertemplate="%{y}<br>" + STAGE_LABELS[comp] + ": %{x:.0f} ms (median)<extra></extra>",
        ))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="ms after source speech offset (median per group; components sum to end lag, arrival)")
    return _layout(fig, "Cascade: where the end lag goes", 140 + 34 * len(keys))


def timeline_plot(rows: list[dict], cmap: dict[str, str], gate: float) -> go.Figure | None:
    """One representative run per provider: the run with median end lag (playback)
    among its medium-category runs (falls back to any speech category)."""
    reps = []
    for prov in cmap:
        cand = [r for r in rows if r["provider_label"] == prov and r["status"] == "ok"
                and r.get("end_lag_playback_ms") is not None and r["category"] == "medium"]
        cand = cand or [r for r in rows if r["provider_label"] == prov and r["status"] == "ok"
                        and r.get("end_lag_playback_ms") is not None and r["category"] in EVS_CATEGORIES]
        if cand:
            cand.sort(key=lambda r: r["end_lag_playback_ms"])
            reps.append((prov, cand[len(cand) // 2]))
    if not reps:
        return None
    fig = make_subplots(rows=len(reps), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{p} — {r['clip_id']} (end lag {r['end_lag_playback_ms']:.0f} ms)" for p, r in reps])
    for i, (prov, r) in enumerate(reps, start=1):
        run = read_run(RESULTS_DIR / "raw" / f"{r['run_id']}.jsonl")
        pcm = sf.read(str(RESULTS_DIR / "audio" / run.footer.output_audio_file), dtype="int16")[0] \
            if run.footer.output_audio_file else None
        chunks = output_chunks(run, pcm, gate)
        spans = simulate_playback(chunks)
        t0 = run.sends.sent_ns[0]
        t_on, t_off = source_times(run)
        sec = lambda ns: (ns - t0) / 1e9  # noqa: E731
        fig.add_trace(go.Scatter(x=[sec(t_on), sec(t_off)], y=["source speech"] * 2, mode="lines",
                                 line=dict(color=INK2, width=10), name="source speech", showlegend=i == 1,
                                 hovertemplate="source speech %{x:.2f} s<extra></extra>"), row=i, col=1)
        aud = [c for c in chunks if c.first_audible is not None]
        fig.add_trace(go.Scatter(x=[sec(c.t_recv_ns) for c in aud], y=["output arrival"] * len(aud), mode="markers",
                                 marker=dict(color=cmap[prov], size=8, symbol="line-ns-open", line=dict(width=2)),
                                 name="output chunk arrival", showlegend=i == 1,
                                 hovertemplate="chunk arrived %{x:.3f} s<extra></extra>"), row=i, col=1)
        xs, ys = [], []
        for c, (s, e) in zip(chunks, spans):
            if c.first_audible is not None:
                xs += [sec(s), sec(e), None]
                ys += ["playback", "playback", None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=cmap[prov], width=10),
                                 name="simulated playback", showlegend=i == 1,
                                 hovertemplate="playing %{x:.2f} s<extra></extra>"), row=i, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=["playback", "output arrival", "source speech"],
                         row=i, col=1)
    fig.update_xaxes(title_text="seconds since first source chunk was sent", row=len(reps), col=1)
    for a in fig.layout.annotations:
        a.font = dict(size=12, color=INK2)
    return _layout(fig, "Timeline: source speech vs output arrival vs simulated playback", 170 * len(reps) + 120)


def behavior_section(rows: list[dict]) -> str:
    br = [r for r in rows if r["category"] in BEHAVIOR_CATEGORIES]
    if not br:
        return "<p class=muted>No same_language / silence / overlap runs in this report.</p>"
    br.sort(key=lambda r: (CATEGORY_ORDER.index(r["category"]), r["provider_label"], r["clip_id"], r["repetition"]))
    out = ["<table class=data><thead><tr><th>Category</th><th>Provider</th><th>Clip</th><th>Rep</th>"
           "<th>Output?</th><th class=n>Audible s</th><th>Provider transcript</th><th>ASR of output audio</th>"
           "<th>Audio</th></tr></thead><tbody>"]
    for r in br:
        if r["status"] != "ok":
            out.append(f"<tr><td>{r['category']}</td><td>{_fmt(r['provider_label'])}</td><td>{_fmt(r['clip_id'])}</td>"
                       f"<td>{r['repetition']}</td><td colspan=5 class=muted>{_fmt(r['status'])}: {_fmt(r.get('error'))}</td></tr>")
            continue
        audio = (f"<audio controls preload=none src='audio/{html.escape(r['output_audio_file'])}'></audio>"
                 f"<br><a href='audio/{html.escape(r['output_audio_file'])}'>wav</a>") if r.get("output_audio_file") else "—"
        out.append(
            f"<tr><td>{r['category']}</td><td>{_fmt(r['provider_label'])}</td><td>{_fmt(r['clip_id'])}</td>"
            f"<td>{r['repetition']}</td><td>{'yes' if r.get('produced_output') else 'no'}</td>"
            f"<td class=n>{_fmt(r.get('audible_output_s'), 2)}</td><td>{_fmt(r.get('provider_output_text') or '')}</td>"
            f"<td>{_fmt(r.get('asr_output_text') or '')}</td><td>{audio}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def methodology(cfg: BenchConfig) -> str:
    a = cfg.audio
    prov_cfg = {k: {kk: vv for kk, vv in v.items() if "key" not in kk.lower()} for k, v in cfg.providers.items()}
    llm = {k: {"provider": v.provider, "model": v.model, "base_url": v.base_url} for k, v in cfg.llm.items()}
    return f"""
<ul>
<li><b>Pacing.</b> Source audio is streamed at exactly 1× real time in {a.chunk_ms} ms PCM16 mono chunks on an
absolute schedule (chunk <i>i</i> is due at <i>t</i><sub>start</sub> + <i>i</i>·{a.chunk_ms} ms), so per-chunk
scheduling error never accumulates. Waits use a high-resolution timer thread plus a short final spin; the actual
send time of every chunk is recorded and used in all metrics. Each raw file records pacing quality
(max / p99 lateness, systematic drift).</li>
<li><b>One clock.</b> Every timestamp is <code>time.perf_counter_ns()</code> from the same process. Wall-clock time
is used only to name batches.</li>
<li><b>Padding.</b> Every clip gets {a.pad_pre_s:g} s of room tone before and {a.pad_post_s:g} s after
(pink-ish noise at {a.room_tone_dbfs:g} dBFS RMS, never digital zero), so providers have time to flush.
Prepared clips are stored at {a.prepared_sample_rate} Hz and resampled (soxr HQ) to each provider's required
input rate.</li>
<li><b>Source timing.</b> silero-vad (threshold {cfg.vad.threshold}, min silence {cfg.vad.min_silence_ms} ms,
speech pad {cfg.vad.speech_pad_ms} ms) on the exact padded clip finds speech onset/offset; each is mapped to the
send time of the chunk that carried that sample. A live microphone would add up to one chunk ({a.chunk_ms} ms) of
capture buffering on top of every lag reported here.</li>
<li><b>Audible output.</b> Output samples with |x| above {a.output_gate_dbfs:g} dBFS count as audible; leading
silent frames do not start the clock.</li>
<li><b>Session setup</b> = session ready − connect start (TCP + TLS + WebSocket + provider handshake). Reported
separately; never included in translation lag.</li>
<li><b>Start lag</b> = arrival of the first audible output sample − source speech onset.</li>
<li><b>End lag (arrival)</b> = arrival of the last audible output sample − source speech offset.</li>
<li><b>End lag (playback)</b> = a simulated listener plays output chunks back-to-back at 1× speed, each starting at
max(its arrival, end of the previous chunk); time the last audible sample finishes playing − source speech offset.
The closest proxy for “how long after they stop talking do I finish hearing it”.</li>
<li><b>EVS (approximate).</b> faster-whisper (<code>{cfg.analysis.whisper_model}</code>, word timestamps)
transcribes the source clip (mapped to send times) and the captured output (mapped to simulated playback times);
an LLM aligner (<code>{cfg.llm_cfg(cfg.analysis.evs.llm).model}</code>, cached) links source content words to target
words; EVS = target word playback start − source word send time. Mean and median per run.</li>
<li><b>Cascade stages</b> come from stage marks at each boundary (Deepgram Flux EndOfTurn, LLM request / first token /
done, Cartesia first text / first audio / done) for the segment that carries the last audible output.</li>
<li><b>Quality sanity:</b> chrF of the ASR transcript of the output audio vs. the reference translation
{'plus an LLM adequacy judge (1–5)' if cfg.analysis.quality.llm_judge else '(LLM judge disabled)'}; never combined
with latency.</li>
<li><b>Run protocol.</b> (provider, clip, repetition) order randomized with seed {cfg.seed}; concurrency
{cfg.concurrency}; {cfg.inter_run_delay_s:g} s between runs. Before each batch, TCP-connect and TLS-handshake times to
every provider host are measured ({cfg.network_probe.samples} samples, median shown below). Failures are recorded
and reported, never dropped.</li>
</ul>
<details><summary>Provider configuration used</summary><pre>{html.escape(json.dumps(prov_cfg, indent=1, default=str))}</pre></details>
<details><summary>LLM roles</summary><pre>{html.escape(json.dumps(llm, indent=1))}</pre></details>
<details><summary>config/bench.yaml (verbatim)</summary><pre>{html.escape(config_raw_text())}</pre></details>
"""


def environment(batches: list[str]) -> str:
    parts = []
    for b in batches:
        p = RESULTS_DIR / "raw" / f"batch_{b}.json"
        if not p.exists():
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        m = meta.get("machine", {})
        net_rows = "".join(
            f"<tr><td>{html.escape(h)}</td><td class=n>{_fmt(v.get('tcp_connect_ms_median'), 1)}</td>"
            f"<td class=n>{_fmt(v.get('tls_handshake_ms_median'), 1)}</td><td>{html.escape('; '.join(v.get('errors', [])))}</td></tr>"
            for h, v in meta.get("network", {}).items())
        skipped = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in meta.get("skipped", {}).items())
        parts.append(f"""
<h3>Batch {html.escape(b)}</h3>
<p>{html.escape(m.get('os', '?'))} · Python {html.escape(str(m.get('python')))} · {html.escape(str(m.get('processor')))}
· network: <b>{html.escape(str(m.get('network_interface')))}</b> · power: {html.escape(str(m.get('power_source', '?')))}
({html.escape(str(m.get('power_plan', '?')))}) · VPN adapters: {html.escape(', '.join(m.get('vpn_adapters', [])) or 'none')}
· location: {html.escape(str(m.get('location')))} · {meta.get('n_jobs')} runs, seed {meta.get('seed')}</p>
<table class=data><thead><tr><th>Host</th><th class=n>TCP connect (median ms)</th><th class=n>TLS handshake (median ms)</th>
<th>Errors</th></tr></thead><tbody>{net_rows or '<tr><td colspan=4 class=muted>no network probes</td></tr>'}</tbody></table>
{f'<p>Skipped providers:</p><ul>{skipped}</ul>' if skipped else ''}""")
    return "".join(parts) or "<p class=muted>No batch metadata found.</p>"


def failures(rows: list[dict]) -> str:
    bad = [r for r in rows if r["status"] in ("failed", "unsupported")]
    if not bad:
        return "<p>No failed or unsupported runs.</p>"
    counts: dict[tuple, int] = defaultdict(int)
    for r in bad:
        counts[(r["status"], r["provider_label"], r["pair"], (r.get("error") or "")[:160])] += 1
    items = "".join(f"<tr><td>{s}</td><td>{_fmt(p)}</td><td>{pair}</td><td class=n>{n}</td><td>{_fmt(e)}</td></tr>"
                    for (s, p, pair, e), n in sorted(counts.items()))
    return (f"<table class=data><thead><tr><th>Status</th><th>Provider</th><th>Pair</th><th class=n>Runs</th>"
            f"<th>Error</th></tr></thead><tbody>{items}</tbody></table>")


def pacing(rows: list[dict]) -> str:
    ok = [r for r in rows if r["status"] == "ok" and r.get("send_max_lateness_ms") is not None]
    if not ok:
        return ""
    worst = max(r["send_max_lateness_ms"] for r in ok)
    drift = max(abs(r.get("send_drift_ms") or 0) for r in ok)
    late = sum(1 for r in ok if r["send_max_lateness_ms"] > 20)
    return (f"<p>Source pacing across {len(ok)} runs: worst systematic drift {drift:.2f} ms; worst single-chunk "
            f"lateness {worst:.1f} ms; {late} run(s) had a chunk more than 20 ms late (OS preemption; lags are "
            f"still measured from the recorded actual send times).</p>")


CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--ring:rgba(11,11,11,.10)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10)}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10)}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:40px 0 8px;border-bottom:1px solid var(--grid);padding-bottom:6px}
h3{font-size:15px;margin:20px 0 6px} .muted,.note{color:var(--muted)} .note{font-size:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:12px;margin:12px 0;overflow-x:auto}
table.data{border-collapse:collapse;width:100%;font-size:13px} table.data th,table.data td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
table.data th{color:var(--ink2);font-weight:600} .n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
pre{white-space:pre-wrap;font-size:12px;background:var(--surface);padding:8px;border-radius:6px;border:1px solid var(--ring)}
audio{height:28px;max-width:220px} code{font-size:12px}
.wrap{overflow-x:auto}
"""


def build_report(cfg: BenchConfig, batches: list[str] | None = None, round2: bool = True,
                 rewrite_summary: bool = False) -> Path:
    """Round-1 sections from results/analysis/runs.jsonl (read-only) plus, when present, the
    round-2 section from results/round2/. results/summary.csv (a round-1 artifact) is only
    rewritten when rewrite_summary=True."""
    rows = load_rows()
    if batches:
        rows = [r for r in rows if r["batch_id"] in batches]
    if rewrite_summary:
        write_summary(summarize(rows))
    batch_ids = sorted({r["batch_id"] for r in rows})
    labels = sorted({r["provider_label"] for r in rows})
    cmap = _color_map(labels)
    figs = [box_plot(rows, cmap), stage_bars(rows), timeline_plot(rows, cmap, cfg.audio.output_gate_dbfs)]
    divs, first = [], True
    for f in figs:
        if f is None:
            continue
        divs.append("<div class=card>" + f.to_html(full_html=False, include_plotlyjs="inline" if first else False,
                                                    config={"displaylogo": False, "responsive": True}) + "</div>")
        first = False
    round2_section = ""
    if round2:
        from bench.report.round2 import round2_html

        round2_section = round2_html()
    n_ok = sum(r["status"] == "ok" for r in rows)
    synth = sorted({r["audio_source"] for r in rows})
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Translation Latency Report</title><style>{CSS}</style></head><body><main>
<h1>Live speech-to-speech translation latency</h1>
<p class=muted>{len(rows)} runs ({n_ok} ok) · providers: {html.escape(', '.join(labels))} · batches:
{html.escape(', '.join(batch_ids))} · audio sources: {html.escape(', '.join(synth))}.
Every number below is computed from timestamps recorded during these runs; nothing is vendor-claimed.
Results reflect the machine and network they were run on.</p>
<h2>Summary</h2><div class="card wrap">{summary_table(rows)}</div>
{''.join(divs) or '<p class=muted>No successful runs to chart.</p>'}
<h2>Behavior: same-language, silence, overlap</h2>
<p class=note>Does the provider stay silent, pass speech through, or re-speak it? Listen to the captured output.</p>
<div class="card wrap">{behavior_section(rows)}</div>
<h2>Failures and unsupported runs</h2><div class="card wrap">{failures(rows)}</div>
{round2_section}
<h2>Methodology</h2><div class=card>{methodology(cfg)}{pacing(rows)}</div>
<h2>Network and machine</h2><div class=card>{environment(batch_ids)}</div>
<p class=note>Per provider × pair × category statistics (median, p90, min, max, count) are in
<code>results/summary.csv</code>; per-run metrics in <code>results/analysis/runs.jsonl</code>; raw events in
<code>results/raw/</code>.</p>
</main>
<script>
// Dark mode: re-step chart chrome and series to the dark palette (same hues, dark steps).
(function(){{
  const root=document.documentElement, t=root.getAttribute('data-theme');
  const dark = t==='dark' || (t!=='light' && window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches);
  if(!dark || !window.Plotly) return;
  const L={json.dumps(SERIES_LIGHT)}, D={json.dumps(SERIES_DARK)};
  const map=c=>{{const i=L.indexOf((c||'').toLowerCase()); return i<0?(c==='#52514e'?'#c3c2b7':c):D[i];}};
  document.querySelectorAll('.js-plotly-plot').forEach(gd=>{{
    gd.data.forEach((tr,i)=>{{const u={{}};
      if(tr.marker&&typeof tr.marker.color==='string') u['marker.color']=map(tr.marker.color);
      if(tr.line&&tr.line.color) u['line.color']=map(tr.line.color);
      if(tr.marker&&tr.marker.line&&tr.marker.line.color==='#fcfcfb') u['marker.line.color']='#1a1a19';
      Plotly.restyle(gd,u,[i]);}});
    const ax={{gridcolor:'#2c2c2a',linecolor:'#383835',zerolinecolor:'#383835'}}, u={{paper_bgcolor:'#1a1a19',plot_bgcolor:'#1a1a19',
      'font.color':'#c3c2b7','title.font.color':'#ffffff','legend.font.color':'#c3c2b7','hoverlabel.bgcolor':'#2c2c2a','hoverlabel.font.color':'#ffffff'}};
    Object.keys(gd.layout).filter(k=>/^[xy]axis/.test(k)).forEach(k=>{{for(const [kk,v] of Object.entries(ax)) u[k+'.'+kk]=v;}});
    (gd.layout.annotations||[]).forEach((a,i)=>u['annotations['+i+'].font.color']='#c3c2b7');
    Plotly.relayout(gd,u);
  }});
}})();
</script>
</body></html>"""
    REPORT_PATH.write_text(page, encoding="utf-8")
    return REPORT_PATH
