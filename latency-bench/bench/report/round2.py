"""Round-2 section of results/report.html: experiment tables, box plots per condition,
rotation timelines (experiment 3) and the catch-up re-score table (experiment 1)."""

from __future__ import annotations

import html
import re
from pathlib import Path

import plotly.graph_objects as go

from bench.config import RESULTS_DIR
from bench.events import read_run
from bench.report.build import INK2, SERIES_LIGHT, SURFACE, _layout
from bench.round2.analyze import load_rows
from bench.round2.common import SECTIONS
from bench.round2.summaries import primary_end_lag

SECTION_ORDER = ["exp1", "exp2ab", "exp2c", "exp3", "exp4", "exp5", "exp6", "prep78"]
EXPERIMENT_ROWS = {"exp2c": ["exp2c"], "exp3": ["exp3"], "exp4": ["exp4"], "exp5": ["exp5a", "exp5b"], "exp6": ["exp6"]}


def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def md_to_html(md: str) -> str:
    """Small markdown subset: headings, paragraphs, bullet lists, pipe tables."""
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[-| ]+\|$", r)]
            head, body = cells[0], cells[1:]
            out.append("<div class='frame'><table class=data><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>" + "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table></div>")
            continue
        if ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
            continue
        m = re.match(r"^(#{2,4}) (.*)", ln)
        if m:
            lvl = len(m.group(1)) + 1
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def box_by_condition(rows: list[dict], title: str) -> go.Figure | None:
    ok = [r for r in rows if r["status"] == "ok" and primary_end_lag(r) is not None]
    if not ok:
        return None
    systems = sorted({r["system"] for r in ok})
    cats = [c for c in ("short", "medium", "noisy", "long", "two_utterances", "hesitation", "three_turns")
            if any(r["category"] == c for r in ok)]
    fig = go.Figure()
    for i, s in enumerate(systems):
        xs = [r["category"] for r in ok if r["system"] == s and r["category"] in cats]
        ys = [primary_end_lag(r) for r in ok if r["system"] == s and r["category"] in cats]
        fig.add_trace(go.Box(x=xs, y=ys, name=s, marker_color=SERIES_LIGHT[i % len(SERIES_LIGHT)], boxpoints="all",
                             jitter=0.35, pointpos=0, marker=dict(size=6, opacity=0.55), line=dict(width=2),
                             hovertemplate=f"{s}<br>%{{x}}: %{{y:.0f}} ms<extra></extra>"))
    fig.update_layout(boxmode="group")
    fig.update_xaxes(categoryorder="array", categoryarray=cats)
    fig.update_yaxes(title_text="End lag at playback (ms)", rangemode="tozero")
    return _layout(fig, title, 470)


def rotation_timelines(rows: list[dict]) -> go.Figure | None:
    """One representative run per flush policy on the two-utterance clip (else any clip)."""
    from bench.analysis.metrics import output_chunks, simulate_playback, utterance_times
    from bench.round2.common import load_pcm

    picks = []
    for s in sorted({r["system"] for r in rows}):
        cand = [r for r in rows if r["system"] == s and r["status"] == "ok" and r["category"] == "two_utterances"]
        cand = cand or [r for r in rows if r["system"] == s and r["status"] == "ok"]
        if cand:
            cand.sort(key=lambda r: primary_end_lag(r) or 0)
            picks.append((s, cand[len(cand) // 2]))
    if not picks:
        return None
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=len(picks), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"{s} — {r['clip_id']} ({r.get('rotations', 0)} rotations)" for s, r in picks])
    for i, (s, r) in enumerate(picks, start=1):
        run = read_run(RESULTS_DIR / "raw" / f"{r['run_id']}.jsonl")
        t0 = run.sends.sent_ns[0]
        sec = lambda ns: (ns - t0) / 1e9  # noqa: E731
        for a, b in utterance_times(run):
            fig.add_trace(go.Scatter(x=[sec(a), sec(b)], y=["source speech"] * 2, mode="lines",
                                     line=dict(color=INK2, width=9), showlegend=False,
                                     hovertemplate="speech %{x:.2f} s<extra></extra>"), row=i, col=1)
        chunks = output_chunks(run, load_pcm(run), -45.0)
        spans = simulate_playback(chunks)
        sess = [e.meta.get("session", 0) for e in run.events if e.kind == "audio_out" and e.n_samples]
        for k, (c, (a, b)) in enumerate(zip(chunks, spans)):
            if c.first_audible is None:
                continue
            col = SERIES_LIGHT[(sess[k] if k < len(sess) else 0) % len(SERIES_LIGHT)]
            fig.add_trace(go.Scatter(x=[sec(a), sec(b)], y=["playback (colour = session)"] * 2, mode="lines",
                                     line=dict(color=col, width=9), showlegend=False,
                                     hovertemplate=f"session {sess[k] if k < len(sess) else 0}<br>%{{x:.2f}} s<extra></extra>"),
                          row=i, col=1)
        for e in run.events:
            if e.stage in ("rotation.trigger", "rotation.promote", "openai.session.closed", "watchdog.arm", "vad.speech_end"):
                label = {"rotation.trigger": "flush", "rotation.promote": "promote", "openai.session.closed": "closed",
                         "watchdog.arm": "watchdog armed", "vad.speech_end": "VAD speech end"}[e.stage]
                fig.add_trace(go.Scatter(x=[sec(e.t_recv_ns)], y=["events"], mode="markers",
                                         marker=dict(size=9, symbol="line-ns-open", line=dict(width=2, color=INK2)),
                                         showlegend=False, hovertemplate=f"{label} %{{x:.2f}} s<extra></extra>"),
                              row=i, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=["playback (colour = session)", "events", "source speech"],
                         row=i, col=1)
    fig.update_xaxes(title_text="seconds since first source chunk was sent", row=len(picks), col=1)
    return _layout(fig, "Experiment 3: flush policies on one representative run each", 150 * len(picks) + 140)


def round2_html() -> str:
    parts = ["<h2>Round 2</h2>"]
    first_plot = True
    for name in SECTION_ORDER:
        p = SECTIONS / f"{name}.md"
        if not p.exists():
            continue
        parts.append("<div class=card>" + md_to_html(p.read_text(encoding="utf-8")) + "</div>")
        rows_name = EXPERIMENT_ROWS.get(name)
        figs = []
        if rows_name:
            rows = [r for n in rows_name for r in load_rows(n)]
            if rows:
                figs.append(box_by_condition(rows, f"{name}: end lag at playback by condition"))
                if name == "exp3":
                    figs.append(rotation_timelines(rows))
        for f in figs:
            if f is None:
                continue
            parts.append("<div class=card>" + f.to_html(full_html=False, include_plotlyjs=False,
                                                        config={"displaylogo": False, "responsive": True}) + "</div>")
            first_plot = False
    return "\n".join(parts)


def needs_plotly() -> bool:
    return any((SECTIONS / f"{n}.md").exists() for n in EXPERIMENT_ROWS)


def report_path() -> Path:
    return RESULTS_DIR / "report.html"
