"""Cascade baseline: Deepgram Flux STT -> streaming LLM translation -> Cartesia Sonic TTS.

Every stage boundary emits a stage_mark so the report can show where time goes:

  stt.ready / stt.start_of_turn / stt.eager_eot / stt.turn_resumed / stt.end_of_turn {turn}
  llm.request {seg, turn, speculative} / llm.first_token {seg} / llm.done {seg}
  llm.cancelled {seg} / llm.speculative_hit {seg}
  tts.first_text {seg} / tts.first_audio {seg} / tts.done {seg}

Docs used (read 2026-10-13):
  Deepgram Flux  developers.deepgram.com/docs/flux/quickstart, /flux/state, /flux/close-stream,
                 /flux/force-end-turn, /flux/language-prompting
                 wss://api.deepgram.com/v2/listen?model=flux-general-multi&encoding=linear16&sample_rate=16000
                 header Authorization: Token <key>; server: Connected, TurnInfo{event,...}, FatalError
  Cartesia       docs.cartesia.ai api-reference/tts/websocket, tts-websocket/contexts, buffering
                 wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-08-14, header X-API-Key
                 server: {type: chunk|done|flush_done|timestamps|error, context_id, data(b64)}

Translation strategies implement `TranslateStrategy`; the runner never sees them:
  on_turn_end   translate the full utterance at EndOfTurn (optionally speculatively started
                at EagerEndOfTurn, held, and released at EndOfTurn if the transcript matches)
  incremental   translate stable transcript prefixes as they appear; translate the tail at EOT
  anticipation  NOT IMPLEMENTED - stub only (see README TODO)

Each segment gets its own Cartesia context so its audio is attributable exactly; a
client-side sequencer releases segment k's audio only after segment k-1 finished, as a
real player would (released chunks are stamped at release time).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import websockets

from bench.clock import now_ns
from bench.events import ProviderEvent
from bench.llm import LLMClient
from bench.providers.base import Provider, UnsupportedPair

LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "ja": "Japanese",
              "zh": "Chinese", "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "hi": "Hindi"}


@dataclass
class Segment:
    seg: int
    turn: int
    source: str
    speculative: bool = False
    held: bool = False
    text: str = ""
    pushed: bool = False
    llm_done: asyncio.Event = field(default_factory=asyncio.Event)
    tts_done: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    cancelled: bool = False
    final: bool = False                 # last fragment of its turn (closes a per-turn TTS context)
    text_done: asyncio.Event = field(default_factory=asyncio.Event)  # all its text pushed to TTS
    flush_seq: int = 0                  # per-turn context: this fragment's flush_id
    t_first_text: int | None = None
    preset: str | None = None           # anticipation: translation already decided, no LLM call
    already_spoken: str | None = None   # anticipation: translation spoken earlier in this turn

    @property
    def ctx(self) -> str:
        return f"seg-{self.seg}"


# ============================================================ translation strategies
class TranslateStrategy(ABC):
    """Decides WHAT to translate and WHEN. The host runs the LLM -> TTS pipeline."""

    def __init__(self, host: "CascadeProvider", cfg: dict[str, Any]):
        self.host = host
        self.cfg = cfg

    async def on_update(self, turn: int, transcript: str, words: list[str], t: int,
                        languages: list[str] | None = None) -> None: ...
    async def on_eager_eot(self, turn: int, transcript: str, t: int, words: list[str] | None = None) -> None: ...
    async def on_turn_resumed(self, turn: int, t: int) -> None: ...

    @abstractmethod
    async def on_end_of_turn(self, turn: int, transcript: str, words: list[str], languages: list[str], t: int) -> None: ...


class OnTurnEnd(TranslateStrategy):
    def __init__(self, host: "CascadeProvider", cfg: dict[str, Any]):
        super().__init__(host, cfg)
        self.use_eager = bool(cfg.get("use_eager_eot", True))
        self.spec: dict[int, Segment] = {}

    async def on_eager_eot(self, turn: int, transcript: str, t: int, words: list[str] | None = None) -> None:
        if not self.use_eager or not transcript.strip():
            return
        old = self.spec.pop(turn, None)
        if old:
            self.host.cancel_segment(old)
        self.spec[turn] = self.host.start_segment(turn, transcript, speculative=True, held=True, final=True)

    async def on_turn_resumed(self, turn: int, t: int) -> None:
        old = self.spec.pop(turn, None)
        if old:
            self.host.cancel_segment(old)

    async def on_end_of_turn(self, turn: int, transcript: str, words: list[str], languages: list[str], t: int) -> None:
        spec = self.spec.pop(turn, None)
        if self.host.skip_same_language(languages, turn):
            if spec:
                self.host.cancel_segment(spec)
            self.host.end_turn(turn)
            return
        if spec and spec.source.strip() == transcript.strip():
            self.host.mark("llm.speculative_hit", seg=spec.seg, turn=turn)
            self.host.release_segment(spec)
            return
        if spec:
            self.host.cancel_segment(spec)
        if transcript.strip():
            self.host.start_segment(turn, transcript, final=True)
        else:
            self.host.end_turn(turn)


def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


class Incremental(TranslateStrategy):
    """Simultaneous-interpretation style: commit stable transcript prefixes as fragments
    while the speaker is still talking; translate the remaining tail at end of turn.
    With use_eager_eot, the tail starts speculatively at EagerEndOfTurn (held, released at
    EndOfTurn if the transcript is unchanged), and no new fragments are committed while
    that speculative tail is pending."""

    def __init__(self, host: "CascadeProvider", cfg: dict[str, Any]):
        super().__init__(host, cfg)
        inc = cfg.get("incremental", {})
        self.k = int(inc.get("stable_updates", 2))
        self.holdback = int(inc.get("holdback_words", 2))
        self.min_words = int(inc.get("min_segment_words", 4))
        self.stable_ms = float(inc.get("stable_ms", 0) or 0)   # >0: time-based stability instead of update count
        self.use_eager = bool(cfg.get("use_eager_eot", True))
        self.history: dict[int, list[list[str]]] = {}
        self.since: dict[int, list[tuple[str, int]]] = {}     # turn -> per position (word, first seen ns)
        self.committed: dict[int, int] = {}
        self.spec: dict[int, tuple[Segment, int]] = {}   # turn -> (held tail segment, committed-at)

    def stable_prefix(self, turn: int, words: list[str], t: int) -> int:
        """Number of leading words considered stable.

        stable_ms > 0: a word is stable once the same word has stood at its position, with
        every earlier position also unchanged, for at least stable_ms.
        Otherwise: identical across the last `stable_updates` partial updates (round 1)."""
        if self.stable_ms > 0:
            prev = self.since.get(turn, [])
            cur: list[tuple[str, int]] = []
            unchanged = True  # a change at position i restarts the clock for i and everything after it
            for i, w in enumerate(words):
                nw = _norm(w)
                if unchanged and i < len(prev) and prev[i][0] == nw:
                    cur.append(prev[i])
                else:
                    unchanged = False
                    cur.append((nw, t))
            self.since[turn] = cur
            n = 0
            for _w, t0 in cur:
                if t - t0 >= self.stable_ms * 1e6:
                    n += 1
                else:
                    break
            return n
        hist = self.history.setdefault(turn, [])
        hist.append(words)
        if len(hist) < self.k:
            return 0
        recent = hist[-self.k:]
        stable = 0
        for i in range(min(len(w) for w in recent)):
            if len({_norm(w[i]) for w in recent}) == 1:
                stable += 1
            else:
                break
        return stable

    async def on_update(self, turn: int, transcript: str, words: list[str], t: int,
                        languages: list[str] | None = None) -> None:
        if turn in self.spec or self.host.is_same_language(languages or []):
            return
        stable = self.stable_prefix(turn, words, t)
        upto = stable - self.holdback
        done = self.committed.get(turn, 0)
        if upto - done >= self.min_words:
            self.committed[turn] = upto
            self.host.start_segment(turn, " ".join(words[done:upto]), incremental=True)

    async def on_eager_eot(self, turn: int, transcript: str, t: int, words: list[str] | None = None) -> None:
        if not self.use_eager or self.host.is_same_language(self.host.last_languages):
            return
        words = words or transcript.split()
        done = self.committed.get(turn, 0)
        tail = " ".join(words[done:])
        old = self.spec.pop(turn, None)
        if old:
            self.host.cancel_segment(old[0])
        if tail.strip():
            seg = self.host.start_segment(turn, tail, speculative=True, held=True, incremental=done > 0, final=True)
            self.spec[turn] = (seg, done)

    async def on_turn_resumed(self, turn: int, t: int) -> None:
        old = self.spec.pop(turn, None)
        if old:
            self.host.cancel_segment(old[0])

    async def on_end_of_turn(self, turn: int, transcript: str, words: list[str], languages: list[str], t: int) -> None:
        spec = self.spec.pop(turn, None)
        if self.host.skip_same_language(languages, turn):
            if spec:
                self.host.cancel_segment(spec[0])
            self.host.end_turn(turn)
            return
        words = words or transcript.split()
        done = self.committed.get(turn, 0)
        tail = " ".join(words[done:])
        if spec and spec[1] == done and spec[0].source.strip() == tail.strip():
            self.host.mark("llm.speculative_hit", seg=spec[0].seg, turn=turn)
            self.host.release_segment(spec[0])
        else:
            if spec:
                self.host.cancel_segment(spec[0])
            if tail.strip():
                self.host.start_segment(turn, tail, incremental=done > 0, final=True)
            else:
                self.host.end_turn(turn)
        self.history.pop(turn, None)
        self.since.pop(turn, None)


def longest_common_prefix(texts: list[str]) -> str:
    """Longest word-level common prefix of several translations.

    Words are compared case- and punctuation-insensitively; the returned words are taken
    from the first text, so its casing and punctuation are kept. Empty if texts disagree
    on the first word or the list is empty."""
    if not texts:
        return ""
    split = [t.split() for t in texts]
    out: list[str] = []
    for i, w in enumerate(split[0]):
        if all(i < len(o) and _norm(o[i]) == _norm(w) for o in split[1:]) and _norm(w):
            out.append(w)
        else:
            break
    return " ".join(out)


def strip_committed(prefix: str, committed: str) -> str:
    """Words of `prefix` beyond what was already committed (word-level, normalized)."""
    p, c = prefix.split(), committed.split()
    n = 0
    while n < len(c) and n < len(p) and _norm(p[n]) == _norm(c[n]):
        n += 1
    return " ".join(p[n:]) if n == len(c) else ""


class Anticipation(TranslateStrategy):
    """Anticipatory translation (prepared for experiments 7 and 8; NOT run in round 2).

    Behind a feature flag: requires `anticipation_enabled: true`.
      single_call  ask the LLM for only the translation prefix it is confident will not
                   change given the partial source; commit it as speech.
      sampled      ask for N short continuations of the partial source in parallel,
                   translate prefix+continuation for each, and commit the longest common
                   translated prefix beyond what is already committed.
    The rest of the turn is translated at end of turn as a continuation of what was spoken."""

    def __init__(self, host: "CascadeProvider", cfg: dict[str, Any]):
        super().__init__(host, cfg)
        an = cfg.get("anticipation", {})
        self.strategy = an.get("strategy", "single_call")
        self.n = int(an.get("samples", 3))
        self.min_new_words = int(an.get("min_new_words", 3))
        self.committed_tr: dict[int, str] = {}
        self.busy: dict[int, bool] = {}

    async def on_update(self, turn: int, transcript: str, words: list[str], t: int,
                        languages: list[str] | None = None) -> None:
        if self.busy.get(turn) or len(words) < 3 or self.host.is_same_language(languages or []):
            return
        self.busy[turn] = True
        self.host.tasks.append(asyncio.create_task(self._anticipate(turn, " ".join(words))))

    async def _anticipate(self, turn: int, partial: str) -> None:
        try:
            committed = self.committed_tr.get(turn, "")
            if self.strategy == "sampled":
                conts = await asyncio.gather(*(self.host.llm_complete(
                    "Continue the speaker's unfinished utterance in the same language with a short, plausible "
                    "ending (at most 8 words). Output only the continuation.", partial) for _ in range(self.n)))
                trans = await asyncio.gather(*(self.host.llm_translate(f"{partial} {c}") for c in conts))
                prefix = longest_common_prefix(list(trans))
            else:
                prefix = await self.host.llm_complete(
                    self.host.anticipation_prompt(), f"Partial source (still being spoken): {partial}\n"
                    f"Already spoken translation: {committed}")
            new = strip_committed(prefix, committed) if committed else prefix
            if len(new.split()) >= self.min_new_words:
                self.committed_tr[turn] = (committed + " " + new).strip()
                self.host.start_segment(turn, partial, preset=new)
        finally:
            self.busy[turn] = False

    async def on_end_of_turn(self, turn: int, transcript: str, words: list[str], languages: list[str], t: int) -> None:
        if self.host.skip_same_language(languages, turn):
            self.host.end_turn(turn)
            return
        committed = self.committed_tr.pop(turn, "")
        if not transcript.strip():
            self.host.end_turn(turn)
            return
        self.host.start_segment(turn, transcript, final=True, already_spoken=committed or None)


STRATEGIES: dict[str, type[TranslateStrategy]] = {
    "on_turn_end": OnTurnEnd, "incremental": Incremental, "anticipation": Anticipation,
}


# ============================================================ provider
@dataclass
class TurnCtx:
    """One Cartesia context per turn (tts_context: per_turn)."""

    turn: int
    uid: str
    segs: list[Segment] = field(default_factory=list)
    pushed: bool = False
    closed: bool = False
    done: asyncio.Event = field(default_factory=asyncio.Event)
    transcripts: int = 0  # transcripts (fragments) sent; flush_id of fragment k is k (1-based)


@dataclass
class Unit:
    """A playback-ordered output unit: one segment (per_fragment) or one turn context (per_turn)."""

    uid: str
    done: asyncio.Event = field(default_factory=asyncio.Event)


CONCISE_PROMPT = ("Speak like a skilled conference interpreter: natural, concise spoken {t}, using about 10% fewer "
                  "syllables than a literal translation, but never drop any content, name or number.")


class CascadeProvider(Provider):
    name = "cascade"
    hosts = ("api.deepgram.com", "api.cartesia.ai")

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.input_sample_rate = int(cfg.get("input_sample_rate", 16000))
        self.stt_cfg = cfg.get("stt", {})
        self.tts_cfg = cfg.get("tts", {})
        bench = cfg.get("_bench")
        self.llm = LLMClient(bench.llm_cfg(cfg.get("llm", "translate"))) if bench else None
        self.mode = cfg.get("translate_mode", "on_turn_end")
        # round-2 options (defaults = round-1 behaviour)
        self.tts_context = cfg.get("tts_context", "per_fragment")
        self.flush_each = bool(cfg.get("tts_flush_each", False))
        self.prompt_variant = cfg.get("prompt_variant", "literal")
        self.pipeline = bool(cfg.get("pipeline_fragments", False))
        ad = cfg.get("adaptive_speed") or {}
        self.adaptive = bool(ad.get("enabled", False))
        self.ad_up_ms, self.ad_up2_ms = float(ad.get("up_ms", 500)), float(ad.get("up2_ms", 1000))
        self.ad_down_ms, self.ad_speed = float(ad.get("down_ms", 250)), float(ad.get("speed", 1.2))
        self.ad_cap = min(float(ad.get("cap", 1.3)), 1.5)  # Cartesia accepts 0.6-1.5
        self.speed = 1.0
        self._play_end_ns = 0
        self.strategy = STRATEGIES[self.mode](self, cfg)
        self.stt: websockets.ClientConnection | None = None
        self.tts: websockets.ClientConnection | None = None
        self.tasks: list[asyncio.Task] = []
        self.segments: list[Segment] = []
        self.by_ctx: dict[str, Segment] = {}
        self.turn_ctx: dict[int, TurnCtx] = {}
        self.ctx_by_uid: dict[str, TurnCtx] = {}
        self.units: list[Unit] = []
        self.unit_of: dict[str, Unit] = {}
        self.buffered: dict[str, list[tuple[bytes, int, dict]]] = {}
        self.in_turn = False
        self.last_eot = asyncio.Event()
        self.src = self.tgt = ""
        self.same_language_policy = cfg.get("same_language", "silent")
        self.translated_so_far: dict[int, list[tuple[str, str]]] = {}
        self._seen_audio: set[int] = set()
        self.last_languages: list[str] = []
        self._aclosed = False

    def public_config(self) -> dict[str, Any]:
        d = super().public_config()
        if self.llm:
            d["llm"] = {"provider": self.llm.cfg.provider, "model": self.llm.cfg.model, "base_url": self.llm.cfg.base_url}
        return d

    @classmethod
    def check_available(cls, cfg: dict[str, Any]) -> str | None:
        missing = [k for k in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY") if not os.environ.get(k)]
        bench = cfg.get("_bench")
        if bench:
            env = bench.llm_cfg(cfg.get("llm", "translate")).api_key_env
            if not os.environ.get(env):
                missing.append(env)
        if cfg.get("translate_mode") == "anticipation" and not cfg.get("anticipation_enabled"):
            return "translate_mode 'anticipation' is behind a feature flag (anticipation_enabled: true)"
        return f"{', '.join(missing)} not set in latency-bench/.env" if missing else None

    # ------------------------------------------------------------ helpers
    def mark(self, stage: str, t: int | None = None, **meta: Any) -> None:
        self._mark(stage, t, **meta)

    def is_same_language(self, languages: list[str]) -> bool:
        return (self.same_language_policy == "silent" and bool(languages)
                and languages[0].split("-")[0] == self.tgt and self.src != self.tgt)

    def skip_same_language(self, languages: list[str], turn: int) -> bool:
        if self.is_same_language(languages):
            self.mark("cascade.skip_same_language", turn=turn, languages=languages)
            return True
        return False

    # ------------------------------------------------------------ connect
    async def connect(self, source_lang: str, target_lang: str) -> None:
        self.src, self.tgt = source_lang, target_lang
        model = self.stt_cfg.get("model", "flux-general-multi")
        supported = self.stt_cfg.get("supported_languages", [])
        if (model == "flux-general-en" and source_lang != "en") or (supported and source_lang not in supported):
            raise UnsupportedPair(f"Deepgram {model} does not support source language '{source_lang}'")
        params: list[tuple[str, Any]] = [
            ("model", model), ("encoding", "linear16"), ("sample_rate", self.input_sample_rate),
            ("eot_threshold", self.stt_cfg.get("eot_threshold", 0.7)),
            ("eot_timeout_ms", self.stt_cfg.get("eot_timeout_ms", 5000)),
        ]
        if self.stt_cfg.get("eager_eot_threshold") is not None:
            params.append(("eager_eot_threshold", self.stt_cfg["eager_eot_threshold"]))
        if model == "flux-general-multi":
            # a shared-mic product expects both participants' languages
            for lang in dict.fromkeys([source_lang, target_lang]):
                if not supported or lang in supported:
                    params.append(("language_hint", lang))
        stt_url = f"{self.stt_cfg.get('url', 'wss://api.deepgram.com/v2/listen')}?{urlencode(params)}"
        tts_url = f"{self.tts_cfg.get('url', 'wss://api.cartesia.ai/tts/websocket')}?" + urlencode(
            {"cartesia_version": self.tts_cfg.get("cartesia_version", "2026-08-14")})
        stt_ready = asyncio.Event()

        async def open_stt() -> None:
            self.stt = await websockets.connect(stt_url, additional_headers={
                "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}, max_size=None, open_timeout=20)
            self.tasks.append(asyncio.create_task(self._stt_loop(stt_ready)))
            await asyncio.wait_for(stt_ready.wait(), 20)

        async def open_tts() -> None:
            self.tts = await websockets.connect(tts_url, additional_headers={
                "X-API-Key": os.environ["CARTESIA_API_KEY"]}, max_size=None, open_timeout=20)
            self.tasks.append(asyncio.create_task(self._tts_loop()))
            self.mark("tts.ready")

        warm = self.llm.warm() if self.llm else asyncio.sleep(0)
        await asyncio.gather(open_stt(), open_tts(), warm)
        self.mark("llm.warmed")
        self._emit(ProviderEvent("session_ready", now_ns()))

    # ------------------------------------------------------------ STT
    async def _stt_loop(self, ready: asyncio.Event) -> None:
        assert self.stt is not None
        try:
            async for raw in self.stt:
                t = now_ns()
                msg = json.loads(raw)
                typ = msg.get("type")
                if typ == "Connected":
                    self.mark("stt.ready", t)
                    ready.set()
                elif typ == "TurnInfo":
                    await self._on_turn_info(msg, t)
                elif typ in ("FatalError", "Error"):
                    self._emit(ProviderEvent("error", t, len(raw), text=json.dumps(msg)[:500], meta={"fatal": True}))
                else:
                    self.mark(f"stt.{typ}", t)
        except websockets.ConnectionClosed as e:
            self.mark("stt.closed", detail=str(e)[:200])
        except Exception as e:  # surface strategy errors instead of hanging
            self._emit(ProviderEvent("error", now_ns(), text=f"stt loop: {type(e).__name__}: {e}", meta={"fatal": True}))

    async def _on_turn_info(self, msg: dict[str, Any], t: int) -> None:
        ev, turn = msg.get("event"), int(msg.get("turn_index", 0))
        transcript = msg.get("transcript", "") or ""
        words = [w.get("word", "") for w in msg.get("words", []) or []]
        langs = msg.get("languages") or []
        if langs:
            self.last_languages = langs
        if ev == "StartOfTurn":
            self.in_turn = True
            self.mark("stt.start_of_turn", t, turn=turn)
            await self.strategy.on_update(turn, transcript, words, t, langs)
        elif ev == "Update":
            if transcript:
                self.mark("stt.update", t, turn=turn, transcript=transcript)
            await self.strategy.on_update(turn, transcript, words, t, langs)
        elif ev == "EagerEndOfTurn":
            self.mark("stt.eager_eot", t, turn=turn, transcript=transcript)
            await self.strategy.on_eager_eot(turn, transcript, t, words)
        elif ev == "TurnResumed":
            self.mark("stt.turn_resumed", t, turn=turn)
            await self.strategy.on_turn_resumed(turn, t)
        elif ev == "EndOfTurn":
            self.in_turn = False
            self.mark("stt.end_of_turn", t, turn=turn, transcript=transcript, trigger=msg.get("trigger"), languages=langs)
            if transcript:
                self._emit(ProviderEvent("source_text_delta", t, text=transcript + " ", meta={"turn": turn}))
            await self.strategy.on_end_of_turn(turn, transcript, words, langs, t)
            self.last_eot.set()
        else:
            self.mark(f"stt.{ev}", t, turn=turn)

    # ------------------------------------------------------------ units (playback order)
    def _new_unit(self, uid: str) -> Unit:
        u = Unit(uid)
        self.units.append(u)
        self.unit_of[uid] = u
        return u

    def _turn(self, turn: int) -> TurnCtx:
        tc = self.turn_ctx.get(turn)
        if tc is None:
            tc = TurnCtx(turn, f"turn-{turn}")
            self.turn_ctx[turn] = tc
            self.ctx_by_uid[tc.uid] = tc
            self._new_unit(tc.uid)
        return tc

    def _unit_done(self, uid: str) -> None:
        u = self.unit_of.get(uid)
        if u is not None and not u.done.is_set():
            u.done.set()
            self._advance_sequencer()

    def end_turn(self, turn: int) -> None:
        """A turn ended without a final (tail) segment: close its context once earlier text is pushed."""
        if self.tts_context != "per_turn" or turn not in self.turn_ctx:
            return
        tc = self.turn_ctx[turn]
        segs = [s for s in tc.segs if not s.cancelled]

        async def close() -> None:
            for s in segs:
                await s.text_done.wait()
            await self._close_turn(tc)

        self.tasks.append(asyncio.create_task(close()))

    async def _close_turn(self, tc: TurnCtx) -> None:
        if tc.closed:
            return
        tc.closed = True
        if tc.pushed:
            await self._tts_send(tc.uid, "", cont=False)
            self.mark("tts.context_close_sent", turn=tc.turn)
        else:  # nothing was ever spoken in this turn
            tc.done.set()
            for s in tc.segs:
                s.tts_done.set()
            self._unit_done(tc.uid)

    # ------------------------------------------------------------ segments (LLM -> TTS)
    def start_segment(self, turn: int, source: str, speculative: bool = False, held: bool = False,
                      incremental: bool = False, final: bool = False, preset: str | None = None,
                      already_spoken: str | None = None) -> Segment:
        seg = Segment(seg=len(self.segments), turn=turn, source=source, speculative=speculative, held=held)
        seg.final = final
        seg.preset = preset
        seg.already_spoken = already_spoken
        if not held:
            seg.released.set()
        prev_turn = [s for s in self.segments if s.turn == turn and not s.cancelled]
        self.segments.append(seg)
        self.by_ctx[seg.ctx] = seg
        if self.tts_context == "per_turn":
            self._turn(turn).segs.append(seg)
        else:
            self._new_unit(seg.ctx)
        prev = self.segments[-2] if len(self.segments) > 1 else None
        gate = prev_turn[-1] if prev_turn else None
        seg.task = asyncio.create_task(self._run_segment(seg, prev if incremental else None, gate))
        return seg

    def release_segment(self, seg: Segment) -> None:
        seg.held = False
        seg.released.set()

    def cancel_segment(self, seg: Segment) -> None:
        seg.cancelled = True
        if seg.task and not seg.task.done():
            seg.task.cancel()
        self.mark("llm.cancelled", seg=seg.seg, turn=seg.turn)
        seg.llm_done.set()
        seg.text_done.set()
        seg.tts_done.set()
        if self.tts_context == "per_turn":
            tc = self.turn_ctx.get(seg.turn)
            if tc is not None and seg in tc.segs:
                tc.segs.remove(seg)
        else:
            self._unit_done(seg.ctx)

    def _system_prompt(self) -> str:
        s, t = LANG_NAMES.get(self.src, self.src), LANG_NAMES.get(self.tgt, self.tgt)
        system = (f"You are a professional simultaneous interpreter. Translate spoken {s} into natural spoken {t}. "
                  f"Output ONLY the {t} translation, with no quotes, notes or explanations. "
                  f"If the input is already in {t}, output it unchanged.")
        if self.prompt_variant == "concise":
            system += " " + CONCISE_PROMPT.format(t=t)
        return system

    def anticipation_prompt(self) -> str:
        t = LANG_NAMES.get(self.tgt, self.tgt)
        return (self._system_prompt() + f" The source is still being spoken and may continue. Output ONLY the "
                f"part of the {t} translation that you are confident will not change however the sentence ends, "
                f"continuing the already spoken translation. Output nothing if unsure.")

    async def llm_complete(self, system: str, user: str) -> str:
        assert self.llm is not None
        return (await self.llm.complete(system, user)).strip()

    async def llm_translate(self, text: str) -> str:
        return await self.llm_complete(self._system_prompt(), text)

    def _prompts(self, seg: Segment, prev_pairs: list[tuple[str, str]]) -> tuple[str, str]:
        t = LANG_NAMES.get(self.tgt, self.tgt)
        system = self._system_prompt()
        if seg.already_spoken:
            return system, (f"Full source utterance: {seg.source}\nAlready spoken translation: {seg.already_spoken}\n"
                            f"Output ONLY the rest of the {t} translation, continuing the spoken part naturally.")
        if not prev_pairs:
            return system, seg.source
        ps = " ".join(p[0] for p in prev_pairs)
        pt = " ".join(p[1] for p in prev_pairs)
        user = (f"An utterance is being interpreted in fragments.\nSource so far: {ps}\nTranslation so far: {pt}\n"
                f"Next source fragment: {seg.source}\n"
                f"Output ONLY the {t} translation of the next fragment, continuing the translation so far naturally.")
        return system, user

    async def _run_segment(self, seg: Segment, prev: Segment | None, gate: Segment | None) -> None:
        """LLM -> TTS for one fragment. `prev` (same turn, incremental) supplies context;
        `gate` (previous fragment of the same turn) must finish pushing text first."""
        try:
            if prev is not None and not self.pipeline:
                await prev.llm_done.wait()           # serial: context = finished translations
                pairs = list(self.translated_so_far.get(seg.turn, []))
            elif prev is not None:                   # pipelined: committed source + translation so far
                pairs = [(s.source, s.text) for s in self.segments
                         if s.turn == seg.turn and s.seg < seg.seg and not s.cancelled]
            else:
                pairs = []
            system, user = self._prompts(seg, pairs)
            assert self.llm is not None
            self.mark("llm.request", seg=seg.seg, turn=seg.turn, speculative=seg.speculative, source=seg.source,
                      pipelined=self.pipeline and prev is not None, preset=seg.preset is not None)
            pending: list[str] = []
            first = True

            async def tokens():
                if seg.preset is not None:
                    yield seg.preset + " "
                    return
                async for x in self.llm.stream(system, user):
                    yield x

            async for tok in tokens():
                if first:
                    self.mark("llm.first_token", seg=seg.seg)
                    first = False
                seg.text += tok
                if not seg.released.is_set() or (gate is not None and not gate.text_done.is_set()):
                    pending.append(tok)
                    continue
                if pending:
                    await self._tts_push(seg, "".join(pending))
                    pending.clear()
                await self._tts_push(seg, tok)
            self.mark("llm.done", seg=seg.seg, text=seg.text)
            seg.llm_done.set()
            self.translated_so_far.setdefault(seg.turn, []).append((seg.source, seg.text))
            await seg.released.wait()
            if gate is not None:
                await gate.text_done.wait()
            if pending:
                await self._tts_push(seg, "".join(pending))
            await self._finish_segment_text(seg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._emit(ProviderEvent("error", now_ns(), text=f"segment {seg.seg}: {type(e).__name__}: {e}",
                                     meta={"fatal": True}))
            seg.llm_done.set()
            seg.text_done.set()
            seg.tts_done.set()
            self._unit_done(seg.ctx if self.tts_context != "per_turn" else f"turn-{seg.turn}")

    async def _finish_segment_text(self, seg: Segment) -> None:
        if self.tts_context == "per_turn":
            tc = self._turn(seg.turn)
            if seg.pushed:
                self._emit(ProviderEvent("text_out_delta", now_ns(), text=seg.text.strip() + " ", meta={"seg": seg.seg}))
            if seg.final:
                seg.text_done.set()
                await self._close_turn(tc)
            else:
                if seg.pushed and self.flush_each:
                    await self._tts_send(tc.uid, "", cont=True, flush=True)
                    self.mark("tts.flush_sent", seg=seg.seg, turn=seg.turn)
                seg.text_done.set()
            return
        # per_fragment (round 1): one context per segment
        seg.text_done.set()
        if seg.pushed:
            await self._tts_send(seg.ctx, "", cont=False)
            self._emit(ProviderEvent("text_out_delta", now_ns(), text=seg.text.strip() + " ", meta={"seg": seg.seg}))
        else:
            seg.tts_done.set()
            self._unit_done(seg.ctx)

    # ------------------------------------------------------------ TTS
    def _voice(self) -> str:
        return dict(self.tts_cfg.get("voices") or {}).get(self.tgt) or self.tts_cfg.get("default_voice_id")

    def _update_speed(self) -> None:
        if not self.adaptive:
            return
        queued_ms = max(0.0, (self._play_end_ns - now_ns()) / 1e6)
        new = self.speed
        if queued_ms > self.ad_up2_ms:
            new = self.ad_cap
        elif queued_ms > self.ad_up_ms and self.speed < self.ad_speed:
            new = self.ad_speed
        elif queued_ms < self.ad_down_ms:
            new = 1.0
        if new != self.speed:
            self.mark("tts.speed_change", speed=new, previous=self.speed, queued_ms=round(queued_ms, 1))
            self.speed = new

    async def _tts_send(self, context_id: str, text: str, cont: bool, flush: bool = False) -> None:
        assert self.tts is not None
        msg: dict[str, Any] = {
            "model_id": self.tts_cfg.get("model_id", "sonic-3.6"),
            "transcript": text,
            "voice": {"id": self._voice()},
            "output_format": {"container": "raw", "encoding": "pcm_s16le",
                              "sample_rate": int(self.tts_cfg.get("sample_rate", 24000))},
            "language": self.tgt,
            "context_id": context_id,
            "continue": cont,
            "max_buffer_delay_ms": int(self.tts_cfg.get("max_buffer_delay_ms", 1000)),
        }
        if flush:
            msg["flush"] = True
        if self.adaptive:
            msg["generation_config"] = {"speed": self.speed}
        await self.tts.send(json.dumps(msg))

    async def _tts_push(self, seg: Segment, text: str) -> None:
        if not text:
            return
        if not seg.pushed:
            seg.pushed = True
            seg.t_first_text = now_ns()
            self.mark("tts.first_text", seg=seg.seg)
            self._update_speed()
            if self.tts_context == "per_turn":
                tc = self._turn(seg.turn)
                tc.transcripts += 1
                seg.flush_seq = tc.transcripts
        if self.tts_context == "per_turn":
            tc = self._turn(seg.turn)
            tc.pushed = True
            await self._tts_send(tc.uid, text, cont=True)
        else:
            await self._tts_send(seg.ctx, text, cont=True)

    def _attribute(self, tc: TurnCtx, msg: dict[str, Any], t: int) -> Segment | None:
        """Which fragment produced a chunk in a per-turn context: exact with flush_id,
        otherwise the latest fragment whose text had been sent (approximate)."""
        segs = [s for s in tc.segs if s.pushed]
        if not segs:
            return None
        # flush_id identifies a fragment only when WE flush after each fragment; with a short
        # buffer delay Cartesia flushes on its own (observed: one flush per LLM token at 0 ms)
        fid = msg.get("flush_id") if self.flush_each else None
        if fid is not None:
            hit = [s for s in segs if s.flush_seq == int(fid)]
            if hit:
                return hit[0]
        sent = [s for s in segs if s.t_first_text is not None and s.t_first_text <= t]
        return sent[-1] if sent else segs[0]

    async def _tts_loop(self) -> None:
        assert self.tts is not None
        sr = int(self.tts_cfg.get("sample_rate", 24000))
        try:
            async for raw in self.tts:
                t = now_ns()
                msg = json.loads(raw)
                cid = msg.get("context_id", "")
                typ = msg.get("type")
                if self.tts_context == "per_turn":
                    tc = self.ctx_by_uid.get(cid)
                    if tc is None:
                        if typ == "error":
                            self._emit(ProviderEvent("error", t, len(raw), text=json.dumps(msg)[:500], meta={"fatal": True}))
                        continue
                    if typ == "chunk" and msg.get("data"):
                        pcm = base64.b64decode(msg["data"])
                        seg = self._attribute(tc, msg, t)
                        if seg is not None and seg.seg not in self._seen_audio:
                            self._seen_audio.add(seg.seg)
                            self.mark("tts.first_audio", t, seg=seg.seg, attributed="flush_id" if "flush_id" in msg else "approx")
                        self._deliver(tc.uid, pcm, sr, t, {"seg": seg.seg if seg else None, "turn": tc.turn})
                    elif typ == "flush_done":
                        self.mark("tts.flush_done", t, turn=tc.turn, flush_id=msg.get("flush_id"))
                    elif typ == "done" or msg.get("done") is True:
                        for s in tc.segs:
                            self.mark("tts.done", t, seg=s.seg)
                            s.tts_done.set()
                        tc.done.set()
                        self._unit_done(tc.uid)
                    elif typ == "error":
                        self._emit(ProviderEvent("error", t, len(raw), text=json.dumps(msg)[:500], meta={"fatal": True}))
                        for s in tc.segs:
                            s.tts_done.set()
                        self._unit_done(tc.uid)
                    continue
                seg = self.by_ctx.get(cid)
                if seg is None or seg.cancelled:
                    if typ == "error":
                        self._emit(ProviderEvent("error", t, len(raw), text=json.dumps(msg)[:500], meta={"fatal": True}))
                    continue
                if typ == "chunk" and msg.get("data"):
                    pcm = base64.b64decode(msg["data"])
                    if seg.seg not in self._seen_audio:
                        self._seen_audio.add(seg.seg)
                        self.mark("tts.first_audio", t, seg=seg.seg)
                    self._deliver(seg.ctx, pcm, sr, t, {"seg": seg.seg})
                elif typ == "done" or msg.get("done") is True:
                    self.mark("tts.done", t, seg=seg.seg)
                    seg.tts_done.set()
                    self._unit_done(seg.ctx)
                elif typ == "error":
                    self._emit(ProviderEvent("error", t, len(raw), text=json.dumps(msg)[:500], meta={"fatal": True}))
                    seg.tts_done.set()
                    self._unit_done(seg.ctx)
        except websockets.ConnectionClosed as e:
            self.mark("tts.closed", detail=str(e)[:200])

    # ------------------------------------------------------------ playback-order sequencer
    def _current_unit(self) -> Unit | None:
        for u in self.units:
            if not u.done.is_set():
                return u
        return None

    def _emit_audio(self, pcm: bytes, sr: int, t: int, meta: dict) -> None:
        dur = int(len(pcm) / 2 / sr * 1e9)
        self._play_end_ns = max(t, self._play_end_ns) + dur  # client playback clock (adaptive speed)
        self._emit(ProviderEvent("audio_out", t, len(pcm), pcm=pcm, sample_rate=sr, meta=meta))

    def _deliver(self, uid: str, pcm: bytes, sr: int, t: int, meta: dict) -> None:
        cur = self._current_unit()
        if cur is not None and cur.uid == uid:
            self._emit_audio(pcm, sr, t, meta)
        else:
            self.buffered.setdefault(uid, []).append((pcm, t, meta))

    def _advance_sequencer(self) -> None:
        sr = int(self.tts_cfg.get("sample_rate", 24000))
        while True:
            cur = self._current_unit()
            if cur is None or cur.uid not in self.buffered:
                return
            t_rel = now_ns()
            for pcm, t_arr, meta in self.buffered.pop(cur.uid):
                self._emit_audio(pcm, sr, t_rel, {**meta, "buffered": True, "t_arrived_ns": t_arr})
            if not cur.done.is_set():
                return

    # ------------------------------------------------------------ audio in / finish
    async def send_audio(self, pcm16_chunk: bytes, t_sent_ns: int) -> None:
        assert self.stt is not None
        await self.stt.send(pcm16_chunk)

    async def finish(self) -> None:
        assert self.stt is not None
        if self.in_turn:
            self.last_eot.clear()
            await self.stt.send(json.dumps({"type": "ForceEndTurn"}))
            self.mark("cascade.force_end_turn")
            try:
                await asyncio.wait_for(self.last_eot.wait(), 5)
            except asyncio.TimeoutError:
                pass
        await self.stt.send(json.dumps({"type": "CloseStream"}))
        self.mark("stt.close_sent")
        for s in self.segments:  # a speculative draft whose EndOfTurn never came is discarded
            if s.held and not s.released.is_set() and not s.cancelled:
                self.cancel_segment(s)
        if self.tts_context == "per_turn":  # contexts whose turn never got a final segment
            for tc in list(self.turn_ctx.values()):
                if not tc.closed and not any(s.final and not s.cancelled for s in tc.segs):
                    self.end_turn(tc.turn)
        deadline = float(self.cfg.get("drain_max_s", 30.0))
        try:
            await asyncio.wait_for(asyncio.gather(*(s.tts_done.wait() for s in self.segments)), deadline)
        except asyncio.TimeoutError:
            self._emit(ProviderEvent("error", now_ns(), text="cascade drain timeout", meta={"fatal": False}))
        self.mark("cascade.drained")
        await self.aclose()

    async def aclose(self) -> None:
        if self._aclosed:
            self._emit_closed()
            return
        self._aclosed = True
        for s in self.segments:
            if s.task and not s.task.done():
                s.task.cancel()
        for ws in (self.stt, self.tts):
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
        for t in self.tasks:
            if not t.done():
                t.cancel()
        if self.llm:
            await self.llm.aclose()
        self._emit_closed()
