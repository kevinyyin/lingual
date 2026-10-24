"""Runs the grammar and emotion readers on a finished signer turn (separate readers, never mixed)."""

from .emotion import read_emotion, signing_stats
from .grammar import read_grammar
from .protocol import Frame
from .session import Baseline, Emotion, Grammar


class FaceReaders:
    def read(self, frames: list[Frame], baseline: Baseline) -> tuple[Grammar, Emotion]:
        if baseline.speed is None and len(frames) > 1:
            baseline.speed, baseline.size = signing_stats(frames)  # the first turn sets "usual signing"
        return read_grammar(frames, baseline), read_emotion(frames, baseline)
