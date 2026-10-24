"""Causal voice detection on the audio exactly as it is sent (round 2, experiment 3).

Runs silero-vad's streaming VADIterator window by window (512 samples = 32 ms at 16 kHz)
on the stream as it leaves the harness, with no look-ahead. Input at any rate is
resampled to 16 kHz with a streaming soxr resampler. A `speech_end` fires once
`min_silence_ms` of non-speech has followed speech; it carries both the time it was
detected (the send time of the chunk that completed the silence) and the estimated
end-of-speech sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soxr

VAD_SR = 16000
WINDOW = 512


@dataclass
class VadEvent:
    kind: str            # "speech_start" | "speech_end"
    t_detect_ns: int     # when the harness knew (send time of the completing chunk)
    sample_16k: int      # estimated boundary, in 16 kHz samples since stream start


class CausalVAD:
    def __init__(self, input_sr: int, min_silence_ms: int = 600, threshold: float = 0.5):
        from silero_vad import VADIterator, load_silero_vad

        self.model = load_silero_vad(onnx=True)  # own instance: streaming state is per stream
        self.it = VADIterator(self.model, threshold=threshold, sampling_rate=VAD_SR,
                              min_silence_duration_ms=min_silence_ms, speech_pad_ms=0)
        self.rs = soxr.ResampleStream(input_sr, VAD_SR, 1, dtype="float32") if input_sr != VAD_SR else None
        self.buf = np.zeros(0, np.float32)
        self.min_silence_ms = min_silence_ms
        self.in_speech = False

    def push(self, pcm16: bytes, t_ns: int) -> list[VadEvent]:
        x = np.frombuffer(pcm16, "<i2").astype(np.float32) / 32768.0
        if self.rs is not None:
            x = self.rs.resample_chunk(x)
        self.buf = np.concatenate([self.buf, x])
        events: list[VadEvent] = []
        while len(self.buf) >= WINDOW:
            w, self.buf = self.buf[:WINDOW], self.buf[WINDOW:]
            d = self.it(w)
            if d:
                if "start" in d:
                    self.in_speech = True
                    events.append(VadEvent("speech_start", t_ns, int(d["start"])))
                if "end" in d:
                    self.in_speech = False
                    events.append(VadEvent("speech_end", t_ns, int(d["end"])))
        return events
