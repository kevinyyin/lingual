"""Reader for NCSLGR SignStream 2 database files (the public 2011 archive on the BU
download page). Only what the spike needs: per utterance, the main-gloss sequence,
the English translation, and which grammatical non-manual tracks are present
(negative, wh question, yes-no question, rhetorical question) with time spans.

Format notes (reverse-engineered against the three XML exports that ship with the
public SignStream XML parser; validated in validate_ss2()):
  * strings are 1-byte length-prefixed, followed by 00 00 00 16 00 01
  * a track starts with 00 05 00 00 <field id: u16 BE> 00 01 00 00
  * times are u32 BE in 1/600 s (QuickTime time scale 600 = 00 00 02 58)
  * an utterance record ends with its English translation string
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

NMM_FIELDS = {16: "negative", 17: "wh", 18: "yn", 19: "rhq", 20: "topic", 21: "cond", 5: "headshake", 10: "brows"}
STR_RE = re.compile(rb"(?<=[\x00-\xff])([\x01-\xff])")
TRACK_RE = re.compile(rb"\x00\x05\x00\x00(.{2})\x00\x01\x00\x00", re.S)
TIME_RE = re.compile(rb"\x00\x00(.{2})\x00\x00\x00\x00\x00\x00(.{2})\x00\x00\x02\x58", re.S)


@dataclass
class Utt:
    glosses: list[str] = field(default_factory=list)
    english: str = ""
    tracks: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    start_s: float | None = None
    end_s: float | None = None


def _strings(b: bytes):
    """Yield (offset, text) for length-prefixed strings followed by the 00 00 00 16 00 01 tag."""
    for m in re.finditer(rb"\x00\x00\x00\x16\x00\x01", b):
        end = m.start()
        # walk back: find n such that b[end-n-1] == n and all chars printable
        for n in range(1, 255):
            s = end - n
            if s - 1 < 0:
                break
            if b[s - 1] == n:
                txt = b[s:end]
                if all(32 <= c < 127 for c in txt):
                    yield s - 1, txt.decode("ascii")
                break


def is_translation(t: str) -> bool:
    return bool(re.search(r"[a-z]{2,} [a-z]{2,}", t)) and t.strip()[-1:] in ".?!\"'" and not t.isupper()


def parse(path: str) -> list[Utt]:
    b = open(path, "rb").read()
    events = [(o, "s", t) for o, t in _strings(b)]
    for m in TRACK_RE.finditer(b):
        fid = struct.unpack(">H", m.group(1))[0]
        if fid < 100:
            events.append((m.start(), "t", fid))
    events.sort()
    utts, cur, cur_track, last_end = [], Utt(), None, 0
    track_starts = []
    for i, (o, kind, v) in enumerate(events):
        if kind == "s":
            if is_translation(v) and (cur.glosses or cur.tracks):
                # english closes the utterance; the track region ended here
                cur.english = v
                _fill_times(b, cur, track_starts, o)
                utts.append(cur)
                cur, track_starts = Utt(), []
            elif not track_starts:
                cur.glosses.append(v)
            else:
                # a gloss after the track region starts the next utterance (no translation given)
                _fill_times(b, cur, track_starts, o)
                utts.append(cur)
                cur, track_starts = Utt(glosses=[v]), []
        else:
            track_starts.append((o, v))
    return utts


def _fill_times(b: bytes, u: Utt, track_starts: list[tuple[int, int]], end_off: int) -> None:
    allt = []
    for k, (o, fid) in enumerate(track_starts):
        stop = track_starts[k + 1][0] if k + 1 < len(track_starts) else end_off
        times = [struct.unpack(">H", m.group(1))[0] / 600 for m in TIME_RE.finditer(b, o, stop)]
        allt += times
        if fid in NMM_FIELDS and times:
            u.tracks.setdefault(NMM_FIELDS[fid], []).append((min(times), max(times)))
    if allt:
        u.start_s, u.end_s = min(allt), max(allt)


def parse_xml(path: str) -> list[dict]:
    x = open(path, encoding="utf-8").read()
    out = []
    for u in re.findall(r"<UTTERANCE .*?</UTTERANCE>", x, re.S):
        tr = dict(re.findall(r'<TRACK FID="(\d+)"[^>]*>(.*?)</TRACK>', u, re.S))
        gl = re.findall(r"<A [^>]*>([^<]*)</A>", tr.get("10000", ""))
        eng = re.findall(r"<A[^>]*>([^<]*)</A>", tr.get("20001", ""))
        out.append({"glosses": [g for g in gl], "english": (eng[0] if eng else ""),
                    "nmm": sorted(NMM_FIELDS[int(f)] for f in tr if int(f) in NMM_FIELDS)})
    return out
