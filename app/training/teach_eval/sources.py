"""Allowed teaching-video sources -> local clips (only sources.json "allow" entries; polite: one request per second,
an identifying User-Agent, robots.txt respected, nothing re-uploaded).

ASL Signbank (aslsignbank.com, CC BY-NC-SA 4.0, attribution "(ASL Signbank, 2026)"): citation-form videos, one per
entry. Each sign is searched by its English word; entries whose ID gloss matches ours (THANK-YOU = THANKYOU, variants
COFFEE-2 etc.) are kept, up to 2 per sign.

  uv run python -m training.teach_eval.sources aslsignbank
"""
import argparse
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser

from server.classifier import SignPredictor
from server.vocab import load_vocab
from training.teach_eval.clips import DATA, Clip, allowed_sources, load_clips, save_clips

UA = "lingual-sign-eval/0.1 (student hackathon research, non-commercial; github.com/kevinyyin/lingual)"
SB = "https://aslsignbank.com"
_last = [0.0]


def _get(url: str) -> bytes:
    wait = 1.0 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return r.read()


def _norm(g: str) -> str:
    return re.sub(r"[^A-Z]", "", re.sub(r"[-_:]?\d+$", "", g.upper()))


def _save(sign, gid, m, out_dir, match: str) -> Clip:
    dst = out_dir / f"{sign.gloss}_{gid}.mp4"
    if not dst.exists():
        dst.write_bytes(_get(f"{SB}/dictionary/{m.group(0)}"))
    return Clip(id=f"aslsignbank:{gid}", source="aslsignbank", gloss=sign.gloss, path=str(dst), signer="aslsignbank",
                url=f"{SB}/dictionary/gloss/{gid}.html", licence="CC BY-NC-SA 4.0",
                attribution=f"ASL Signbank (2026), entry {m.group(1)}", extra={"idgloss": m.group(1), "match": match})


def aslsignbank(per_sign: int = 2) -> list[Clip]:
    src = allowed_sources().get("aslsignbank") or next((s for s in allowed_sources().values() if "signbank" in s["id"].lower()), None)
    if not src:
        raise SystemExit("ASL Signbank is not on the allow list in sources.json")
    rp = urllib.robotparser.RobotFileParser(SB + "/robots.txt")
    try:
        rp.read()
    except Exception:
        rp = None  # no robots.txt (404): nothing is disallowed
    labels = set(SignPredictor().labels)
    out_dir = DATA / "videos" / "aslsignbank"
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {c.gloss for c in load_clips() if c.source == "aslsignbank"}
    clips = []
    for sign in load_vocab(extended=None).active:
        if sign.gloss not in labels or sign.gloss in have:
            continue
        words = list(dict.fromkeys([sign.english.split("/")[0].strip(), sign.gloss.lower().replace("-", " ")]))

        def search(term):
            url = f"{SB}/signs/search/?search_type=sign&translation={urllib.parse.quote(term)}"
            if rp and not rp.can_fetch(UA, url):
                return []
            m = re.search(r"var objects_on_this_page = \[([^\]]*)\]", _get(url).decode("utf-8", "replace"))
            return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []

        # exact translation first (^word$: the search is a regex; a plain word matched YESTERDAY for "yes"), then contains
        exact = [i for w in words for i in search(f"^{w}$")]
        ids = list(dict.fromkeys(exact + (search(words[0]) if len(exact) < 2 else [])))
        got, fallback = 0, []
        for gid in ids[:8]:
            page = _get(f"{SB}/dictionary/gloss/{gid}.html").decode("utf-8", "replace")
            m = re.search(r"protected_media/glossvideo/ASL/[^\"' ]+?/([^/\"' ]+)-(\d+)\.mp4", page)
            if not m:
                continue
            if _norm(m.group(1)) != _norm(sign.gloss):
                if gid in exact:  # an exact English translation under another ID gloss: kept if nothing better, flagged
                    fallback.append((gid, m))
                continue
            clips.append(_save(sign, gid, m, out_dir, "idgloss"))
            got += 1
            if got >= per_sign:
                break
        for gid, m in fallback[: per_sign - got]:
            clips.append(_save(sign, gid, m, out_dir, "translation"))
            got += 1
        print(f"  {sign.gloss}: {got}", flush=True)
    return clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["aslsignbank"])
    ap.add_argument("--per-sign", type=int, default=2)
    a = ap.parse_args()
    new = aslsignbank(a.per_sign)
    save_clips(load_clips() + new)
    print(f"{len(new)} ASL Signbank clips, {len({c.gloss for c in new})} signs")


if __name__ == "__main__":
    main()
