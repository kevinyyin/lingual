"""Screen-record the running app (backend :8000 + web :5173) with Playwright, for demos and regression evidence.

  speech_to_sign: typed sentences (dev box) -> captions, sign chips, the avatar signing
  sign_to_speech: real ASL Citizen test-signer videos fed to the page *as its webcam* (Chromium fake capture), so the
                  whole live path runs: MediaPipe in the page -> calibration -> segmenter -> recognizer -> English
  avatar_tour:    review mode, one sign after another

Run: uv run --with playwright python scripts/record_demo.py [--out DIR] [--only speech_to_sign,...]
(first time: uv run --with playwright playwright install chromium). Videos (.webm) land in --out.
"""

import argparse
import csv
import re
import shutil
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:5173"
SENTENCES = ["Hello! What is your name?", "Do you want coffee or tea?", "I'm sorry, we don't have milk today.",
             "Where is the bathroom?", "My friend is deaf.", "Thank you, see you tomorrow!"]
SIGNS_TO_SHOW = ["HELLO", "THANK-YOU", "COFFEE", "WANT", "NAME", "YES", "NO", "HELP", "WATER", "FRIEND", "HAPPY",
                 "SORRY", "PLEASE", "BOOK", "COMPUTER", "DOG"]
# One test signer throughout, like a real user (stitching different signers made every cut look like motion); P35
# has test videos for all of the curated vocabulary. Names are ASL Citizen glosses.
CITIZEN_SIGNER = "P35"
CITIZEN_SIGNS = ["HELLO", "THANKYOU", "COFFEE", "WATER", "HELP", "FRIEND", "HAPPY", "BATHROOM", "SORRY"]


WINDOW = ["--window-size=1456,1000"]  # headed window larger than the 1440x860 viewport, so the page fills the recording


def wait_state(page, word: str, timeout_s: float) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if word.lower() in (page.text_content("#state") or "").lower():
            return True
        time.sleep(0.2)
    return False


def speech_to_sign(pw, out: Path) -> Path:
    browser = pw.chromium.launch(headless=False, args=WINDOW)
    ctx = browser.new_context(viewport={"width": 1440, "height": 860}, record_video_dir=str(out / "tmp"),
                              record_video_size={"width": 1440, "height": 860})
    page = ctx.new_page()
    page.goto(f"{BASE}/?dev=1&camera=0")
    wait_state(page, "ready", 30)
    time.sleep(2)
    for s in SENTENCES:
        page.fill("#dev-text", s)
        page.press("#dev-text", "Enter")
        wait_state(page, "signing", 15)
        wait_state(page, "ready", 30)
        time.sleep(1.5)
    return _finish(ctx, browser, page, out / "speech_to_sign.webm")


def y4m_from_citizen(out: Path, signs: list[str], signer: str | None = CITIZEN_SIGNER) -> tuple[Path, list[str]]:
    """Webcam stand-in: 3 s of the signer at rest (calibration), then each test clip with rest holds between."""
    test = ROOT / "data" / "asl_citizen" / "splits" / "test.csv"
    vids = ROOT / "data" / "asl_citizen" / "videos"
    with open(test, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    picked = []
    for g in signs:
        for r in rows:
            if re.sub(r"\d+$", "", r[2].strip()) == g and (vids / r[1]).exists() and signer in (None, r[0]):
                picked.append((g, vids / r[1]))
                break
    path = out / "tmp" / "webcam.y4m"
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 480, 360  # Chromium reads the whole file before the camera starts; 640x480 (460 MB) took ~30 s

    def frames(p):
        cap = cv2.VideoCapture(str(p))
        fs = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fs.append(cv2.resize(f, (W, H)))
        cap.release()
        return fs

    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{W} H{H} F30:1 Ip A1:1 C420jpeg\n".encode())

        def put(f):
            fh.write(b"FRAME\n" + cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).tobytes())

        for k, (_, p) in enumerate(picked):
            fs = frames(p)
            if not fs:
                continue
            for _ in range(90 if k == 0 else 30):  # calibration on the first signer, then a rest hold
                put(fs[0])
            for f in fs:
                put(f)
            for _ in range(60):  # hands down long enough to end the turn (segmenter.REST_MS, with margin)
                put(fs[-1])
    return path, [g for g, _ in picked]


def sign_to_speech(pw, out: Path) -> Path:
    cam, shown = y4m_from_citizen(out, CITIZEN_SIGNS)
    seconds = (cam.stat().st_size / (480 * 360 * 1.5 + 6)) / 30
    browser = pw.chromium.launch(headless=False, args=WINDOW + ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                                        f"--use-file-for-fake-video-capture={cam}"])
    ctx = browser.new_context(viewport={"width": 1440, "height": 860}, permissions=["camera"],
                              record_video_dir=str(out / "tmp"), record_video_size={"width": 1440, "height": 860})
    page = ctx.new_page()
    page.goto(f"{BASE}/?dev=1")
    wait_state(page, "ready", 120)  # the camera starts once Chromium has read the file, then calibration runs
    # The fake camera plays the clips once (then loops); leave time for the last turn. When the picker asks, answer as
    # an honest signer would: their real sign if it is offered, else "none of these" (Esc).
    want = [{"THANKYOU": "THANK-YOU"}.get(g, g) for g in shown]
    end = time.time() + seconds + 8
    while time.time() < end:
        if page.is_visible("#picker"):
            offered = [s.strip("()") for s in page.locator("#picker .gloss").all_text_contents()]
            hit = next((i for i, g in enumerate(offered) if g in want), None)
            page.locator("#picker button").nth(hit).click() if hit is not None else page.click("#picker button.none")
            if hit is not None:
                want.remove(offered[hit])
            time.sleep(1.0)
        time.sleep(0.3)
    (out / "sign_to_speech_signs.txt").write_text(f"Signed (ASL Citizen test signer {CITIZEN_SIGNER}, in order): " + " ".join(shown) + "\n",
                                                  encoding="utf-8")
    return _finish(ctx, browser, page, out / "sign_to_speech.webm")


def avatar_tour(pw, out: Path) -> Path:
    browser = pw.chromium.launch(headless=False, args=WINDOW)
    ctx = browser.new_context(viewport={"width": 1440, "height": 860}, record_video_dir=str(out / "tmp"),
                              record_video_size={"width": 1440, "height": 860})
    page = ctx.new_page()
    page.goto(f"{BASE}/?mode=review")
    page.wait_for_selector("#gloss option", state="attached", timeout=30000)
    time.sleep(2)
    have = set(page.eval_on_selector_all("#gloss option", "os => os.map(o => o.value)"))
    for g in [g for g in SIGNS_TO_SHOW if g in have]:
        page.select_option("#gloss", g)
        page.click("#play")
        time.sleep(3.0)
    return _finish(ctx, browser, page, out / "avatar_tour.webm")


def _finish(ctx, browser, page, dst: Path) -> Path:
    video = page.video
    ctx.close()
    browser.close()
    shutil.move(video.path(), dst)
    return dst


def main() -> None:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo")
    ap.add_argument("--only", default="speech_to_sign,sign_to_speech,avatar_tour")
    a = ap.parse_args()
    a.out = a.out.resolve()  # Chromium opens the fake-camera file from its own working directory
    a.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        for name in a.only.split(","):
            print("recorded", {"speech_to_sign": speech_to_sign, "sign_to_speech": sign_to_speech,
                               "avatar_tour": avatar_tour}[name](pw, a.out), flush=True)
    shutil.rmtree(a.out / "tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
