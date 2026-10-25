import "./style.css";
import { createStage } from "./avatar/stage";
import { loadAvatar } from "./avatar";
import { startCamera } from "./camera";
import { rememberPhone } from "./phone";
import { PositionWatch } from "./position";
import { bindControls, currentMode, currentState, highlight, renderMsg, setCameraOff, showHint } from "./ui";
import { connect } from "./ws";
import { savedBuddy } from "./buddies";
import { showMini } from "./avatar/mini";

const MAX_BUFFERED = 256 * 1024; // drop frames rather than queue stale ones
const params = new URLSearchParams(location.search);
rememberPhone();

if (params.get("mode") === "record") {
  import("./record").then((m) => m.startRecorder(params));
} else if (params.get("mode") === "bench") {
  import("./bench").then((m) => m.startBench());
} else if (params.get("mode") === "mirror") {
  import("./mirror").then((m) => m.startMirror());
} else if (params.get("mode") === "camperf") {
  import("./camperf").then((m) => m.startCamPerf());
} else if (params.get("mode") === "review") {
  import("./review").then((m) => m.startReview());
} else {
  converse();
}

function converse() {
  const avatar = loadAvatar(document.getElementById("avatar") as HTMLCanvasElement);
  const stage = avatar.then(createStage);
  let playing = false; // camera detection pauses while the avatar signs (see startCamera)
  if (params.get("dev")) Object.assign(window, { asl: { avatar, stage } }); // console access for avatar debugging
  const sock = connect(async (m) => {
    if (m.type === "you_said" && m.read?.length) { // the buddy signs back what the camera read
      showMini(document.querySelector<HTMLElement>(".said-card")!, m.read, `I saw: ${m.read.map((t) => t.gloss ?? t.word).join(" ")}`);
    }
    if (m.type !== "sign_plan") return renderMsg(m);
    try {
      renderMsg(m); // inside the try: a render error must not leave the session stuck in "avatar"
      playing = true;
      await (await stage).play(m, (gloss) => highlight(gloss));
    } finally {
      playing = false;
      sock.send({ type: "avatar_done" }); // never leave the session waiting on the avatar
    }
  }, (e) => { // a dead page that looks alive is worse than a clear message
    const badge = document.getElementById("state")!;
    badge.textContent = e.code === 4001 ? "Open elsewhere" : "Connection lost";
    badge.dataset.state = "error";
    const reclaim = document.getElementById("reclaim")!; // reloading takes the session back (the server keeps one page)
    reclaim.textContent = e.code === 4001 ? "Use it here" : "Reconnect";
    reclaim.hidden = false;
  }, { mic: true });
  sock.send({ type: "buddy", buddy: savedBuddy().id }); // the signer's sentences are said in the buddy's voice
  if (params.get("camera") === "0") { // speech -> sign only: no signer turns from stray hand movement
    document.getElementById("mode")!.hidden = true; // nothing to switch to
    bindControls(sock);
    setCameraOff();
    document.querySelector(".cam-card figcaption")!.textContent = "Camera off · speech to sign only";
    document.getElementById("cam")!.hidden = true;
    return;
  }
  // opened the first time signing mode is on (voice mode never asks for the camera); detection runs only in signing
  // mode and not while the avatar signs
  let camera: Promise<void> | null = null;
  const watch = new PositionWatch(); // directs the signer into position; calibrates again when they move (position.ts)
  const openCamera = () => (camera ??= startCamera(document.getElementById("cam") as HTMLVideoElement, (f) => {
    const { hint, send, recalibrate } = watch.check(f, currentState(), performance.now());
    showHint(hint);
    if (recalibrate) sock.send({ type: "calibrate" });
    if (send && sock.buffered() < MAX_BUFFERED) sock.send({ type: "frame", ...f });
  }, () => playing || currentMode() !== "signing").catch((e) => {
    camera = null; // switching to signing again retries
    const badge = document.getElementById("state")!;
    badge.textContent = e.name === "NotAllowedError" ? "Camera blocked: allow it in Settings" : "No camera: use Voice";
    badge.dataset.state = "error";
  }));
  bindControls(sock, (m) => { if (m === "signing") void openCamera(); });
}
