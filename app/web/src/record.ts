// Recorder: ?mode=record&signer=NAME [&grid=1 | &letters=1 | &names=1]. Saves landmarks only, never video.
import { fetchVocab } from "./avatar/stage";
import { startCamera } from "./camera";
import { connect, type Frame } from "./ws";

interface Item { gloss: string; label: string; reference?: string }

const TAKES_SIGN = 5;
const TAKES_GRID = 3;
const TAKES_LETTER = 5;
const QTYPES = { yn: "Yes/no question: “Do you want coffee?” (YOU WANT COFFEE, brows up)", wh: "Wh-question: “What do you want?” (YOU WANT WHAT, brows down)", statement: "Statement: “I want coffee.” (I WANT COFFEE)" };
const EMOTIONS = ["happy", "sad", "angry", "neutral"];
const NAMES = ["ANA", "SAM", "PRIYA", "LEO", "MAYA", "JOSE", "KIM", "OMAR", "ZOE", "BEN"]; // spelling eval (<= 8 letters)
const TAKES_NAME = 2;

async function buildQueue(params: URLSearchParams): Promise<Item[]> {
  const out: Item[] = [];
  if (params.get("grid")) {
    for (const [q, label] of Object.entries(QTYPES))
      for (const e of EMOTIONS) for (let i = 0; i < TAKES_GRID; i++) out.push({ gloss: `_grid/${q}_${e}`, label: `${label} — feeling ${e.toUpperCase()}` });
  } else if (params.get("names")) {
    for (const n of NAMES) for (let i = 0; i < TAKES_NAME; i++) out.push({ gloss: `_name/${n}`, label: `Fingerspell the name ${n}` });
  } else if (params.get("letters")) {
    for (const c of "ABCDEFGHIJKLMNOPQRSTUVWXYZ") for (let i = 0; i < TAKES_LETTER; i++) out.push({ gloss: `_letter/${c}`, label: `Fingerspell the letter ${c}` });
  } else {
    const vocab = await fetchVocab();
    for (const s of vocab) for (let i = 0; i < TAKES_SIGN; i++) out.push({ gloss: s.gloss, label: `${s.gloss} — “${s.english}”`, reference: `/reference/${encodeURIComponent(s.gloss)}` });
  }
  return out;
}

export async function startRecorder(params: URLSearchParams): Promise<void> {
  const signer = params.get("signer") ?? "";
  document.body.innerHTML = `
    <main id="recorder">
      <h1>Recorder — <span id="who"></span></h1>
      <p id="progress"></p>
      <p id="prompt"></p>
      <video id="ref" controls loop muted playsinline aria-label="Reference video"></video>
      <video id="cam" muted playsinline aria-label="Your camera"></video>
      <p><strong>Space</strong> starts a take, <strong>Space</strong> again saves it. <strong>S</strong> skips this item.</p>
      <p id="status" aria-live="polite"></p>
    </main>`;
  const $ = (id: string) => document.getElementById(id)!;
  $("who").textContent = signer || "(add &signer=NAME to the URL)";
  const queue = await buildQueue(params);
  let i = 0;
  let take: Frame[] | null = null;
  const sock = connect((m) => {
    if (m.type === "recorded") $("status").textContent = m.ok ? `Saved ${m.detail}` : `Not saved: ${m.detail}`;
  });
  const show = () => {
    const it = queue[i];
    $("progress").textContent = `${Math.min(i + 1, queue.length)} / ${queue.length}`;
    $("prompt").textContent = it ? it.label : "All done — thank you!";
    const ref = $("ref") as HTMLVideoElement;
    if (it?.reference && ref.getAttribute("src") !== it.reference) { ref.src = it.reference; ref.play().catch(() => {}); }
    ref.hidden = !it?.reference;
  };
  document.addEventListener("keydown", (e) => {
    if (e.code === "KeyS" && !take) { i++; show(); return; }
    if (e.code !== "Space" || !queue[i]) return;
    e.preventDefault();
    if (!take) { take = []; $("status").textContent = "Recording…"; return; }
    sock.send({ type: "record", signer, gloss: queue[i].gloss, frames: take });
    take = null;
    i++;
    show();
  });
  show();
  await startCamera($("cam") as HTMLVideoElement, (f) => take?.push(f));
}
