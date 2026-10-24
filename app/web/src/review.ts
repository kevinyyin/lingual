// ?mode=review: play each sign's avatar clip next to its ASL Citizen reference video.
// Mark a sign captions-only by setting avatar_ok: false for it in app/vocab/vocab.yaml (target: under 10%).
import { createStage, fetchVocab } from "./avatar/stage";
import { loadAvatar } from "./avatar";

export async function startReview(): Promise<void> {
  document.body.innerHTML = `
    <main id="review">
      <h1>Avatar review</h1>
      <p><label>Sign <select id="gloss"></select></label> <button id="play" type="button">Play</button>
         <button id="next" type="button">Next</button> <span id="info"></span></p>
      <div class="pair"><canvas id="avatar" aria-label="Avatar"></canvas>
        <video id="ref" controls muted playsinline aria-label="Reference video"></video></div>
      <p>Does the avatar clearly match the reference (handshape, location, movement)? If not, set
      <code>avatar_ok: false</code> for that sign in <code>app/vocab/vocab.yaml</code>.</p>
    </main>`;
  const vocab = await fetchVocab();
  const select = document.getElementById("gloss") as HTMLSelectElement;
  select.append(...vocab.map((v) => new Option(`${v.gloss}${v.avatar_ok ? "" : " (captions-only)"}`, v.gloss)),
    ...[..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"].map((c) => new Option(`Letter ${c} (fingerspelling)`, `#${c}`)));
  const avatar = await loadAvatar(document.getElementById("avatar") as HTMLCanvasElement);
  if (new URLSearchParams(location.search).get("debug")) Object.assign(window, { asl: { avatar } }); // evaluation scripts
  const stage = await createStage(avatar);
  const ref = document.getElementById("ref") as HTMLVideoElement;
  const play = async () => {
    const g = select.value;
    const letter = g.startsWith("#") ? g.slice(1) : null; // letters have no reference video (Kaggle data is landmarks)
    if (letter) ref.removeAttribute("src");
    else ref.src = `/reference/${encodeURIComponent(g)}`;
    ref.play().catch(() => {});
    const path = letter ? `/letters/${letter}.json` : `/signs/${encodeURIComponent(g)}.json`;
    const ok = (await fetch(path)).headers.get("content-type")?.includes("json");
    document.getElementById("info")!.textContent = ok ? "" : "no clip built yet";
    const token = letter ? { kind: "spell" as const, word: letter } : { kind: "sign" as const, gloss: g };
    await stage.play({ type: "sign_plan", tokens: [token], grammar: "statement", tone: "neutral", english: letter ?? g }, () => {});
  };
  document.getElementById("play")!.onclick = play;
  document.getElementById("next")!.onclick = () => { select.selectedIndex = (select.selectedIndex + 1) % vocab.length; play(); };
}
