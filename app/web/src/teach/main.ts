// Teaching mode (teach.html): a Duolingo-style ASL course on top of the translator's avatar and sign recognizer.
// Translator mode (index.html) is untouched; the two share only the avatar, camera and backend modules.
import "./teach.css";
import { ALL_LESSONS, restrictCourse, signsTaughtThrough } from "./curriculum";
import { h, svg } from "./dom";
import { ICON } from "./icons";
import { renderHome, TAB_IDS, weekDone, type HomeCtx, type Tab } from "./home";
import { buildLesson, practiceSigns, type Exercise } from "./lesson";
import { runLesson } from "./lessonView";
import { dayKey, finishLesson, fresh, type Progress } from "./progress";
import { isPlaying, playTokens, practice } from "./signer";
import { lookupSign } from "./showMe";
import { fetchVocab, type VocabSign } from "../avatar/stage";
import { rememberPhone } from "../phone";
import { loadTrackers } from "../camera";
import { load, loadRun, loadSettings, save, saveRun, saveSettings, type Settings } from "./store";

rememberPhone();
const root = document.getElementById("teach")!;
let progress: Progress = load();
let settings: Settings = loadSettings();
let cameraThisSession = true;
const english = new Map<string, string>();
let vocab: string[] = [];
let signs: VocabSign[] = [];

const setProgress = (p: Progress) => { progress = p; save(p); };
const tabOf = (): Tab => {
  const t = location.hash.replace(/^#\/?/, "");
  return TAB_IDS.includes(t as Tab) ? (t as Tab) : "learn";
};

const ctx = (): HomeCtx => ({
  p: progress,
  settings,
  english: (g) => english.get(g) ?? g.toLowerCase(),
  update: (p) => { setProgress(p); home(); },
  updateSettings: (s) => { settings = s; saveSettings(s); home(); },
  start: (id) => startLesson(id),
  startPractice: (kind) => startPractice(kind),
  words: signs.map((s) => s.english),
  showMe: (word) => {
    const t = lookupSign(word, signs);
    if (!t || isPlaying()) return; // playTokens drops a request while one plays; say nothing about it either
    if (t.kind === "spell" && settings.sound) practice.say("Here's how to spell it.");
    void playTokens([t]);
  },
  reset: () => { setProgress(fresh()); home(); },
});

function home() {
  document.title = "ASL Dragon";
  renderHome(root, tabOf(), ctx());
}

const camera = () => settings.camera && cameraThisSession && !!navigator.mediaDevices?.getUserMedia;

function play(id: string, exercises: Exercise[]) {
  document.onclick = null;
  window.scrollTo(0, 0);
  runLesson(root, exercises, {
    english: ctx().english,
    sound: () => settings.sound,
    say: (t) => { if (settings.sound) practice.say(t); },
    cameraOn: camera,
    disableCamera: () => { cameraThisSession = false; },
    quit: () => { home(); window.scrollTo(0, 0); },
    save: (run) => saveRun(id, run),
    finish: (run) => {
      const now = new Date();
      const { progress: p, earned } = finishLesson(progress, {
        lessonId: id, answers: run.answers, mistakes: run.mistakes, bestCombo: run.bestCombo, ms: performance.now() - run.started,
      }, now);
      setProgress(p);
      return { earned, streak: progress.streak, week: weekDone(progress, now) };
    },
  }, loadRun(id) ?? undefined);
}

function startLesson(id: string) {
  const lesson = ALL_LESSONS.find((l) => l.id === id);
  if (!lesson) return;
  play(id, buildLesson({
    signs: lesson.signs, known: signsTaughtThrough(id), vocab, review: id.endsWith("review"),
    perform: camera(), rng: Math.random,
  }));
}

function startPractice(kind: "weak" | "signing") {
  if (kind === "signing") cameraThisSession = true; // choosing to sign to the camera ends "camera off for now"
  const now = new Date();
  const signs = practiceSigns(progress, dayKey(now), kind === "signing" ? 4 : 6);
  if (!signs.length) return;
  const exercises: Exercise[] = kind === "signing"
    ? signs.map((g) => ({ kind: "perform", gloss: g }))
    : buildLesson({ signs, known: Object.keys(progress.signs), vocab, review: true, perform: false, rng: Math.random });
  play(`practice-${kind}`, exercises);
}

async function boot() {
  try {
    const v = (await fetchVocab()).filter((s) => s.avatar_ok);
    for (const s of v) english.set(s.gloss, s.english);
    vocab = v.map((s) => s.gloss);
    signs = v;
    restrictCourse(new Set(vocab));
  } catch {
    root.replaceChildren(h("div", { class: "error-screen" },
      h("div", { class: "error-mark" }, svg(ICON.mark)), h("h1", null, "Can't reach the backend"),
      h("p", { class: "muted" }, "Start it with ", h("code", null, "uv run asl-server"), " in ", h("code", null, "app/"), ", then reload."),
      h("button", { class: "btn", onclick: () => location.reload() }, "Reload")));
    return;
  }
  window.addEventListener("hashchange", () => { if (!root.querySelector(".lesson, .done-screen")) home(); });
  home();
  // the first camera exercise would otherwise wait for ~30 MB of trackers; start once the home screen is up
  if (camera()) setTimeout(() => void loadTrackers().catch(() => {}), 1500);
}

void boot();
