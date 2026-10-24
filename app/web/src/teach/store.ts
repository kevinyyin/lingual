// Saves progress in this browser (no accounts in the MVP). Storage can be missing or blocked: the page still works,
// it just forgets on reload.
import { catchUp, fresh, type Progress } from "./progress";
import type { SavedRun } from "./lesson";

const KEY = "lingual.teach.v1";
const SETTINGS_KEY = "lingual.teach.settings";

export function load(now = new Date()): Progress {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return catchUp({ ...fresh(), ...JSON.parse(raw) }, now); // fields added later get their defaults
  } catch { /* unreadable: start over */ }
  return fresh();
}

export function save(p: Progress): void {
  try { localStorage.setItem(KEY, JSON.stringify(p)); } catch { /* storage blocked */ }
}

const runKey = (lessonId: string) => `lingual.teach.run.${lessonId}`;
/** A lesson left part way, to resume from; null once it's finished or there is none. */
export function loadRun(lessonId: string): SavedRun | null {
  try {
    const r = JSON.parse(localStorage.getItem(runKey(lessonId)) ?? "null");
    return Array.isArray(r?.queue) && r.queue.length && Array.isArray(r.past) ? r : null; // anything else: start the lesson over
  } catch { return null; }
}
export function saveRun(lessonId: string, r: SavedRun | null): void {
  try {
    if (r) localStorage.setItem(runKey(lessonId), JSON.stringify(r));
    else localStorage.removeItem(runKey(lessonId));
  } catch { /* storage blocked: the lesson starts over next time */ }
}

export interface Settings { sound: boolean; camera: boolean }
export function loadSettings(): Settings {
  try { return { sound: true, camera: true, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") }; } catch { return { sound: true, camera: true }; }
}
export function saveSettings(s: Settings): void {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch { /* storage blocked */ }
}
