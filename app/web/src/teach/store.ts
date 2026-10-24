// Saves progress in this browser (no accounts in the MVP). Storage can be missing or blocked: the page still works,
// it just forgets on reload.
import { catchUp, fresh, type Progress } from "./progress";

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

export interface Settings { sound: boolean; camera: boolean }
export function loadSettings(): Settings {
  try { return { sound: true, camera: true, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") }; } catch { return { sound: true, camera: true }; }
}
export function saveSettings(s: Settings): void {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch { /* storage blocked */ }
}
