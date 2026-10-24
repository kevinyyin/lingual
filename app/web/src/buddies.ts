// The signing buddies, picked on the Profile tab: each a toon avatar (public/avatar/toon/<id>.glb, from
// scripts/build_toon_avatar.py) with its own Cartesia voice (server/voice.py BUDDY_VOICES). Remembered in this browser
// for both pages; Leo until one is picked.
export const BUDDIES = [
  { id: "leo", name: "Leo" },
  { id: "benji", name: "Benji" },
  { id: "zeke", name: "Zeke" },
  { id: "lexi", name: "Lexi" },
  { id: "avery", name: "Avery" },
  { id: "daisy", name: "Daisy" },
] as const;
export type Buddy = (typeof BUDDIES)[number];

const KEY = "lingual.buddy";

export function savedBuddy(): Buddy {
  let id: string | null = null;
  try { id = localStorage.getItem(KEY); } catch { /* storage blocked: Leo */ }
  return BUDDIES.find((b) => b.id === id) ?? BUDDIES[0];
}

export function saveBuddy(id: Buddy["id"]): void {
  try { localStorage.setItem(KEY, id); } catch { /* storage blocked: the pick isn't remembered */ }
}
