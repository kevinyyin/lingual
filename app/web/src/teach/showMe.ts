// "Show me a sign": a parent types a word, the avatar signs it, or fingerspells it when the vocabulary has no sign.
import type { SignToken } from "../ws";
import type { VocabSign } from "../avatar/stage";

export function lookupSign(word: string, vocab: VocabSign[]): SignToken | null {
  const w = word.trim().toLowerCase();
  if (!w) return null;
  const hits = vocab.filter((s) => s.english.toLowerCase() === w || s.gloss.toLowerCase() === w);
  const hit = hits.find((s) => s.avatar_ok) ?? hits[0];
  if (hit) return { kind: "sign", gloss: hit.gloss };
  const letters = w.toUpperCase().replace(/[^A-Z]/g, "");
  return letters ? { kind: "spell", word: letters } : null;
}
