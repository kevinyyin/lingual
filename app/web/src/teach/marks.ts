// Drawn marks, in go (the motion color), and the rose strike for a wrong answer.
import { svg } from "./dom";

/** Where the course line has got to: a V pointing down the line, its tip on the stop. */
export const ARROWHEAD = '<svg viewBox="0 0 24 24"><path d="M5 8l7 8 7-8" fill="none" stroke="#F2592A" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

const PATHS = {
  loop: "M10 64C0 34 28 6 52 6C80 6 99 22 97 50C95 80 70 96 46 95C20 94 2 78 5 52C7 34 20 20 36 14",
  strike: "M5 62L28 40 24 62 50 38 46 62 72 40 68 60 95 42",
};

/** A teacher's mark over an answer tile: a loop around the right one, a strike through a wrong choice. It stretches to
 *  the tile; the CSS keeps its stroke one width (vector-effect). */
export function markSvg(kind: "loop" | "strike"): SVGSVGElement {
  return svg(`<svg viewBox="0 0 100 100" preserveAspectRatio="none"><path d="${PATHS[kind]}"/></svg>`, `mark ${kind}`);
}
