// The signing avatar (rocketbox.ts): the toon character by default, ?model= for the others.
import type { Avatar } from "./rocketbox";

export async function loadAvatar(canvas: HTMLCanvasElement): Promise<Avatar> {
  return (await import("./rocketbox")).loadRocketbox(canvas);
}
