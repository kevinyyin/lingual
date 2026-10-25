// The signing avatar (rocketbox.ts): the toon character by default, ?model= for the others.
import type { Avatar, AvatarOptions } from "./rocketbox";

export async function loadAvatar(canvas: HTMLCanvasElement, opts?: AvatarOptions): Promise<Avatar> {
  return (await import("./rocketbox")).loadRocketbox(canvas, opts);
}
