import { expect, it, vi } from "vitest";

const createTrackers = vi.fn(async () => ({}));
vi.mock("./landmarks", async (orig) => ({ ...(await orig<typeof import("./landmarks")>()), createTrackers }));

it("runs the trackers on the page's thread where workers can't (no Worker here, as in an old browser)", async () => {
  vi.spyOn(console, "warn").mockImplementation(() => {});
  const { loadTrackers } = await import("./camera");
  const { PARTS } = await import("./landmarks");
  const tracker = await loadTrackers();
  expect(createTrackers).toHaveBeenCalledWith(false, PARTS);
  expect(await tracker.detect({} as HTMLVideoElement, 1)).toEqual({ parts: {}, ms: {} });
  expect(await loadTrackers()).toBe(tracker); // created once
});
