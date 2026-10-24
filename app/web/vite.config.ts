import { defineConfig } from "vite";

const api = process.env.API ?? "http://localhost:8000"; // API=http://localhost:8010 for a second copy (another worktree)
export default defineConfig({
  // Two modes, two pages: translator (index.html, the original two-way app) and teaching (teach.html, the course).
  // public/ (900+ MB of sign clips) is served in place by the backend (server/main.py mount_web), never copied
  build: { copyPublicDir: false, rollupOptions: { input: { translator: "index.html", teach: "teach.html" } } },
  server: {
    allowedHosts: [".trycloudflare.com"], // the phone reaches this dev server through a Cloudflare quick tunnel
    hmr: process.env.TUNNEL ? false : undefined, // HMR can't reach the phone through the tunnel; its reconnect reloads the page
    proxy: {
      "/ws": { target: api.replace(/^http/, "ws"), ws: true },
      "/vocab": api,
      "/reference": api,
      "/dev": api,
    },
  },
  test: { environment: "jsdom" },
} as any);
