import type { CapacitorConfig } from "@capacitor/cli";

// The app bundles only the start screen (shell/); it then opens the laptop's dev server through a Cloudflare quick
// tunnel, so the 900 MB of sign clips stay on the laptop. Navigation to the tunnel must stay inside the app.
const config: CapacitorConfig = {
  appId: "com.lingual.plushie",
  appName: "ASL Dragon",
  webDir: "shell",
  server: { allowNavigation: ["*.trycloudflare.com"] },
};

export default config;
