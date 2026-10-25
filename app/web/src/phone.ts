// The iOS app (Capacitor shell) opens the page with ?phone=1: this page then brings its own mic and speaker
// instead of the laptop's. Remembered per origin, so a laptop browser on localhost is never a phone; ?phone=0 forgets it.

const KEY = "lingual.phone";

export function rememberPhone(): void {
  try {
    const p = new URLSearchParams(location.search).get("phone");
    if (p === "1") localStorage.setItem(KEY, "1");
    else if (p === "0") localStorage.removeItem(KEY);
  } catch { /* storage blocked: stays a laptop page */ }
}

export function isPhone(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

const MIC_RATE = 16000; // server/stt.py SAMPLE_RATE
const VOICE_RATE = 24000; // server/voice.py SAMPLE_RATE

export function toPcm16(f: Float32Array): Int16Array {
  const out = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = Math.round(Math.max(-1, Math.min(1, f[i])) * 32767);
  return out;
}

/** iOS 16.4+: the default (ambient) session is muted by the ringer switch, so the plushie would go silent.
 * Once the mic is on, stay in play-and-record (dropping back would cut the mic). */
export function setAudioSession(type: "playback" | "play-and-record"): void {
  const s = (navigator as Navigator & { audioSession?: { type: string } }).audioSession;
  if (!s || (s.type === "play-and-record" && type === "playback")) return;
  try { s.type = type; } catch { /* unsupported value: keep the default */ }
}

let speaker: AudioContext | null = null;
let micCtx: AudioContext | null = null;
let next = 0;

/** The page's one sound output, for the voice and the lesson chimes. iOS plays only from an AudioContext a tap started
 *  (unlockAudio): the chimes' own, first made for a camera-graded answer with no tap, stayed suspended and silent. */
export function speakerOut(): AudioContext {
  return (speaker ??= new AudioContext());
}

/** Plays the plushie's voice: 24 kHz PCM16 chunks from the server, each right after the one before. */
export function playPcm(buf: ArrayBuffer): void {
  setAudioSession("playback");
  const out = speakerOut();
  const pcm = new Int16Array(buf);
  if (!pcm.length) return;
  const b = out.createBuffer(1, pcm.length, VOICE_RATE);
  const ch = b.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = out.createBufferSource();
  src.buffer = b;
  src.connect(out.destination);
  next = Math.max(out.currentTime, next);
  src.start(next);
  next += b.duration;
}

/** The phone mic as 16 kHz PCM16 chunks for the server's speech-to-text. */
export async function startMic(send: (b: ArrayBuffer) => void): Promise<void> {
  setAudioSession("play-and-record");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  micCtx = new AudioContext({ sampleRate: MIC_RATE });
  const src = micCtx.createMediaStreamSource(stream);
  // shortcut: ScriptProcessorNode is deprecated but works in Safari/WKWebView; move to an AudioWorklet if it stutters
  const proc = micCtx.createScriptProcessor(2048, 1, 1);
  proc.onaudioprocess = (e) => send(toPcm16(e.inputBuffer.getChannelData(0)).buffer as ArrayBuffer);
  src.connect(proc);
  proc.connect(micCtx.destination);
}

/** iOS keeps audio silent until a touch: resume both contexts on the first one. */
export function unlockAudio(): void {
  void speakerOut().resume();
  void micCtx?.resume();
}
