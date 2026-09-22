"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// Dynamically imported to avoid SSR — MediaPipe requires browser APIs
type HandLandmarker = import("@mediapipe/tasks-vision").HandLandmarker;
type NormalizedLandmark = import("@mediapipe/tasks-vision").NormalizedLandmark;

// Pin to installed package version for WASM compatibility
const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const INFER_INTERVAL_MS = 3000;
const PROMPT = "ohwx glove";

// ─── Crossfade display state ─────────────────────────────────────────────────
// Two image slots (A/B) — we write to the inactive slot, then flip `active`.
// CSS opacity transitions handle the crossfade.
type DisplayState = {
  frameA: string;
  frameB: string;
  active: "A" | "B";
};

// ─── Hand crop utility ────────────────────────────────────────────────────────
// Extracts a square region around the detected hand landmarks,
// draws it into a 512×512 canvas, and returns a base64 JPEG string.
function cropHand(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  landmarks: NormalizedLandmark[]
): string {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const xs = landmarks.map((l) => l.x);
  const ys = landmarks.map((l) => l.y);

  const pad = 0.15;
  const minX = Math.max(0, Math.min(...xs) - pad);
  const minY = Math.max(0, Math.min(...ys) - pad);
  const maxX = Math.min(1, Math.max(...xs) + pad);
  const maxY = Math.min(1, Math.max(...ys) + pad);

  // Square crop centered on the hand bounding box
  const bw = (maxX - minX) * vw;
  const bh = (maxY - minY) * vh;
  const size = Math.max(bw, bh);
  const cx = ((minX + maxX) / 2) * vw;
  const cy = ((minY + maxY) / 2) * vh;

  const sx = Math.max(0, cx - size / 2);
  const sy = Math.max(0, cy - size / 2);
  const sw = Math.min(size, vw - sx);
  const sh = Math.min(size, vh - sy);

  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 512, 512);

  return canvas.toDataURL("image/jpeg", 0.85).split(",")[1];
}

// ─── Component ────────────────────────────────────────────────────────────────
export function GloveExperience() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cropCanvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const inferringRef = useRef(false);
  const lastInferRef = useRef(0);
  const currentLandmarksRef = useRef<NormalizedLandmark[] | null>(null);

  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading"
  );
  const [errorMsg, setErrorMsg] = useState("");
  const [handDetected, setHandDetected] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [display, setDisplay] = useState<DisplayState>({
    frameA: "",
    frameB: "",
    active: "A",
  });

  const runInference = useCallback(async () => {
    if (
      !videoRef.current ||
      !cropCanvasRef.current ||
      !currentLandmarksRef.current ||
      inferringRef.current
    )
      return;

    inferringRef.current = true;
    lastInferRef.current = Date.now();
    setInferring(true);

    try {
      const imageB64 = cropHand(
        videoRef.current,
        cropCanvasRef.current,
        currentLandmarksRef.current
      );

      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageB64, prompt: PROMPT }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { image } = await res.json();
      const dataUrl = `data:image/jpeg;base64,${image}`;

      // Write to the inactive slot, then flip active
      setDisplay((prev) => {
        const loadInto = prev.active === "A" ? "B" : "A";
        return { ...prev, [`frame${loadInto}`]: dataUrl, active: loadInto };
      });
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      inferringRef.current = false;
      setInferring(false);
    }
  }, []);

  useEffect(() => {
    let stream: MediaStream | null = null;

    async function setup() {
      try {
        // Dynamic import keeps MediaPipe out of the SSR bundle
        const { HandLandmarker, FilesetResolver } = await import(
          "@mediapipe/tasks-vision"
        );

        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_PATH,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });
        landmarkerRef.current = landmarker;

        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: "user" },
        });

        if (!videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        setStatus("ready");
        startDetectionLoop();
      } catch (err) {
        console.error("Setup error:", err);
        setErrorMsg(err instanceof Error ? err.message : "Setup failed");
        setStatus("error");
      }
    }

    function startDetectionLoop() {
      function detect() {
        const video = videoRef.current;
        const landmarker = landmarkerRef.current;

        if (!video || !landmarker || video.readyState < 2) {
          rafRef.current = requestAnimationFrame(detect);
          return;
        }

        const results = landmarker.detectForVideo(video, performance.now());
        const hasHand = results.landmarks.length > 0;
        setHandDetected(hasHand);

        if (hasHand) {
          currentLandmarksRef.current = results.landmarks[0];
          const now = Date.now();
          if (
            !inferringRef.current &&
            now - lastInferRef.current >= INFER_INTERVAL_MS
          ) {
            runInference();
          }
        } else {
          currentLandmarksRef.current = null;
        }

        rafRef.current = requestAnimationFrame(detect);
      }

      rafRef.current = requestAnimationFrame(detect);
    }

    setup();

    return () => {
      cancelAnimationFrame(rafRef.current);
      stream?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    };
  }, [runInference]);

  if (status === "error") {
    return (
      <div className="flex h-screen items-center justify-center">
        <p className="text-red-400 text-sm font-mono">
          {errorMsg || "something went wrong"}
        </p>
      </div>
    );
  }

  const hasAnyFrame = Boolean(display.frameA || display.frameB);

  return (
    <div className="relative h-screen w-screen bg-black overflow-hidden">
      {/* ── Generated glove — full screen, crossfading ── */}
      <div className="absolute inset-0">
        {(["A", "B"] as const).map((slot) => (
          <img
            key={slot}
            src={display[`frame${slot}`]}
            alt=""
            draggable={false}
            className="absolute inset-0 h-full w-full object-contain select-none"
            style={{
              opacity: display.active === slot ? 1 : 0,
              transition: "opacity 700ms ease-in-out",
            }}
          />
        ))}

        {/* Placeholder before any frames arrive */}
        {!hasAnyFrame && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-zinc-700 text-sm tracking-widest uppercase">
              {status === "loading"
                ? "loading…"
                : handDetected
                ? "generating…"
                : "show your hand"}
            </p>
          </div>
        )}
      </div>

      {/* ── Webcam PiP — bottom right ── */}
      <div className="absolute bottom-4 right-4 w-44 aspect-video rounded overflow-hidden border border-zinc-900 opacity-50 hover:opacity-80 transition-opacity">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }} // mirror for selfie view
          playsInline
          muted
        />
      </div>

      {/* ── Status — bottom left ── */}
      <div className="absolute bottom-4 left-4 flex items-center gap-2 text-xs text-zinc-600 font-mono">
        <span
          className="w-1.5 h-1.5 rounded-full transition-colors duration-300"
          style={{ background: handDetected ? "#22c55e" : "#3f3f46" }}
        />
        {handDetected ? "hand detected" : "no hand"}
        {inferring && <span className="text-zinc-700">· generating</span>}
      </div>

      {/* Off-screen canvas used for cropping frames before sending to inference */}
      <canvas ref={cropCanvasRef} width={512} height={512} className="hidden" />
    </div>
  );
}
