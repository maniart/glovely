import { NextRequest } from "next/server";

// Allow up to 5 minutes — Modal cold start can take ~60s on first boot.
export const maxDuration = 300;

// Proxies inference requests to the Modal backend.
// Keeps MODAL_INFERENCE_URL server-side — never exposed to the browser.
export async function POST(req: NextRequest) {
  const inferenceUrl = process.env.MODAL_INFERENCE_URL;
  if (!inferenceUrl) {
    return Response.json(
      { error: "MODAL_INFERENCE_URL is not configured" },
      { status: 500 }
    );
  }

  const body = await req.json();

  const upstream = await fetch(inferenceUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    return Response.json(
      { error: `Upstream error: ${upstream.status}`, detail: text },
      { status: upstream.status }
    );
  }

  const result = await upstream.json();
  return Response.json(result);
}
