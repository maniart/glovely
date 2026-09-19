"""
Glovely — Modal inference server
StreamDiffusion img2img pipeline with glove LoRA.

Deploy:   modal deploy modal_inference.py
Dev:      modal serve modal_inference.py

Upload weights first:
  modal run modal_inference.py::upload_weights
"""

import io
import base64
import modal

# ---------------------------------------------------------------------------
# Image — CUDA + StreamDiffusion + dependencies
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "xformers==0.0.23.post1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "diffusers>=0.28.0",
        "transformers>=4.36.0",
        "accelerate>=0.26.0",
        "safetensors",
        "Pillow",
        "fastapi[standard]",
        "python-multipart",
        "websockets",
    )
    .run_commands(
        # StreamDiffusion with TensorRT support
        "pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]",
        "python -m streamdiffusion.tools.install-tensorrt",
    )
)

app = modal.App("glovely", image=image)

# Volume stores LoRA weights so they persist across container restarts
# and don't need re-downloading on every cold start.
volume = modal.Volume.from_name("glovely-weights", create_if_missing=True)
VOLUME_PATH = "/weights"
LORA_FILENAME = "pytorch_lora_weights.safetensors"
BASE_MODEL = "KiwiXR/stable-diffusion-v1-5"  # community mirror; same weights


# ---------------------------------------------------------------------------
# Pipeline class — kept warm between requests
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    keep_warm=1,           # keeps one container alive to avoid cold starts
    container_idle_timeout=300,  # scale to zero after 5min of no requests
)
class GlovePipeline:

    @modal.enter()
    def load(self):
        import torch
        from streamdiffusion import StreamDiffusion
        from streamdiffusion.image_utils import postprocess_image
        from diffusers import AutoPipelineForImage2Image

        lora_path = f"{VOLUME_PATH}/{LORA_FILENAME}"

        print(f"Loading pipeline from {BASE_MODEL}...")
        pipe = AutoPipelineForImage2Image.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        print(f"Loading LoRA weights from {lora_path}...")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=1.0)

        print("Wrapping in StreamDiffusion...")
        self.stream = StreamDiffusion(
            pipe,
            t_index_list=[32, 45],   # denoising timestep indices — controls strength
            torch_dtype=torch.float16,
            cfg_type="none",          # faster, no classifier-free guidance overhead
        )
        self.stream.load_lcm_lora()   # LCM LoRA for fast few-step inference
        self.stream.fuse_lora()
        self.stream.enable_similar_image_filter(
            similar_image_strength=0.98,
            similar_distance=10,
        )
        self.stream = self.stream.to(device="cuda", dtype=torch.float16)

        # Warmup — first few calls are slow due to CUDA compilation
        from PIL import Image
        dummy = Image.new("RGB", (512, 512), color=(128, 128, 128))
        self.stream.prepare(
            prompt="ohwx glove",
            negative_prompt="blurry, deformed, low quality",
            num_inference_steps=50,
            guidance_scale=1.0,
        )
        for _ in range(3):
            self.stream(dummy)

        self.postprocess = postprocess_image
        print("Pipeline ready.")

    @modal.method()
    def generate(self, image_b64: str, prompt: str = "ohwx glove") -> str:
        """
        Takes a base64-encoded JPEG/PNG (cropped hand frame),
        returns a base64-encoded PNG (generated glove frame).
        """
        import torch
        from PIL import Image

        # Decode input
        image_bytes = base64.b64decode(image_b64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))

        # Run StreamDiffusion
        output = self.stream(input_image)
        output_image = self.postprocess(output, output_type="pil")[0]

        # Encode output
        buf = io.BytesIO()
        output_image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# HTTP endpoint — called from the Next.js backend
# ---------------------------------------------------------------------------

@app.function()
@modal.web_endpoint(method="POST")
def infer(body: dict) -> dict:
    """
    POST { "image": "<base64>", "prompt": "ohwx glove" }
    → { "image": "<base64>" }
    """
    result = GlovePipeline().generate.remote(
        image_b64=body["image"],
        prompt=body.get("prompt", "ohwx glove"),
    )
    return {"image": result}


# ---------------------------------------------------------------------------
# WebSocket endpoint — for lower-latency streaming (optional, use instead of HTTP)
# ---------------------------------------------------------------------------

@app.function()
@modal.asgi_app()
def ws_app():
    from fastapi import FastAPI, WebSocket
    import json

    fastapi_app = FastAPI()
    pipeline = GlovePipeline()

    @fastapi_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("WebSocket client connected")
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                result_b64 = await pipeline.generate.remote.aio(
                    image_b64=msg["image"],
                    prompt=msg.get("prompt", "ohwx glove"),
                )
                await websocket.send_text(json.dumps({"image": result_b64}))
        except Exception as e:
            print(f"WebSocket closed: {e}")

    return fastapi_app


# ---------------------------------------------------------------------------
# Local entrypoint — upload LoRA weights to the Modal volume
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def upload_weights():
    """
    Run locally to push your trained LoRA to the Modal volume:
      modal run modal_inference.py::upload_weights
    """
    import os

    local_path = "./glove-lora/pytorch_lora_weights.safetensors"
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Not found: {local_path}")

    print(f"Uploading {local_path} → Modal volume:{VOLUME_PATH}/{LORA_FILENAME}")
    with open(local_path, "rb") as f:
        data = f.read()

    volume.commit()  # ensure volume exists

    # Write via a remote function so Modal handles the volume mount
    _upload_file.remote(data, LORA_FILENAME)
    print("Done. Weights are available in the Modal volume.")


@app.function(volumes={VOLUME_PATH: volume})
def _upload_file(data: bytes, filename: str):
    path = f"{VOLUME_PATH}/{filename}"
    with open(path, "wb") as f:
        f.write(data)
    volume.commit()
    print(f"Wrote {len(data) / 1e6:.1f}MB → {path}")
