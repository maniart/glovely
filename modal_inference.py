"""
Glovely — Modal inference server
Stable Diffusion 1.5 img2img + glove LoRA, accelerated with LCM-LoRA (4-step).

First-time setup:
  modal run modal_inference.py::upload_weights   # upload glove LoRA
  modal run modal_inference.py::download_models  # cache SD1.5 + LCM-LoRA to volume

Deploy:
  modal deploy modal_inference.py

Dev:
  modal serve modal_inference.py
"""

import io
import base64
import modal

# ---------------------------------------------------------------------------
# Image — lean: just packages, no model downloads at build time
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "diffusers>=0.31.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors",
        "Pillow",
        "fastapi[standard]",
        "websockets",
    )
)

app = modal.App("glovely", image=image)

# ---------------------------------------------------------------------------
# Volume — stores both LoRA weights and cached base models
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("glovely-weights", create_if_missing=True)
VOLUME_PATH = "/weights"
LORA_FILENAME = "pytorch_lora_weights.safetensors"
MODEL_CACHE = f"{VOLUME_PATH}/model-cache"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    min_containers=1,
    scaledown_window=300,
)
class GlovePipeline:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler

        lora_path = f"{VOLUME_PATH}/{LORA_FILENAME}"

        print(f"Loading {BASE_MODEL} from volume cache...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        print("Swapping in LCM scheduler...")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        print("Loading LCM-LoRA from volume cache...")
        self.pipe.load_lora_weights(
            LCM_LORA_ID,
            cache_dir=MODEL_CACHE,
            adapter_name="lcm",
        )

        print(f"Loading glove LoRA from {lora_path}...")
        self.pipe.load_lora_weights(lora_path, adapter_name="glove")

        self.pipe.set_adapters(["lcm", "glove"], adapter_weights=[1.0, 0.8])
        self.pipe.enable_attention_slicing()

        # Warmup
        from PIL import Image as PILImage
        self.pipe(
            prompt="ohwx glove",
            image=PILImage.new("RGB", (512, 512)),
            num_inference_steps=4,
            guidance_scale=1.0,
            strength=0.6,
        )
        print("Pipeline ready.")

    @modal.method()
    def generate(self, image_b64: str, prompt: str = "ohwx glove") -> str:
        import torch
        from PIL import Image as PILImage

        image_bytes = base64.b64decode(image_b64)
        input_image = (
            PILImage.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))
        )

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt="blurry, deformed, low quality, extra fingers",
                image=input_image,
                num_inference_steps=4,
                guidance_scale=1.0,
                strength=0.6,
            )

        buf = io.BytesIO()
        result.images[0].save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@app.function()
@modal.fastapi_endpoint(method="POST")
def infer(body: dict) -> dict:
    """POST { "image": "<base64>", "prompt": "ohwx glove" } → { "image": "<base64>" }"""
    result = GlovePipeline().generate.remote(
        image_b64=body["image"],
        prompt=body.get("prompt", "ohwx glove"),
    )
    return {"image": result}


# ---------------------------------------------------------------------------
# WebSocket endpoint
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
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                result_b64 = await pipeline.generate.remote.aio(
                    image_b64=msg["image"],
                    prompt=msg.get("prompt", "ohwx glove"),
                )
                await websocket.send_text(json.dumps({"image": result_b64}))
        except Exception as e:
            print(f"WebSocket closed: {e}")

    return fastapi_app


# ---------------------------------------------------------------------------
# One-time setup: download base models to the volume
# Run once after first deploy: modal run modal_inference.py::download_models
# ---------------------------------------------------------------------------

@app.function(volumes={VOLUME_PATH: volume}, timeout=1800)
def _download_models():
    from diffusers import StableDiffusionImg2ImgPipeline
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_CACHE, exist_ok=True)

    print(f"Downloading {BASE_MODEL}...")
    StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL, cache_dir=MODEL_CACHE
    )

    print(f"Downloading {LCM_LORA_ID}...")
    snapshot_download(LCM_LORA_ID, cache_dir=MODEL_CACHE)

    volume.commit()
    print("Models cached to volume.")


@app.local_entrypoint()
def download_models():
    """modal run modal_inference.py::download_models"""
    print("Downloading base models to Modal volume (runs once)...")
    _download_models.remote()
    print("Done.")


# ---------------------------------------------------------------------------
# Upload glove LoRA weights to the volume
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def upload_weights():
    """modal run modal_inference.py::upload_weights"""
    import os

    local_path = "./glove-lora/pytorch_lora_weights.safetensors"
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Not found: {local_path}")

    print(f"Uploading {local_path} → {VOLUME_PATH}/{LORA_FILENAME}")
    with open(local_path, "rb") as f:
        _upload_file.remote(f.read(), LORA_FILENAME)
    print("Done.")


@app.function(volumes={VOLUME_PATH: volume})
def _upload_file(data: bytes, filename: str):
    path = f"{VOLUME_PATH}/{filename}"
    with open(path, "wb") as f:
        f.write(data)
    volume.commit()
    print(f"Wrote {len(data) / 1e6:.1f}MB → {path}")
