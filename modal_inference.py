"""
Glovely — Modal inference server
Stable Diffusion 1.5 img2img + glove LoRA, accelerated with LCM-LoRA (4-step).

Deploy:   modal deploy modal_inference.py
Dev:      modal serve modal_inference.py

Upload weights first:
  modal run modal_inference.py::upload_weights
"""

import io
import base64
import modal

# ---------------------------------------------------------------------------
# Image — CUDA + diffusers (no StreamDiffusion; uses LCM-LoRA for speed)
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "xformers==0.0.23.post1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "diffusers==0.27.2",
        "transformers==4.38.2",
        "accelerate==0.27.2",
        "safetensors",
        "Pillow",
        "fastapi[standard]",
        "websockets",
    )
    # Bake model weights into the image so cold starts don't re-download ~4GB.
    # Image build is slow once; container startup becomes ~30s instead of 5min.
    .run_commands(
        "python -c \""
        "from diffusers import StableDiffusionImg2ImgPipeline; "
        "StableDiffusionImg2ImgPipeline.from_pretrained("
        "'runwayml/stable-diffusion-v1-5', cache_dir='/model-cache'"
        ")\"",
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('latent-consistency/lcm-lora-sdv1-5', cache_dir='/model-cache')"
        "\"",
    )
)

app = modal.App("glovely", image=image)

# Volume stores LoRA weights — persists across container restarts.
volume = modal.Volume.from_name("glovely-weights", create_if_missing=True)
VOLUME_PATH = "/weights"
LORA_FILENAME = "pytorch_lora_weights.safetensors"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
MODEL_CACHE = "/model-cache"


# ---------------------------------------------------------------------------
# Pipeline class — kept warm between requests
# ---------------------------------------------------------------------------

@app.cls(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    min_containers=1,      # one container stays alive to avoid cold starts
    scaledown_window=300,  # scale to zero after 5min idle
)
class GlovePipeline:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler

        lora_path = f"{VOLUME_PATH}/{LORA_FILENAME}"

        print(f"Loading {BASE_MODEL} from cache...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        # LCM-LoRA: swap scheduler + load LoRA for 4-step inference
        print("Loading LCM-LoRA from cache...")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(LCM_LORA_ID, cache_dir=MODEL_CACHE, adapter_name="lcm")

        # Glove LoRA on top of LCM
        print(f"Loading glove LoRA from {lora_path}...")
        self.pipe.load_lora_weights(lora_path, adapter_name="glove")

        # Combine both LoRAs: LCM drives speed, glove drives style
        self.pipe.set_adapters(["lcm", "glove"], adapter_weights=[1.0, 0.8])

        self.pipe.enable_xformers_memory_efficient_attention()

        # Warmup
        from PIL import Image
        dummy = Image.new("RGB", (512, 512))
        self.pipe(
            prompt="ohwx glove",
            image=dummy,
            num_inference_steps=4,
            guidance_scale=1.0,
            strength=0.6,
        )
        print("Pipeline ready.")

    @modal.method()
    def generate(self, image_b64: str, prompt: str = "ohwx glove") -> str:
        """
        Takes a base64-encoded JPEG (cropped hand frame, 512×512).
        Returns a base64-encoded JPEG (generated glove frame).
        """
        import torch
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt="blurry, deformed, low quality, extra fingers",
                image=input_image,
                num_inference_steps=4,   # LCM: 4 steps is enough
                guidance_scale=1.0,      # LCM works best at guidance_scale=1
                strength=0.6,            # how much to deviate from input pose
            )

        output_image = result.images[0]

        buf = io.BytesIO()
        output_image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@app.function()
@modal.fastapi_endpoint(method="POST")
def infer(body: dict) -> dict:
    """
    POST { "image": "<base64>", "prompt": "ohwx glove" }
    →    { "image": "<base64>" }
    """
    result = GlovePipeline().generate.remote(
        image_b64=body["image"],
        prompt=body.get("prompt", "ohwx glove"),
    )
    return {"image": result}


# ---------------------------------------------------------------------------
# WebSocket endpoint (alternative to HTTP for lower round-trip latency)
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
    modal run modal_inference.py::upload_weights
    """
    import os

    local_path = "./glove-lora/pytorch_lora_weights.safetensors"
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Not found: {local_path}")

    print(f"Uploading {local_path} → {VOLUME_PATH}/{LORA_FILENAME}")
    with open(local_path, "rb") as f:
        data = f.read()

    _upload_file.remote(data, LORA_FILENAME)
    print("Done.")


@app.function(volumes={VOLUME_PATH: volume})
def _upload_file(data: bytes, filename: str):
    path = f"{VOLUME_PATH}/{filename}"
    with open(path, "wb") as f:
        f.write(data)
    volume.commit()
    print(f"Wrote {len(data) / 1e6:.1f}MB → {path}")
