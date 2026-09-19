"""
Validate the trained glove LoRA by generating test images.
Run after training completes: python validate_lora.py
"""

import os
import torch
from diffusers import StableDiffusionPipeline

LORA_PATH = "./glove-lora"
OUTPUT_DIR = "./lora_validation"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# Test prompts — trigger word is 'ohwx'
PROMPTS = [
    "ohwx glove",
    "ohwx work glove on sidewalk",
    "ohwx worn leather glove on pavement",
    "ohwx glove, surreal, found object",
    # Control: no trigger word — should look like generic SD output
    "a work glove on the ground",
]

NEGATIVE_PROMPT = "blurry, low quality, deformed"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Loading base model: {BASE_MODEL}")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,  # MPS requires float32
    )

    lora_weights = os.path.join(LORA_PATH, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_weights):
        # Fall back to latest checkpoint
        checkpoints = sorted(
            [d for d in os.listdir(LORA_PATH) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Final weights not found, using checkpoint: {latest}")
            lora_weights = os.path.join(LORA_PATH, latest, "pytorch_lora_weights.safetensors")
        else:
            raise FileNotFoundError(f"No LoRA weights found in {LORA_PATH}")

    print(f"Loading LoRA weights: {lora_weights}")
    pipe.load_lora_weights(lora_weights)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(SEED)

    print(f"\nGenerating {len(PROMPTS)} images → {OUTPUT_DIR}/\n")
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt}")
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

        # Sanitize filename
        fname = prompt[:60].replace(" ", "_").replace(",", "").replace("/", "-")
        out_path = os.path.join(OUTPUT_DIR, f"{i+1:02d}_{fname}.png")
        image.save(out_path)
        print(f"    Saved: {out_path}")

    print(f"\nDone. Open {OUTPUT_DIR}/ and inspect:")
    print("  - Images 1-4 should show the glove dataset style (surreal, found-object texture)")
    print("  - Image 5 (no trigger word) should look like generic SD output")
    print("\nIf images 1-4 look too similar to 5, the LoRA didn't transfer style.")
    print("If images 1-4 look overfit/blurry/repetitive, consider fewer steps (~800-1000).")


if __name__ == "__main__":
    main()
