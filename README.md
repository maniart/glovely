# glovely

An interactive web installation: a webcam detects your hand in real time, and a fine-tuned diffusion model renders it as a glove — in the surreal, weathered style of ~100 work gloves photographed on sidewalks. Every few seconds the render settles into a new generated glove. Users can snapshot a glove into a growing gallery.

## How it works

```
Browser (webcam)
  → MediaPipe Hands  — 21-landmark detection, crops hand region
  → StreamDiffusion  — img2img at moderate denoising strength (~0.5–0.7)
  → LoRA             — fine-tuned on the glove photo dataset (trigger: ohwx)
  → generated frame  → displayed with 3s hold + crossfade
  → snapshot         → object storage → Postgres gallery
```

The generative pipeline runs on a cloud GPU (Modal, A10G). The browser client handles webcam capture and MediaPipe entirely client-side.

## Stack

| Layer | Technology |
|---|---|
| Hand detection | MediaPipe Hands (`@mediapipe/tasks-vision`) |
| Generative model | Stable Diffusion 1.5 + custom LoRA |
| Real-time inference | StreamDiffusion (img2img) |
| GPU hosting | Modal (A10G, scales to zero) |
| Frontend | Next.js + MediaPipe |
| Storage | TBD (S3/R2/Supabase) |
| Database | TBD (Postgres) |

## Project structure

```
modal_inference.py   # Modal GPU inference server (StreamDiffusion + LoRA)
validate_lora.py     # Test inference script — run after training to inspect outputs
generate_captions.py # BLIP caption generation for training dataset
resize-512.py        # Center-crop + resize raw images to 512×512
check_token.py       # Verify trigger word tokenization (ohwx)
```

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU for training (or Apple Silicon via MPS, slower)
- [Modal](https://modal.com) account for cloud inference

### Local environment

```bash
python -m venv glove-env
source glove-env/bin/activate
pip install torch torchvision accelerate diffusers transformers safetensors Pillow datasets
```

### Training data

Place raw images in `raw_gloves/`, then:

```bash
# Convert HEIC → JPG (macOS)
for f in raw_gloves/*.HEIC; do sips -s format jpeg "$f" --out "${f%.HEIC}.jpg"; done

# Resize to 512×512
python resize-512.py

# Generate captions
python generate_captions.py
```

This produces `glove_dataset/` with images and `metadata.jsonl`.

### LoRA training

```bash
# Install diffusers from source (required for train script)
git clone https://github.com/huggingface/diffusers
pip install -e ./diffusers
pip install -r ./diffusers/examples/text_to_image/requirements.txt

accelerate config   # single machine, no distributed, mixed precision

accelerate launch ./diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="./glove_dataset" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --max_train_steps=1500 \
  --rank=16 \
  --output_dir="./glove-lora"
```

Trained weights land in `glove-lora/pytorch_lora_weights.safetensors`.

### Validate the LoRA

```bash
python validate_lora.py
# Outputs 5 test images to lora_validation/
# Images 1–4 use trigger word (ohwx), image 5 is a baseline without it
```

## Deployment

### Upload weights to Modal

```bash
pip install modal
modal token new
modal run modal_inference.py::upload_weights
```

### Deploy inference server

```bash
modal deploy modal_inference.py    # production
modal serve modal_inference.py     # dev (hot reload)
```

This creates two endpoints:

- `POST /infer` — HTTP, one frame per request `{ image: "<base64>" } → { image: "<base64>" }`
- `WebSocket /ws` — persistent connection for lower-latency streaming

## Trigger word

The LoRA was trained with trigger word **`ohwx`** — a single clean CLIP token that doesn't collide with boxing-glove imagery in the base model. Include it in all inference prompts: `"ohwx glove"`, `"ohwx worn glove on sidewalk"`, etc.

## License

TBD
