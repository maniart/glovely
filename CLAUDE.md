# Project: Gesture-Triggered Generative Glove Installation

## Concept
An interactive web experience: webcam detects hand gestures, and a fine-tuned
generative model renders the gesture "as a glove" — in the surreal, morphing
style of an old Runway.ml "Latent Spacewalk" made years ago from ~100 photos
of work gloves found on sidewalks. Every 3 seconds the render settles into a
new generated glove. Users can snapshot a glove into a growing gallery/database.

## Architecture decision (already made, don't relitigate)
- **Hand detection (client-side, browser):** MediaPipe Hands
  (`@mediapipe/tasks-vision`) — 21 landmarks, real-time, used to crop/mask
  just the hand region from the webcam feed (no background).
- **Generation (real-time img2img):** StreamDiffusion
  (github.com/cumulo-autumn/StreamDiffusion) — feeds the cropped hand frame
  in as the init image at moderate denoising strength (~0.5–0.7), so gesture
  shape/pose is preserved while texture is hallucinated as "glove."
- **Style conditioning:** a LoRA fine-tuned on the glove photo dataset,
  loaded into the StreamDiffusion pipeline.
- **Cadence:** generation runs continuously/fast; client-side display logic
  holds a frame for 3s with a crossfade into the next, rather than showing
  every raw frame.
- **Snapshot flow:** canvas capture of current displayed frame → object
  storage (S3/Supabase/R2) → Postgres row (image URL, gesture landmarks,
  timestamp) → growing gallery.
- Rejected approach: StyleGAN2-ADA latent walk (the original Runway
  technique) — doesn't support real-time conditioned input the way this
  interaction model needs. Diffusion + img2img was chosen instead.

## Progress so far
- [x] `accelerate config` completed (single machine, no distributed
  training, mixed precision set appropriately for the hardware, no dynamo)
- [x] HEIC → JPG conversion done via a zsh function (`heic2jpg`)
- [x] Images center-cropped and resized to 512×512 into `./glove_dataset`
- [x] Captions auto-generated with BLIP (`Salesforce/blip-image-captioning-base`)
      into `./glove_dataset/metadata.jsonl` (HF imagefolder format)
- [x] Trigger word chosen: **`ohwx`** (verified as a single clean token via
      `CLIPTokenizer` — not `sks`, which collides with boxing-glove imagery
      in the base model)
- [x] Authenticated with `huggingface-cli login`
- [x] Cloned `huggingface/diffusers` repo for `examples/text_to_image/train_text_to_image_lora.py`
- [ ] **NEXT STEP — not yet run:** launch the actual LoRA training:
  ```bash
  accelerate launch train_text_to_image_lora.py \
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
- [ ] Validate the trained LoRA with test inference before moving on
- [ ] Scaffold StreamDiffusion server + MediaPipe client
- [ ] Wire up the 3s display cadence and crossfade
- [ ] Build snapshot → storage → DB flow

## Context
- Senior JS/TS engineer, comfortable with code — relatively new to ML/DL
  specifically. Using this project to learn ML/DL terminology as it goes
  (weights, parameters, PEFT, LoRA already covered).
- Keeping a running WIP markdown glossary of ML terms as they come up —
  flag new terms plainly when they arise, don't over-explain unless asked.
- Prefers concise, actionable responses by default.
