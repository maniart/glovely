from PIL import Image
import os

src_dir = "./raw_gloves"
out_dir = "./glove_dataset"
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        # center-crop to square, then resize
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((512, 512))
        img.save(os.path.join(out_dir, fname))