from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os, json

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

lines = []
for fname in sorted(os.listdir("./glove_dataset")):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(f"./glove_dataset/{fname}").convert("RGB")
        inputs = processor(img, return_tensors="pt")
        caption = processor.decode(model.generate(**inputs)[0], skip_special_tokens=True)
        lines.append(json.dumps({"file_name": fname, "text": f"a photo of sks glove, {caption}"}))

with open("./glove_dataset/metadata.jsonl", "w") as f:
    f.write("\n".join(lines))