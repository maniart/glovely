from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
print(tokenizer.tokenize("ohwx"))