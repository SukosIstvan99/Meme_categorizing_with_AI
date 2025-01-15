from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Modell és processor betöltése
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Kép és szöveg betöltése
image = Image.open("meme.jpg")
texts = ["positive meme", "negative meme"]

# Adatok előkészítése
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Logitok kinyerése
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Eredmények
print(f"Positive meme: {probs[0][0]:.4f}")
print(f"Negative meme: {probs[0][1]:.4f}")
