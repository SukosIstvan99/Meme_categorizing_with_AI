from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Modell és processor betöltése
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Pozitív és negatív kategóriák szöveges leírása
categories = ["positive meme", "negative meme"]

def classify_meme(image_path):
    """
    Eldönti, hogy egy mém pozitív vagy negatív kategóriába tartozik.
    
    :param image_path: A mém képfájljának elérési útja
    :return: Pozitív és negatív valószínűségek
    """
    # Kép betöltése
    image = Image.open(image_path).convert("RGB")
    
    # Kép és szöveges leírások feldolgozása
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)
    
    # CLIP modell predikciója
    outputs = model(**inputs)
    
    # Valószínűségek kinyerése
    logits_per_image = outputs.logits_per_image  # Képhez tartozó logitok
    probs = logits_per_image.softmax(dim=1)  # Valószínűségek
    
    # Eredmények visszaadása
    return {
        "positive": probs[0][0].item(),
        "negative": probs[0][1].item()
    }

# Példa használat
if __name__ == "__main__":
    image_path = "meme.jpg"  # Add meg a kép elérési útját
    result = classify_meme(image_path)
    
    print(f"Positive meme probability: {result['positive']:.4f}")
    print(f"Negative meme probability: {result['negative']:.4f}")
    
    # Kategorizálás eredménye
    if result["positive"] > result["negative"]:
        print("The meme is classified as Positive.")
    else:
        print("The meme is classified as Negative.")
