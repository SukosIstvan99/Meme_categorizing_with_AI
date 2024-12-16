from open_flamingo import FlamingoModel, FlamingoConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from preprocess_dataset import load_dataset

# Eszköz kiválasztása
device = "cuda" if torch.cuda.is_available() else "cpu"

# Open Flamingo konfiguráció
config = FlamingoConfig(
    vision_model="ViT-B/32",  # CLIP alapú képmodell
    text_model="gpt2",  # GPT alapú szövegmodell
    dim=512,  # Feature dimenzió
    cross_attn_every=4  # Cross-attention rétegek száma
)

# Modell és tokenizer betöltése
model = FlamingoModel(config).to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Adathalmaz betöltése
dataset = load_dataset()  # Visszatérési értéke: (image_tensor, text) párok
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Tokenizálás szövegekre
def tokenize_text(texts, tokenizer, max_len=128):
    tokenized = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    return tokenized.input_ids  # Az input_ids-t adja vissza (bemeneti tokenek)


# Tréning függvény
def train_flamingo(model, data_loader, tokenizer, epochs=3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for images, texts in data_loader:
            # Képek és szövegek feldolgozása
            images = images.to(device)
            tokenized_texts = tokenize_text(texts, tokenizer).to(device)

            # Előrehaladás
            outputs = model(images, tokenized_texts)

            # Ha expliciten nincs veszteség:
            if not hasattr(outputs, "loss"):
                raise ValueError("Az Open Flamingo model nem ad vissza közvetlen loss-t!")

            loss = outputs.loss  # Tegyük fel, hogy van ilyen attribútum

            # Gradiensek frissítése
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")


# Jellemzők kinyerése
def extract_features(model, data_loader):
    model.eval()
    all_features = []

    with torch.no_grad():
        for images, texts in data_loader:
            images = images.to(device)
            tokenized_texts = tokenize_text(texts, tokenizer).to(device)

            # Fő jellemzők kinyerése
            features = model(images, tokenized_texts)
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), "flamingo_features.pt")
    print("Jellemzők elmentve.")


if __name__ == "__main__":
    # Modell, tokenizer, adathalmaz
    model = FlamingoModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Tréning vagy feature extraction
    train_flamingo(model, data_loader, tokenizer, epochs=3)
    print("Tréning kész.")