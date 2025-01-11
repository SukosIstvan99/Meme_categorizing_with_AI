import torch
import clip
from preprocess_dataset import load_dataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-B/32", device=device)


dataset = load_dataset()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


categories = dataset.classes
text_inputs = clip.tokenize(categories).to(device)

def train_clip_model():
    image_features = []
    for images, labels in data_loader:
        images = images.to(device)
        with torch.no_grad():
            batch_features = model.encode_image(images)
            image_features.append(batch_features.cpu())

   
    torch.save(torch.cat(image_features), "image_features.pt")
    print("Képjellemzők mentve.")

if __name__ == "__main__":
    train_clip_model()
    print()
#