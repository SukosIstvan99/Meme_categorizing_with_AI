import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

class MemeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.img_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith(".jpg"):
                        self.img_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

dataset = MemeDataset(root_dir="dataset", transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

categories = dataset.classes
text_inputs = clip.tokenize(categories).to(device)

def extract_features(model, images, text_inputs):
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
    
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
    
    return image_features, text_features

def cosine_similarity(a, b):
    return torch.matmul(a, b.T) / (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True))

def compute_loss(image_features, text_features, labels):
    similarity = cosine_similarity(image_features, text_features)
    loss = torch.nn.functional.cross_entropy(similarity, labels)
    return loss

def train_clip_model():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            image_features, text_features = extract_features(model, images, text_inputs[labels])
            
            loss = compute_loss(image_features, text_features, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss}")

    torch.save(model.state_dict(), "models/fine_tuned_clip_model.pth")
    print("Modell finomhangolása kész és mentve.")

if __name__ == "__main__":
    train_clip_model()
