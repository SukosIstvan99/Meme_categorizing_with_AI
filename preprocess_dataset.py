from torchvision import datasets, transforms

def load_dataset(dataset_path="./dataset"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.CenterCrop(224),    
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))  
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return dataset

if __name__ == "__main__":
    dataset = load_dataset()
    print("Kategóriák:", dataset.classes)
    print(f"Adatok száma: {len(dataset)}")
