from torchvision import datasets, transforms

def load_dataset(dataset_path="./dataset"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return dataset

if __name__ == "__main__":
    dataset = load_dataset()
    print("Kategóriák:", dataset.classes)
    print(f"Adatok száma: {len(dataset)}")
#