import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_fashion_mnist_loader(batch_size, root="dataset/"):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # image normalization
    ])

    # Load FashionMNIST 
    dataset = datasets.FashionMNIST(root=root, transform=transform, download=True)

    # Generate DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader
