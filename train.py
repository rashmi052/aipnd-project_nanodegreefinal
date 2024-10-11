import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset of flowers.")
    parser.add_argument('data_dir', type=str, help='Path to the dataset folder')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (e.g., vgg16, alexnet)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of units in the hidden layer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {running_loss / len(dataloaders['train'])}")

def save_checkpoint(model, arch, save_dir, class_to_idx):
    checkpoint = {
        'architecture': arch,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    
def main():
    args = get_input_args()
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_dir = args.data_dir
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True)
    }
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    train(model, criterion, optimizer, dataloaders, args.epochs, device)
    save_checkpoint(model, args.arch, args.save_dir, image_datasets['train'].class_to_idx)

if __name__ == "__main__":
    main()
