import torch
from torch import nn, optim
from torchvision import models
from PIL import Image
import numpy as np
import argparse
import json

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of a flower image.")
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 102),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).float()

def predict(image_path, model, top_k, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = process_image(image_path).unsqueeze(0).to(device)
        output = model(image)
        probabilities, indices = torch.exp(output).topk(top_k)
        return probabilities[0].tolist(), indices[0].add(1).tolist()

def main():
    args = get_input_args()
    model = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    probabilities, classes = predict(args.image_path, model, args.top_k, device)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        labels = [cat_to_name[str(cls)] for cls in classes]
    else:
        labels = classes
    print("Probabilities:", probabilities)
    print("Classes:", labels)

if __name__ == "__main__":
    main()
