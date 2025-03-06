import torch
import argparse
import json
import numpy as np
from torchvision import models,transforms
from torch import nn,optim
from PIL import Image

# Load checkpoint function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location= "cpu")

    model = models.resnet34(pretrained=True)

    # Define classifier (should match train.py)
    classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(checkpoint["class_to_idx"])),  # Match number of classes
        nn.LogSoftmax(dim=1)
    )

    model.fc = classifier
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    print(model.class_to_idx)
    return model

# Image processing function
def process_image(image_path):
    image = Image.open(image_path)

    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Prediction function
# Prediction function
def predict(image_path, model, top_k=5):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    image = process_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)

    # Get number of classes from model
    num_classes = len(model.class_to_idx)

    # Ensure top_k does not exceed available classes
    top_k = min(top_k, num_classes)

    # Get top-k predictions
    probabilities, indices = torch.exp(output).topk(top_k)

    probabilities = probabilities.cpu().numpy().squeeze()
    indices = indices.cpu().numpy().squeeze()

    # Convert indices to actual class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [str(idx_to_class[idx]) for idx in indices]
    # print(f"Predicted indices: {indices}")
    # print(f"idx_to_class mapping: {idx_to_class}")
    # print(f"Initial mapped classes: {classes}")

    return probabilities, classes

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower name using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to saved model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load model
    model = load_checkpoint(args.checkpoint)
    model = model.to("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Make prediction
    probabilities, classes = predict(args.image_path, model, args.top_k)



    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(cls, "Unknown") for cls in classes]

    # Print results
    print(f"Top {args.top_k} Predictions:")
    for i in range(len(classes)):
        print(f"{classes[i]}: {probabilities[i]:.4f}")
print(f"Predicted class labels before mapping: {classes}")
print(model.class_to_idx)
