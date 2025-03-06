import torch
import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim

def get_input_args():
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument("data_dir", type=str, help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoint")
    parser.add_argument("--arch", type=str, default="resnet34", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()

def train():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    train_dir = args.data_dir + "/train"
    # Load dataset
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # Load model
    model = models.resnet34(pretrained=True)

    # Define classifier
    classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(train_data.classes)),  # Automatically detect number of classes
        nn.LogSoftmax(dim=1)
    )

    model.fc = classifier
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save model properly
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_data.class_to_idx,  # ✅ Save class mapping
        "architecture": "resnet34",
        "classifier": classifier
    }

    torch.save(checkpoint, f"{args.save_dir}/checkpoint.pth")
    print("Training complete. Model saved!")

if __name__ == "__main__":
    train()
