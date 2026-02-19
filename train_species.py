#!/usr/bin/env python3
"""Bird species retraining script — fine-tune a MobileNetV2 on labeled bird crops.

This script is designed to run on a desktop/laptop with a GPU, NOT on the Pi.
It fine-tunes a MobileNetV2 classifier on user-labeled bird crops, then exports
to ONNX format compatible with species_classifier.py.

Labeling workflow:
    1. bird_monitor.py saves crops to bird_crops/YYYY-MM-DD/
    2. User creates bird_crops/labeled/<species_name>/ folders
    3. User moves/copies crops into the correct species folders
    4. Run this script to fine-tune and export

Usage:
    python3 train_species.py --data bird_crops/labeled --epochs 20
    python3 train_species.py --data bird_crops/labeled --epochs 20 --resume models/species_mobilenet.pth

The exported ONNX model can replace the EfficientNet-B7 model on the Pi.
Copy the output files to the Pi's models/ directory:
    scp models/species_retrained.onnx pi@<PI_IP>:/home/pi/ai/models/efficientnet_b7_backyard_birds.onnx
    scp models/species_labels.txt pi@<PI_IP>:/home/pi/ai/models/species_labels.txt

The bird_monitor.py hot-reload will pick up the new model within 60 seconds.

Requirements (install on training machine):
    pip install torch torchvision onnx onnxruntime Pillow
"""

import argparse
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
except ImportError:
    print("PyTorch not found. Install with: pip install torch torchvision")
    print("This script is meant to run on a desktop/laptop with GPU, not the Pi.")
    sys.exit(1)

# Input size for species_classifier.py (must match ONNX model input)
INPUT_SIZE = 600


def get_transforms():
    """Training and validation transforms matching species_classifier.py preprocessing."""
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE + 40, INPUT_SIZE + 40)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),  # scales to [0, 1]
    ])
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform


def build_model(num_classes):
    """Build a MobileNetV2 classifier for the given number of species."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Replace the classifier head
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}, "
                  f"loss={loss.item():.4f}, acc={100.*correct/total:.1f}%", flush=True)

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def export_onnx(model, num_classes, output_path, device):
    """Export the trained model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print(f"  Exported ONNX model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune bird species classifier")
    parser.add_argument("--data", required=True,
                        help="Path to labeled data (ImageFolder format: data/<species>/img.jpg)")
    parser.add_argument("--output", default="models",
                        help="Output directory for model and labels (default: models)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading data from: {args.data}")
    train_transform, val_transform = get_transforms()
    full_dataset = datasets.ImageFolder(args.data, transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        print(f"Error: Need at least 2 species folders, found {num_classes}")
        sys.exit(1)

    print(f"  Found {len(full_dataset)} images across {num_classes} species:")
    for i, name in enumerate(class_names):
        count = sum(1 for _, label in full_dataset.samples if label == i)
        print(f"    {name}: {count} images")

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    # Apply val transform to validation set
    val_dataset.dataset = datasets.ImageFolder(args.data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"  Train: {train_size}, Val: {val_size}")

    # Build model
    model = build_model(num_classes).to(device)
    if args.resume and os.path.isfile(args.resume):
        print(f"  Resuming from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_val_acc = 0.0
    checkpoint_path = os.path.join(args.output, "species_mobilenet.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.1f}%")
        print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint ({val_acc:.1f}%)")

    # Load best model and export
    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Export ONNX
    onnx_path = os.path.join(args.output, "species_retrained.onnx")
    export_onnx(model, num_classes, onnx_path, device)

    # Save labels
    labels_path = os.path.join(args.output, "species_labels.txt")
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"  Saved labels to: {labels_path}")

    # Verify ONNX
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        out = sess.run(None, {"input": dummy})
        print(f"  ONNX verification passed: output shape {out[0].shape}")
    except ImportError:
        print("  (Skipping ONNX verification — onnxruntime not installed)")

    print(f"\nDone! To deploy on Pi:")
    print(f"  scp {onnx_path} pi@<PI_IP>:/home/pi/ai/models/efficientnet_b7_backyard_birds.onnx")
    print(f"  scp {labels_path} pi@<PI_IP>:/home/pi/ai/models/species_labels.txt")


if __name__ == "__main__":
    main()
