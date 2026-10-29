
from pathlib import Path
from collections import Counter
import random
import sys

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

DATASET = Path(r"C:\Phish360Data\phish360\Phish360")
MODEL_PATH = Path("models/efficientnet_b0_phish360.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2
SAMPLES_PER_CLASS = 600
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ScreenshotDataset(Dataset):
    def __init__(self):
        root = DATASET / "trainval"
        by_class = {0: [], 1: []}

        print("Scanning dataset folders...", flush=True)

        if not root.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {root}"
            )

        for folder in root.iterdir():
            if not folder.is_dir():
                continue

            name = folder.name.lower()

            # Official dataset naming convention:
            # Lxxxxx_legitimate = legitimate
            # Pxxxx_brand = phishing
            if name.startswith("l") and name.endswith("_legitimate"):
                label = 0
            elif name.startswith("p") and "_" in name:
                label = 1
            else:
                continue

            image_path = (
                folder / "SCREEN-SHOT" / "screen_shoot.png"
            )

            if image_path.is_file():
                by_class[label].append((image_path, label))

        print(
            "Candidate screenshots:",
            {label: len(paths) for label, paths in by_class.items()},
            flush=True,
        )

        self.samples = []
        skipped = 0

        for label in (0, 1):
            random.shuffle(by_class[label])
            accepted = 0
            candidates = len(by_class[label])

            print(
                f"Checking class {label} "
                f"({ 'legitimate' if label == 0 else 'phishing' })...",
                flush=True,
            )

            for index, (path, sample_label) in enumerate(
                by_class[label], start=1
            ):
                if accepted >= SAMPLES_PER_CLASS:
                    break

                try:
                    # verify() checks image integrity without decoding
                    # the entire image into memory.
                    with Image.open(path) as image:
                        image.verify()

                    self.samples.append((path, sample_label))
                    accepted += 1

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                ):
                    skipped += 1

                if index % 100 == 0:
                    print(
                        f"  Checked {index}/{candidates}; "
                        f"usable: {accepted}",
                        flush=True,
                    )

            print(
                f"  Class {label}: selected {accepted} images",
                flush=True,
            )

        random.shuffle(self.samples)

        counts = Counter(label for _, label in self.samples)
        print("Unreadable images skipped:", skipped, flush=True)
        print("Selected class counts:", dict(counts), flush=True)

        if counts[0] == 0 or counts[1] == 0:
            raise RuntimeError(
                "Both classes need readable screenshots. "
                "Check the dataset structure and folder names."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        # Open by file contents, not by the filename extension.
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def main():
    print("Starting screenshot model training...", flush=True)

    dataset = ScreenshotDataset()

    # Stratified split ensures both classes are represented in train
    # and validation sets.
    class_indices = {0: [], 1: []}

    for index, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(index)

    train_indices = []
    val_indices = []

    for label in (0, 1):
        indices = class_indices[label]
        random.shuffle(indices)

        val_count = max(1, int(0.2 * len(indices)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Loading pretrained EfficientNet-B0...", flush=True)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Freeze the pretrained feature extractor; train the final layer.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    feature_count = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(feature_count, 2)

    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=1e-3,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_number, (images, labels) in enumerate(
            train_loader, start=1
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            if batch_number % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_number}/{len(train_loader)}",
                    flush=True,
                )

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total else 0.0

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / len(train_data):.4f} | "
            f"Holdout accuracy: {accuracy:.4f}",
            flush=True,
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "efficientnet_b0",
        "image_size": IMAGE_SIZE,
        "class_to_idx": {
            "legitimate": 0,
            "phishing": 1,
        },
    }, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
