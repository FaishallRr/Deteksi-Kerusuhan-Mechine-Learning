import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 64

POS_SOURCES = [
    Path("dataset/sajam_crops/sajam_pos"),
    Path("dataset/sajam_google_images"),
    Path("dataset/sajam_extracted_crops"),
]
NEG_DIR = Path("dataset/sajam_crops/sajam_neg")

class SajamDataset(Dataset):
    def __init__(self, pos_files, neg_files, augment=False):
        self.files = [(p, 1) for p in pos_files] + [(n, 0) for n in neg_files]
        self.augment = augment
        print(f"Dataset: {len(pos_files)} positive, {len(neg_files)} negative, total {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.0

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            brightness = 0.8 + 0.4 * np.random.rand()
            img = np.clip(img * brightness, 0, 1)
            if np.random.rand() > 0.5:
                noise = np.random.randn(*img.shape) * 0.02
                img = np.clip(img + noise, 0, 1)

        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class SajamCNN(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)

def load_images():
    pos_files = []
    for src in POS_SOURCES:
        if src.name == "sajam_pos":
            files = list(src.glob("*.jpg"))
        elif src.name == "sajam_google_images":
            files = list(src.rglob("*.*"))
            files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        else:
            files = list(src.rglob("*.jpg"))
        pos_files.extend(files)
        print(f"  {src.name}: {len(files)} images")

    neg_files = list(NEG_DIR.glob("*.jpg"))
    print(f"  sajam_neg: {len(neg_files)} images")

    return pos_files, neg_files

def train():
    print("Loading weapon images...")
    pos_files, neg_files = load_images()
    print(f"Total: {len(pos_files)} positive, {len(neg_files)} negative")

    full = SajamDataset(pos_files, neg_files, augment=True)
    val_size = int(0.2 * len(full))
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=0)

    model = SajamCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LR)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/sajam_cnn_verify.pt")
            print(f"  -> Saved best model ({best_acc:.3f})")

    print(f"\nDone! Best val acc: {best_acc:.3f}")
    print(f"Model saved: models/sajam_cnn_verify.pt")

if __name__ == "__main__":
    train()
