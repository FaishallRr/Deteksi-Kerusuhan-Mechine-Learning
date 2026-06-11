import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from preprocessing.pipeline import PreprocessingPipeline
from core.mil_ranking import MILRankingModel


def generate_synthetic_training_data(normal_videos, anomaly_videos):
    pipeline = PreprocessingPipeline(device="cpu")
    X, y = [], []

    for vid in normal_videos:
        feats = pipeline.run(vid)
        if feats is not None:
            for f in feats:
                X.append(f)
                y.append(0)

    for vid in anomaly_videos:
        feats = pipeline.run(vid)
        if feats is not None:
            for f in feats:
                X.append(f)
                y.append(1)

    return np.array(X), np.array(y)


def train_mil_model(X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    device = torch.device("cpu")
    model = MILRankingModel(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0

        for i in range(0, len(X_train), 16):
            idx = perm[i : i + 16]
            batch_x = torch.from_numpy(X_train[idx]).float().to(device)
            batch_y = torch.from_numpy(y_train[idx]).float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_x = torch.from_numpy(X_val).float().to(device)
        val_y = torch.from_numpy(y_val).float().to(device).unsqueeze(1)
        val_preds = model(val_x)
        val_loss = criterion(val_preds, val_y).item()

        val_acc = ((val_preds > 0.5) == (val_y > 0.5)).float().mean().item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    device = next(model.parameters()).device
    model.eval()
    test_x = torch.from_numpy(X_test).float().to(device)
    test_y = torch.from_numpy(y_test).float().to(device).unsqueeze(1)
    with torch.no_grad():
        preds = model(test_x)
    acc = ((preds > 0.5) == (test_y > 0.5)).float().mean().item()
    print(f"\nTest Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    import glob
    normal_videos = sorted(glob.glob("sample_videos/normal_*.mp4"))
    anomaly_videos = sorted(glob.glob("sample_videos/anomaly_*.mp4"))

    print("Extracting features for training...")
    X, y = generate_synthetic_training_data(normal_videos, anomaly_videos)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features each")
    print(f"Labels: {y.sum()} anomaly, {len(y) - y.sum()} normal")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model = train_mil_model(X_train, y_train, X_val, y_val, epochs=50, lr=0.001)

    evaluate_model(model, X_test, y_test)

    torch.save(model.state_dict(), "models/mil_model.pt")
    print("\nModel saved to models/mil_model.pt")

    test_sample = torch.from_numpy(X_test[:1]).float()
    with torch.no_grad():
        score = model(test_sample)
    print(f"\nSample prediction (should be {'anomaly' if y_test[0] == 1 else 'normal'}): {score.item():.4f}")
