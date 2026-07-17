"""
Train MIL model on final curated dataset.

Labels: demo_rusuh=1 (anomaly), demo_damai=0, normal=0

Usage:
    python core/train_mil_final.py

Expects:
    features/final_dataset/metadata.json (from extract_all_features.py)
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from core.mil_ranking import MILRankingModel

FEAT_DIR = Path("features/final_dataset")
META_PATH = FEAT_DIR / "metadata.json"
OUTPUT_MODEL = "models/mil_final.pt"
PLOT_DIR = Path("runs/final_training")
BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 15


def load_metadata():
    with open(META_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} videos from metadata")
    return data


def prepare_data(metadata):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for item in metadata:
        feat = np.load(item["path"])
        split = item["split"]
        label = item["label"]
        label_arr = np.full(feat.shape[0], label)

        if split == "train":
            X_train.append(feat)
            y_train.append(label_arr)
        elif split == "val":
            X_val.append(feat)
            y_val.append(label_arr)
        elif split == "test":
            X_test.append(feat)
            y_test.append(label_arr)

    def concat(arr_list):
        return np.concatenate(arr_list) if arr_list else np.array([])

    X_train = concat(X_train)
    y_train = concat(y_train)
    X_val = concat(X_val)
    y_val = concat(y_val)
    X_test = concat(X_test)
    y_test = concat(y_test)

    print(f"\nSegments: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        cnt = Counter(y)
        print(f"  {name}: normal={cnt[0]} rusuh={cnt[1]}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train(X_train, y_train, X_val, y_val):
    device = torch.device("cpu")
    input_dim = X_train.shape[1]

    model = MILRankingModel(input_dim=input_dim, hidden_units=512).to(device)

    class_counts = Counter(y_train)
    n_normal = class_counts[0]
    n_rusuh = class_counts[1]
    pos_weight = torch.tensor([n_normal / max(n_rusuh, 1)]).to(device)
    print(f"  pos_weight={pos_weight.item():.2f} (normal={n_normal} rusuh={n_rusuh})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model.forward_raw(batch_X)
            loss = criterion(logits, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model.forward_raw(batch_X)
                loss = criterion(logits, batch_y.unsqueeze(1))
                val_loss += loss.item() * batch_X.size(0)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"  -> Saved best model (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(OUTPUT_MODEL, map_location=device, weights_only=True))
    print(f"\nBest model from epoch {best_epoch} (val_auc={best_val_auc:.4f})")

    # Plot
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_auc"])
    axes[1].set_title("Val AUC"); axes[1].axhline(best_val_auc, color="r", ls="--")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_history.png", dpi=150)
    print(f"Plot saved to {PLOT_DIR / 'training_history.png'}")

    return model


def evaluate(model, X_test, y_test, label="test"):
    device = torch.device("cpu")
    model.eval()
    dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits = model.forward_raw(batch_X)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Best threshold
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        pred_bin = (all_preds >= t).astype(int)
        f1 = f1_score(all_labels, pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    pred_bin = (all_preds >= best_thresh).astype(int)
    cm = confusion_matrix(all_labels, pred_bin)
    precision = precision_score(all_labels, pred_bin, zero_division=0)
    recall = recall_score(all_labels, pred_bin, zero_division=0)
    f1 = f1_score(all_labels, pred_bin, zero_division=0)

    print(f"\n--- {label.upper()} SET EVALUATION ---")
    print(f"  AUC: {auc:.4f}")
    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return auc, precision, recall, f1, best_thresh


def main():
    metadata = load_metadata()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(metadata)

    print("\nTraining...")
    model = train(X_train, y_train, X_val, y_val)

    evaluate(model, X_test, y_test, label="test")
    evaluate(model, X_val, y_val, label="val")

    print(f"\nModel saved: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
