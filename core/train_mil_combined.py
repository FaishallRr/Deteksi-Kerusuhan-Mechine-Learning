import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
from collections import Counter

from core.mil_ranking import MILRankingModel


def load_multiple_features(metadata_paths):
    all_X, all_y, all_sources = [], [], []
    for path in metadata_paths:
        name = Path(path).parent.name
        with open(path) as f:
            metadata = json.load(f)
        for item in metadata:
            feat = np.load(item["path"])
            feat = feat.reshape(feat.shape[0], -1)
            for seg_idx in range(feat.shape[0]):
                all_X.append(feat[seg_idx])
                all_y.append(item["label"])
                all_sources.append(name)
    return np.array(all_X), np.array(all_y), all_sources


def train_with_optimal_params(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    print(f"Using device: {device}")
    print(f"Input dim: {input_dim}")

    model = MILRankingModel(
        input_dim=input_dim,
        hidden_units=512,
    ).to(device)

    class_counts = Counter(y_train)
    n_normal = class_counts[0]
    n_anomaly = class_counts[1]
    pos_weight = torch.tensor([n_normal / max(n_anomaly, 1)]).to(device)
    print(f"Class weights - Normal: {n_normal}, Anomaly: {n_anomaly}, pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float().unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float().unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float("inf")
    best_model_state = None
    patience = 20
    patience_counter = 0
    n_epochs = 500

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    print(f"\nStarting training with {len(X_train)} samples...")
    print(f"  Val samples: {len(X_val)}")
    print(f"  Batch size: 32")
    print(f"  Learning rate: 0.0005")
    print(f"  Hidden units: 512")
    print(f"  Epochs: {n_epochs} (with early stopping)")
    print(f"  Early stopping patience: {patience}")
    print()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model.forward_raw(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model.forward_raw(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                preds = torch.sigmoid(logits)
                all_val_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_val_labels.extend(batch_y.cpu().numpy().flatten().tolist())

        train_loss = epoch_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_auc = roc_auc_score(all_val_labels, all_val_preds) if len(set(all_val_labels)) > 1 else 0.5
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_avg)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss_avg:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model, history, best_val_loss


def evaluate(model, X_test, y_test, name="Test"):
    device = next(model.parameters()).device
    model.eval()

    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float().unsqueeze(1),
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model.forward_raw(batch_x)
            preds = torch.sigmoid(logits)
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(batch_y.cpu().numpy().flatten().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    print(f"\n{'='*50}")
    print(f"EVALUATION - {name}")
    print(f"{'='*50}")

    cm = confusion_matrix(all_labels, pred_binary)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal    {tn:5d}  {fp:5d}")
    print(f"       Anomaly   {fn:5d}  {tp:5d}")

    results = {
        "roc_auc": roc_auc_score(all_labels, all_preds),
        "precision": precision_score(all_labels, pred_binary, zero_division=0),
        "recall": recall_score(all_labels, pred_binary, zero_division=0),
        "f1": f1_score(all_labels, pred_binary, zero_division=0),
        "accuracy": (tp+tn)/(tn+fp+fn+tp),
        "false_positive_rate": fp/(tn+fp) if (tn+fp) > 0 else 0,
        "false_negative_rate": fn/(fn+tp) if (fn+tp) > 0 else 0,
        "confusion_matrix": cm.tolist(),
    }

    print(f"\nMetrics:")
    for k, v in results.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(all_labels, pred_binary, target_names=["Normal", "Anomaly"], zero_division=0))

    return results


def plot_history(history, save_path="models/training_history_combined.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["val_auc"], label="Val AUC", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history["lr"], label="Learning Rate", color="red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining history saved: {save_path}")


if __name__ == "__main__":
    metadata_paths = [
        "features/ucf_crime/metadata.json",
        "features/scvd/metadata.json",
        "features/indonesia_v3/metadata.json",
    ]

    print("Loading features from all datasets...")
    X, y, sources = load_multiple_features(metadata_paths)
    print(f"Total segments: {len(X)}")
    counter = Counter(y)
    source_counter = Counter(sources)
    print(f"Normal: {counter[0]} | Anomaly: {counter[1]}")
    print(f"Sources: {dict(source_counter)}")

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sources, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nSplit: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    train_counter = Counter(y_train)
    print(f"Train: Normal={train_counter[0]}, Anomaly={train_counter[1]}")

    print("\nStarting training...")
    model, history, best_val_loss = train_with_optimal_params(X_train, y_train, X_val, y_val)

    model_path = "models/mil_model_combined.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    plot_history(history, "models/training_history_combined.png")

    results = evaluate(model, X_test, y_test, "Combined Test Set")

    source_test_sets = {}
    for s in set(s_test):
        mask = [st == s for st in s_test]
        source_test_sets[s] = (X_test[mask], y_test[mask])

    for src, (Xs, ys) in source_test_sets.items():
        if len(set(ys)) > 1 and len(Xs) > 10:
            evaluate(model, Xs, ys, f"Per-Source: {src}")
        else:
            print(f"\nSkipping {src}: insufficient class diversity ({len(Xs)} samples)")

    results_path = "models/evaluation_results_combined.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")
