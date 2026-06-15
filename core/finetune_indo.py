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
from collections import Counter
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from core.mil_ranking import MILRankingModel


def load_multiple_features(metadata_paths: list) -> tuple:
    all_X, all_y = [], []
    for meta_path in metadata_paths:
        print(f"\nLoading: {meta_path}")
        with open(meta_path) as f:
            metadata = json.load(f)
        X, y = [], []
        for item in metadata:
            feat = np.load(item["path"])
            for seg_idx in range(feat.shape[0]):
                X.append(feat[seg_idx])
                y.append(item["label"])
        X = np.array(X)
        y = np.array(y)
        counter = Counter(y)
        print(f"  {len(X)} segments, {len(metadata)} videos")
        print(f"  Normal: {counter[0]} | Anomaly: {counter[1]}")
        all_X.append(X)
        all_y.append(y)

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    print(f"\nTotal combined: {len(X_combined)} segments, dim={X_combined.shape[1]}")
    print(f"Normal: {Counter(y_combined)[0]} | Anomaly: {Counter(y_combined)[1]}")
    return X_combined, y_combined


def train_with_optimal_params(X_train, y_train, X_val, y_val, device, model_path, history_path):
    input_dim = X_train.shape[1]

    model = MILRankingModel(input_dim=input_dim, hidden_units=512).to(device)

    class_counts = Counter(y_train)
    n_normal = class_counts[0]
    n_anomaly = class_counts[1]
    pos_weight = torch.tensor([n_normal / max(n_anomaly, 1)]).to(device)

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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    best_val_loss = float("inf")
    best_model_state = None
    patience = 25
    patience_counter = 0
    n_epochs = 500

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    print(f"\nStarting fine-tuning...")
    print(f"  Device: {device}")
    print(f"  Batch size: 64")
    print(f"  Learning rate: 0.0005")
    print(f"  Hidden units: 512")
    print(f"  Epochs: {n_epochs} (early stopping patience={patience})")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
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

    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    print(f"History saved: {history_path}")

    return model, history


def evaluate(model, X_test, y_test):
    device = next(model.parameters()).device
    model.eval()

    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float().unsqueeze(1),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    cm = confusion_matrix(all_labels, pred_binary)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal    {tn:5d}  {fp:5d}")
    print(f"       Anomaly   {fn:5d}  {tp:5d}")

    metrics = {
        "roc_auc": float(roc_auc_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, pred_binary)),
        "recall": float(recall_score(all_labels, pred_binary)),
        "f1": float(f1_score(all_labels, pred_binary)),
        "accuracy": float((tp+tn)/(tn+fp+fn+tp)),
        "false_positive_rate": float(fp/(tn+fp)),
        "false_negative_rate": float(fn/(fn+tp)),
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(all_labels)),
    }

    print(f"\nMetrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


def plot_history(history, save_path="models/training_history_finetune.png"):
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
    print(f"Training history saved: {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metadata_paths = [
        "features/ucf_crime/metadata.json",
        "features/indonesia/metadata.json",
    ]

    X, y = load_multiple_features(metadata_paths)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nSplit: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    model_path = "models/mil_model_indo.pt"
    history_path = "models/training_history_finetune.npz"

    model, history = train_with_optimal_params(
        X_train, y_train, X_val, y_val, device,
        model_path, history_path,
    )

    plot_history(history, "models/training_history_finetune.png")

    results = evaluate(model, X_test, y_test)

    with open("models/evaluation_results_indo.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: models/evaluation_results_indo.json")
