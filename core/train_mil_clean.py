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


def load_features_v3_normal_only(metadata_path):
    """Load only normal (label=0) from Indonesia v3."""
    all_X, all_y, all_sources = [], [], []
    name = Path(metadata_path).parent.name
    with open(metadata_path) as f:
        metadata = json.load(f)
    n_total = len(metadata)
    for item in metadata:
        if item["label"] != 0:
            continue
        feat = np.load(item["path"])
        feat = feat.reshape(feat.shape[0], -1)
        for seg_idx in range(feat.shape[0]):
            all_X.append(feat[seg_idx])
            all_y.append(item["label"])
            all_sources.append(f"{name}_normal")
    loaded = len(all_X)
    print(f"  {name}: loaded {loaded}/{n_total} items (normal only, skipped {n_total - sum(1 for m in metadata if m['label']==0)} anomaly videos)")
    return np.array(all_X), np.array(all_y), all_sources


def load_features(metadata_path):
    """Load all features from a dataset."""
    all_X, all_y, all_sources = [], [], []
    name = Path(metadata_path).parent.name
    with open(metadata_path) as f:
        metadata = json.load(f)
    for item in metadata:
        feat = np.load(item["path"])
        feat = feat.reshape(feat.shape[0], -1)
        for seg_idx in range(feat.shape[0]):
            all_X.append(feat[seg_idx])
            all_y.append(item["label"])
            all_sources.append(name)
    print(f"  {name}: loaded {len(all_X)} segments from {len(metadata)} videos")
    return np.array(all_X), np.array(all_y), all_sources


def train_model(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    model = MILRankingModel(input_dim=input_dim, hidden_units=512).to(device)

    class_counts = Counter(y_train)
    n_normal = class_counts[0]
    n_anomaly = class_counts[1]
    pos_weight = torch.tensor([n_normal / max(n_anomaly, 1)]).to(device)
    print(f"  Class balance: N={n_normal}, A={n_anomaly}, pos_weight={pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1)),
        batch_size=32, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float().unsqueeze(1)),
        batch_size=32, shuffle=False,
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience = 20
    patience_counter = 0
    n_epochs = 500
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    print(f"\nTraining: {len(X_train)} train, {len(X_val)} val, {n_epochs} epochs max")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model.forward_raw(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model.forward_raw(bx)
                val_loss += criterion(logits, by).item()
                all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
                all_labels.extend(by.cpu().numpy().flatten().tolist())

        train_loss = epoch_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_avg)
        history["val_auc"].append(val_auc)
        history["lr"].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val AUC: {val_auc:.4f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model, history


def evaluate(model, X_test, y_test, name="Test"):
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float().unsqueeze(1)),
        batch_size=32, shuffle=False,
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model.forward_raw(bx)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
            all_labels.extend(by.cpu().numpy().flatten().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)
    cm = confusion_matrix(all_labels, pred_binary)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"EVALUATION: {name}")
    print(f"{'='*50}")
    print(f"  Normal: {tn:4d} | Anomaly: {tp:4d}")
    print(f"  FP: {fp:4d} | FN: {fn:4d}")
    print(f"  ROC-AUC:    {roc_auc_score(all_labels, all_preds):.4f}")
    print(f"  Precision:  {precision_score(all_labels, pred_binary, zero_division=0):.4f}")
    print(f"  Recall:     {recall_score(all_labels, pred_binary, zero_division=0):.4f}")
    print(f"  F1:         {f1_score(all_labels, pred_binary, zero_division=0):.4f}")
    print(f"  Accuracy:   {(tp+tn)/(tn+fp+fn+tp):.4f}")
    print(f"  FPR:        {fp/(tn+fp):.4f} ({fp}/{tn+fp})")
    print(f"  FNR:        {fn/(fn+tp):.4f} ({fn}/{fn+tp})")

    return {
        "roc_auc": roc_auc_score(all_labels, all_preds),
        "precision": precision_score(all_labels, pred_binary, zero_division=0),
        "recall": recall_score(all_labels, pred_binary, zero_division=0),
        "f1": f1_score(all_labels, pred_binary, zero_division=0),
        "accuracy": (tp+tn)/(tn+fp+fn+tp),
        "fpr": fp/(tn+fp) if (tn+fp) > 0 else 0,
        "fnr": fn/(fn+tp) if (fn+tp) > 0 else 0,
    }


def load_features_from_indonesia_v3_anomaly(metadata_path):
    """Load only anomaly (label=1) from Indonesia v3 for diagnostic testing."""
    X, y, s = [], [], []
    with open(metadata_path) as f:
        metadata = json.load(f)
    for item in metadata:
        if item["label"] != 1:
            continue
        feat = np.load(item["path"])
        feat = feat.reshape(feat.shape[0], -1)
        for seg_idx in range(feat.shape[0]):
            X.append(feat[seg_idx])
            y.append(item["label"])
            s.append("indonesia_v3_anomaly_DIAGNOSTIC")
    return np.array(X), np.array(y), s


if __name__ == "__main__":
    print("=" * 60)
    print("DIAGNOSTIC: Test OLD (contaminated) model on Indonesia v3 anomaly")
    print("=" * 60)
    old_model = MILRankingModel(input_dim=1024, hidden_units=512)
    old_model.load_state_dict(torch.load("models/mil_model_combined.pt", map_location="cpu"))
    old_model.eval()

    X_diag, y_diag, _ = load_features_from_indonesia_v3_anomaly("features/indonesia_v3/metadata.json")
    print(f"\nIndonesia v3 anomaly diagnostic samples: {len(X_diag)}")
    evaluate(old_model, X_diag, y_diag, "OLD CONTAMINATED model on news clips")

    print("\n\n" + "=" * 60)
    print("CLEAN TRAINING: UCF-Crime + SCVD + Indonesia v3 NORMAL only")
    print("=" * 60)

    all_X, all_y, all_sources = [], [], []
    for path in ["features/ucf_crime/metadata.json", "features/scvd/metadata.json"]:
        Xp, yp, sp = load_features(path)
        all_X.append(Xp); all_y.append(yp); all_sources.extend(sp)

    Xv3, yv3, sv3 = load_features_v3_normal_only("features/indonesia_v3/metadata.json")
    all_X.append(Xv3); all_y.append(yv3); all_sources.extend(sv3)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    sources = all_sources

    print(f"\nTotal clean dataset: {len(X)} segments")
    print(f"  Normal: {Counter(y)[0]} | Anomaly: {Counter(y)[1]}")

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sources, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model, history = train_model(X_train, y_train, X_val, y_val)

    model_path = "models/mil_model_v2_clean.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["val_auc"], label="Val AUC", color="green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(history["lr"], label="LR", color="red")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR"); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig("models/training_history_v2_clean.png")
    print(f"History saved: models/training_history_v2_clean.png")

    results = evaluate(model, X_test, y_test, "CLEAN model - Combined Test")

    print("\n\n--- Per-Source Evaluation on CLEAN model ---")
    for src in set(s_test):
        mask = [st == src for st in s_test]
        Xs, ys = X_test[mask], y_test[mask]
        if len(set(ys)) > 1 and len(Xs) > 10:
            evaluate(model, Xs, ys, f"CLEAN model - {src}")

    # Diagnostic: test clean model on the contaminated news clips
    print("\n\n--- DIAGNOSTIC: Clean model on Indonesia v3 anomaly (news clips) ---")
    old_scores = evaluate(old_model, X_diag, y_diag, "OLD model on news clips (for comparison)")
    new_scores = evaluate(model, X_diag, y_diag, "CLEAN model on news clips")

    print(f"\n\nKEY COMPARISON:")
    print(f"  Old model score on news clips: {old_scores['roc_auc']:.4f} AUC")
    print(f"  Clean model score on news clips: {new_scores['roc_auc']:.4f} AUC")
    print(f"  If clean model scores MUCH LOWER, it confirms old model learned 'news=anomaly'")

    results_path = "models/evaluation_results_v2_clean.json"
    with open(results_path, "w") as f:
        json.dump({"overall": results, "old_model_on_news": old_scores, "clean_model_on_news": new_scores}, f, indent=2)
    print(f"\nResults saved: {results_path}")
