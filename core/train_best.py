"""
Train best MIL model with optimal hyperparams from grid search.
Tests both frame-level (MILRankingModel) and video-level (AttentionMILModel).
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
from collections import Counter, defaultdict

from core.mil_ranking import MILRankingModel
from core.mil_attention import AttentionMILModel, VideoDataset

FEAT_DIR = Path("features/final_dataset")
META_PATH = FEAT_DIR / "metadata.json"
OUTPUT_DIR = Path("models")
PLOT_DIR = Path("runs/best_training")
BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 20

# Best config from grid search
BEST_CONFIG = {
    "lr": 0.001,
    "hidden_units": 256,
    "dropout": 0.3,
    "weight_decay": 1e-5,
}


def load_metadata():
    with open(META_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} videos from metadata")
    return data


def prepare_frame_data(metadata):
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


def prepare_video_data(metadata):
    train_videos, val_videos, test_videos = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    seen = set()
    for item in metadata:
        if item["path"] in seen:
            continue
        seen.add(item["path"])
        feat = np.load(item["path"])
        split = item["split"]
        label = item["label"]

        if split == "train":
            train_videos.append(feat)
            train_labels.append(label)
        elif split == "val":
            val_videos.append(feat)
            val_labels.append(label)
        elif split == "test":
            test_videos.append(feat)
            test_labels.append(label)

    print(f"\nVideos: train={len(train_videos)} val={len(val_videos)} test={len(test_videos)}")
    for name, labs in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        cnt = Counter(labs)
        print(f"  {name}: normal={cnt[0]} rusuh={cnt[1]}")
    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels


def train_frame(model_name):
    print(f"\n{'='*60}")
    print(f"Training FRAME-LEVEL model: {model_name}")
    print(f"Config: {BEST_CONFIG}")
    print(f"{'='*60}")

    metadata = load_metadata()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_frame_data(metadata)
    device = torch.device("cpu")
    input_dim = X_train.shape[1]

    model = MILRankingModel(input_dim=input_dim, hidden_units=BEST_CONFIG["hidden_units"]).to(device)

    class_counts = Counter(y_train)
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)]).to(device)
    print(f"  pos_weight={pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=BEST_CONFIG["lr"],
                            weight_decay=BEST_CONFIG["weight_decay"])

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

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"mil_{model_name}.pt")
            print(f"  -> Saved best")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(OUTPUT_DIR / f"mil_{model_name}.pt", map_location=device, weights_only=True))
    print(f"\nBest epoch {best_epoch} (val_auc={best_val_auc:.4f})")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_auc"])
    axes[1].set_title("Val AUC"); axes[1].axhline(best_val_auc, color="r", ls="--")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"history_{model_name}.png", dpi=150)

    evaluate(model, X_test, y_test, f"{model_name}_test")
    evaluate(model, X_val, y_val, f"{model_name}_val")

    return model, history


def train_video(model_name):
    print(f"\n{'='*60}")
    print(f"Training VIDEO-LEVEL model: {model_name}")
    print(f"Config: {BEST_CONFIG}")
    print(f"{'='*60}")

    metadata = load_metadata()
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels = prepare_video_data(metadata)
    device = torch.device("cpu")
    input_dim = train_videos[0].shape[1]

    model = AttentionMILModel(input_dim=input_dim, hidden_units=BEST_CONFIG["hidden_units"],
                              dropout=BEST_CONFIG["dropout"]).to(device)

    class_counts = Counter(train_labels)
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)]).to(device)
    print(f"  pos_weight={pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=BEST_CONFIG["lr"],
                            weight_decay=BEST_CONFIG["weight_decay"])

    train_dataset = VideoDataset(train_videos, train_labels)
    val_dataset = VideoDataset(val_videos, val_labels)

    def collate_fn(batch):
        feats, labels = zip(*batch)
        return list(feats), torch.tensor(labels, dtype=torch.float32)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch_feats, batch_labels in train_loader:
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            max_len = max(f.shape[0] for f in batch_feats)
            padded = []
            for f in batch_feats:
                f_t = torch.FloatTensor(f)
                if f_t.shape[0] < max_len:
                    pad = torch.zeros(max_len - f_t.shape[0], f_t.shape[1])
                    padded.append(torch.cat([f_t, pad], dim=0))
                else:
                    padded.append(f_t)
            batch_X = torch.stack(padded).to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_labels)
            n_train += len(batch_labels)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_feats, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                max_len = max(f.shape[0] for f in batch_feats)
                padded = []
                for f in batch_feats:
                    f_t = torch.FloatTensor(f)
                    if f_t.shape[0] < max_len:
                        pad = torch.zeros(max_len - f_t.shape[0], f_t.shape[1])
                        padded.append(torch.cat([f_t, pad], dim=0))
                    else:
                        padded.append(f_t)
                batch_X = torch.stack(padded).to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * len(batch_labels)
                n_val += len(batch_labels)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else preds)
                all_labels.extend(batch_labels.cpu().numpy().flatten().tolist())
        val_loss /= n_val
        unique_labels = set(all_labels)
        val_auc = roc_auc_score(all_labels, all_preds) if len(unique_labels) > 1 else 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} ({n_train} train videos)", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"mil_{model_name}.pt")
            print(f"  -> Saved best")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(OUTPUT_DIR / f"mil_{model_name}.pt", map_location=device, weights_only=True))
    print(f"\nBest epoch {best_epoch} (val_auc={best_val_auc:.4f})")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_auc"])
    axes[1].set_title("Val AUC"); axes[1].axhline(best_val_auc, color="r", ls="--")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"history_{model_name}.png", dpi=150)

    evaluate_video(model, test_videos, test_labels, f"{model_name}_test")
    evaluate_video(model, val_videos, val_labels, f"{model_name}_val")

    return model, history


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

    print(f"\n--- {label.upper()} ---")
    print(f"  AUC: {auc:.4f}")
    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    return auc, precision, recall, f1, best_thresh


def evaluate_video(model, test_videos, test_labels, label="test"):
    device = torch.device("cpu")
    model.eval()
    all_preds = []
    for feat, true_label in zip(test_videos, test_labels):
        f_t = torch.FloatTensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(f_t)
            score = torch.sigmoid(logits).item()
        all_preds.append(score)

    all_preds = np.array(all_preds)
    all_labels = np.array(test_labels)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels.tolist())) > 1 else 0.5

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

    print(f"\n--- {label.upper()} (VIDEO-LEVEL) ---")
    print(f"  AUC: {auc:.4f}")
    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    return auc, precision, recall, f1, best_thresh


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Train frame-level model (best config)
    train_frame("best_frame")

    # Train video-level attention model
    train_video("best_video")


if __name__ == "__main__":
    main()
