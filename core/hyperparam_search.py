"""
Hyperparameter search for MIL models.
Tests both frame-level (MILRankingModel) and video-level (AttentionMILModel) architectures.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import itertools
import time
import gc

from core.mil_ranking import MILRankingModel
from core.mil_attention import AttentionMILModel, VideoDataset

FEAT_DIR = Path("features/final_dataset")
META_PATH = FEAT_DIR / "metadata.json"
RESULTS_DIR = Path("runs/hyperparam_search")
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 10

# Hyperparameter grid
GRID = {
    "lr": [0.0003, 0.0005, 0.001],
    "hidden_units": [256, 512],
    "dropout": [0.3, 0.5],
    "weight_decay": [1e-4, 1e-5],
}


def load_metadata():
    with open(META_PATH) as f:
        data = json.load(f)
    return data


def prepare_video_data(metadata):
    video_data = defaultdict(list)
    for item in metadata:
        video_data[item["path"]].append({
            "path": item["path"],
            "label": item["label"],
            "split": item["split"],
            "label_name": item["label_name"],
        })

    video_entries = list(video_data.values())
    result = {"train": [], "val": [], "test": []}
    for entries in video_entries:
        e = entries[0]
        feat = np.load(e["path"])
        result[e["split"]].append({
            "features": feat,
            "label": e["label"],
            "segments": feat.shape[0],
        })
    return result


def train_frame_level(X_train, y_train, X_val, y_val, config):
    device = torch.device("cpu")
    input_dim = X_train.shape[1]

    model = MILRankingModel(input_dim=input_dim, hidden_units=config["hidden_units"]).to(device)

    class_counts = Counter(y_train)
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model.forward_raw(batch_X)
            loss = criterion(logits, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                logits = model.forward_raw(batch_X)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_val_auc


def train_video_level(train_data, val_data, config):
    device = torch.device("cpu")
    input_dim = train_data[0]["features"].shape[1]

    model = AttentionMILModel(input_dim=input_dim, hidden_units=config["hidden_units"],
                              dropout=config["dropout"]).to(device)

    # Build video-level datasets
    train_videos = []
    train_labels = []
    for v in train_data:
        train_videos.append(v["features"])
        train_labels.append(v["label"])

    val_videos = []
    val_labels = []
    for v in val_data:
        val_videos.append(v["features"])
        val_labels.append(v["label"])

    train_dataset = VideoDataset(train_videos, train_labels)
    val_dataset = VideoDataset(val_videos, val_labels)

    def collate_video(batch):
        feats, labs = zip(*batch)
        return list(feats), torch.stack(labs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_video)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_video)

    class_counts = Counter(train_labels)
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_feats, batch_labels in train_loader:
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            # Pad sequences to same length within batch
            max_len = max(f.shape[0] for f in batch_feats)
            padded = []
            for f in batch_feats:
                if f.shape[0] < max_len:
                    pad = torch.zeros(max_len - f.shape[0], f.shape[1])
                    padded.append(torch.cat([torch.FloatTensor(f), pad], dim=0))
                else:
                    padded.append(torch.FloatTensor(f))
            batch_X = torch.stack(padded).to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_feats, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                max_len = max(f.shape[0] for f in batch_feats)
                padded = []
                for f in batch_feats:
                    if f.shape[0] < max_len:
                        pad = torch.zeros(max_len - f.shape[0], f.shape[1])
                        padded.append(torch.cat([torch.FloatTensor(f), pad], dim=0))
                    else:
                        padded.append(torch.FloatTensor(f))
                batch_X = torch.stack(padded).to(device)

                logits = model(batch_X)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_val_auc


def main():
    metadata = load_metadata()
    video_data = prepare_video_data(metadata)

    train_data = video_data["train"]
    val_data = video_data["val"]
    test_data = video_data["test"]

    # Frame-level data
    X_train_frames = np.concatenate([v["features"] for v in train_data])
    y_train_frames = np.concatenate([np.full(v["features"].shape[0], v["label"]) for v in train_data])
    X_val_frames = np.concatenate([v["features"] for v in val_data])
    y_val_frames = np.concatenate([np.full(v["features"].shape[0], v["label"]) for v in val_data])

    print(f"Frame-level: train={len(X_train_frames)} val={len(X_val_frames)}")
    print(f"Video-level: train={len(train_data)} val={len(val_data)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"\nTotal configs to test: {len(combos)} x 2 models = {len(combos)*2} runs\n")

    for i, values in enumerate(combos):
        config = dict(zip(keys, values))
        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(combos)}: {config}")

        # Frame-level model
        t0 = time.time()
        try:
            frame_auc = train_frame_level(X_train_frames, y_train_frames, X_val_frames, y_val_frames, config)
            print(f"  Frame-level val_auc: {frame_auc:.4f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  Frame-level FAILED: {e}")
            frame_auc = 0.0
        gc.collect()

        # Video-level model
        t0 = time.time()
        try:
            video_auc = train_video_level(train_data, val_data, config)
            print(f"  Video-level val_auc: {video_auc:.4f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  Video-level FAILED: {e}")
            video_auc = 0.0
        gc.collect()

        results.append({
            **config,
            "frame_val_auc": round(frame_auc, 4),
            "video_val_auc": round(video_auc, 4),
        })

        # Save incremental results
        with open(RESULTS_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Print summary sorted by best video-level AUC
    print(f"\n{'='*60}")
    print("TOP 10 CONFIGURATIONS (by video-level AUC)")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r["video_val_auc"], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1}. lr={r['lr']} hidden={r['hidden_units']} drop={r['dropout']} wd={r['weight_decay']} "
              f"| frame_auc={r['frame_val_auc']:.4f} | video_auc={r['video_val_auc']:.4f}")

    print(f"\nTOP 10 (by frame-level AUC)")
    sorted_frame = sorted(results, key=lambda r: r["frame_val_auc"], reverse=True)
    for i, r in enumerate(sorted_frame[:10]):
        print(f"  {i+1}. lr={r['lr']} hidden={r['hidden_units']} drop={r['dropout']} wd={r['weight_decay']} "
              f"| frame_auc={r['frame_val_auc']:.4f} | video_auc={r['video_val_auc']:.4f}")

    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
