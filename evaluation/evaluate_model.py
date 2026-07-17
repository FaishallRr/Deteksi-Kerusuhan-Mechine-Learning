"""
Comprehensive model evaluation for UAS ML project.
Evaluates AttentionMIL model on test set + real video files.
"""

import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, precision_recall_curve)

from core.mil_attention import AttentionMILModel

OUTPUT_DIR = Path("reports/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = Path("features/final_dataset")
META_PATH = FEAT_DIR / "metadata.json"

MODEL_PATH = "models/mil_final.pt"
DEVICE = "cpu"

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})


def load_model():
    model = AttentionMILModel(input_dim=1024, hidden_units=256, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def evaluate_on_features(model):
    """Evaluate on pre-extracted test set features (fast)."""
    with open(META_PATH) as f:
        meta = json.load(f)

    test_items = [m for m in meta if m["split"] == "test"]
    print(f"Test set: {len(test_items)} videos")

    y_true, y_pred, y_score = [], [], []
    video_names = []

    for item in test_items:
        feat = np.load(item["path"])
        label = item["label"]
        n_seg = min(16, feat.shape[0])
        feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(model(feat_t)).item()

        y_true.append(label)
        y_score.append(score)
        y_pred.append(1 if score >= 0.5 else 0)
        video_names.append(Path(item["path"]).name)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.array(y_pred)

    # Metrics
    auc = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION ({len(test_items)} videos)")
    print(f"{'='*60}")
    print(f"AUC:         {auc:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  Rusuh")
    print(f"Actual Normal  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Rusuh   {cm[1,0]:4d}   {cm[1,1]:4d}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - AttentionMIL Model")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print(f"\nROC curve saved: {OUTPUT_DIR / 'roc_curve.png'}")

    # Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_vals, prec_vals, "r-", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pr_curve.png", dpi=150)
    plt.close()
    print(f"PR curve saved: {OUTPUT_DIR / 'pr_curve.png'}")

    # Confusion Matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Rusuh"],
                yticklabels=["Normal", "Rusuh"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {OUTPUT_DIR / 'confusion_matrix.png'}")

    # Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, color, title in [
        (axes[0], 0, "green", "Normal Videos"),
        (axes[1], 1, "red", "Rusuh Videos"),
    ]:
        scores = y_score[y_true == label]
        ax.hist(scores, bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.axvline(0.5, color="gray", ls="--", label="threshold=0.5")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=150)
    plt.close()
    print(f"Score distribution saved: {OUTPUT_DIR / 'score_distribution.png'}")

    # Top false positives / false negatives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

    print(f"\nFalse Positives: {len(fp_indices)}")
    if len(fp_indices) > 0:
        print("  Top FPs (highest score):")
        fp_scores = [(i, y_score[i], video_names[i]) for i in fp_indices]
        fp_scores.sort(key=lambda x: -x[1])
        for i, s, name in fp_scores[:5]:
            print(f"    {name}: score={s:.4f}")

    print(f"\nFalse Negatives: {len(fn_indices)}")
    if len(fn_indices) > 0:
        print("  Top FNs (lowest score):")
        fn_scores = [(i, y_score[i], video_names[i]) for i in fn_indices]
        fn_scores.sort(key=lambda x: x[1])
        for i, s, name in fn_scores[:5]:
            print(f"    {name}: score={s:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    target_names = ["Normal/Damai", "Rusuh"]
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1, "cm": cm,
            "fp": len(fp_indices), "fn": len(fn_indices)}


def evaluate_videos(model):
    """Evaluate on actual video files to show real-world performance."""
    from preprocessing.feature_extractor import TemporalFeatureExtractor

    extractor = TemporalFeatureExtractor(architecture="s3d", device=DEVICE)

    test_videos = [
        ("Rusuh (tawuran)", "test_videos/fight_sample_1.mp4", 1),
        ("Rusuh (tawuran grogol)", "test_videos/tawuran_grogol.mp4", 1),
        ("Damai (digulis)", "test_videos/digulis_detected.mp4", 0),
        ("Damai (flamboyan)", "test_videos/flamboyan_detected.mp4", 0),
    ]

    results = []
    for label, vpath, true_label in test_videos:
        if not Path(vpath).exists():
            print(f"  Skip (not found): {vpath}")
            continue

        cap = cv2.VideoCapture(vpath)
        frames = []
        feat_buffer = []
        scores = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (640, 640))
            frames.append(resized)
            if len(frames) >= 16:
                seg = extractor.extract(frames[-16:])
                feat_buffer.append(seg.squeeze())
                min_seg = min(16, len(feat_buffer))
                if min_seg >= 4:
                    bag = torch.FloatTensor(np.array(feat_buffer[-min_seg:])).unsqueeze(0)
                    with torch.no_grad():
                        score = torch.sigmoid(model(bag)).item()
                    scores.append(score)
                frames = frames[-8:]
        cap.release()

        avg_score = np.mean(scores) if scores else 0
        max_score = np.max(scores) if scores else 0
        pct_rusuh = sum(1 for s in scores if s >= 0.5) / max(len(scores), 1) * 100

        verdict = "CORRECT" if (
            (true_label == 1 and avg_score >= 0.5) or
            (true_label == 0 and avg_score < 0.5)
        ) else "WRONG"

        results.append((label, vpath, avg_score, max_score, pct_rusuh, verdict))

    import cv2 as cv2_module
    print(f"\n{'='*60}")
    print("REAL VIDEO EVALUATION")
    print(f"{'='*60}")
    print(f"{'Label':25s} | {'AvgScore':>8s} | {'MaxScore':>8s} | {'%Rusuh':>6s} | {'Verdict':>8s}")
    print("-" * 65)
    for label, vpath, avg_s, max_s, pct_r, verdict in results:
        print(f"{label:25s} | {avg_s:.4f} | {max_s:.4f} | {pct_r:5.1f}% | {verdict:>8s}")

    return results


def main():
    print("=" * 60)
    print("MODEL EVALUATION - UAS Machine Learning")
    print("=" * 60)

    model = load_model()
    print(f"Model: AttentionMIL (hidden=256, dropout=0.3)")
    print(f"Weights: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # Phase 1: Feature-level evaluation (fast, comprehensive)
    metrics = evaluate_on_features(model)

    # Phase 2: Video-level evaluation (real videos)
    evaluate_videos(model)

    print(f"\n{'='*60}")
    print("All evaluation results saved to:", OUTPUT_DIR)
    print(f"{'='*60}")


if __name__ == "__main__":
    import cv2  # noqa
    main()
