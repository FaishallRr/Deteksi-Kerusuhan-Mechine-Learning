"""
Fast model evaluation using pre-extracted test features.
Generates all plots and metrics for UAS reporting.
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
                             classification_report, precision_recall_curve,
                             matthews_corrcoef)
from core.mil_attention import AttentionMILModel

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

OUTPUT_DIR = Path("reports/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = Path("features/final_dataset")
META_PATH = FEAT_DIR / "metadata.json"
MODEL_PATH = "models/mil_final.pt"
DEVICE = "cpu"

LABEL_MAP = {0: "Normal/Damai", 1: "Rusuh"}


def load_model():
    model = AttentionMILModel(input_dim=1024, hidden_units=256, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def main():
    model = load_model()
    print("Model loaded: AttentionMIL (input_dim=1024, hidden=256)")

    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"Total samples: {len(meta)}")

    # Split stats
    splits = Counter(m["split"] for m in meta)
    labels = Counter(m["label"] for m in meta)
    print(f"Splits: train={splits.get('train',0)}, val={splits.get('val',0)}, test={splits.get('test',0)}")
    print(f"Labels: normal={labels.get(0,0)}, rusuh={labels.get(1,0)}")

    test_items = [m for m in meta if m["split"] == "test"]
    print(f"\nTest set: {len(test_items)} videos")

    y_true, y_score, video_names = [], [], []

    for i, item in enumerate(test_items):
        feat = np.load(item["path"])
        label = item["label"]
        n_seg = min(16, feat.shape[0])
        feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(model(feat_t)).item()
        y_true.append(label)
        y_score.append(score)
        video_names.append(Path(item["path"]).stem)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    # ====== Metrics ======
    auc = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = np.mean(y_true == y_pred)

    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS (Test Set)")
    print(f"{'='*60}")
    print(f"Accuracy:            {acc:.4f}")
    print(f"AUC:                 {auc:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"MCC:                 {mcc:.4f}")
    print()

    print(f"Confusion Matrix:")
    print(f"              Normal/Damai  Rusuh")
    print(f"Normal/Damai   {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"Rusuh          {cm[1,0]:4d}        {cm[1,1]:4d}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Normal/Damai", "Rusuh"],
                                zero_division=0))

    # Misclassifications
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    print(f"False Positives (Normal->Rusuh): {len(fp)}")
    print(f"False Negatives (Rusuh->Normal): {len(fn)}")

    # ====== Plots ======

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve - AttentionMIL\n(AUC = {auc:.4f})", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] ROC curve: {OUTPUT_DIR / 'roc_curve.png'}")

    # 2. Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_vals, prec_vals, "r-", lw=2.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - AttentionMIL", fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] PR curve: {OUTPUT_DIR / 'pr_curve.png'}")

    # 3. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal\n(Damai)", "Rusuh"],
                yticklabels=["Normal\n(Damai)", "Rusuh"],
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")

    # 4. Score Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, color, title in [
        (axes[0], 0, "green", "Normal/Damai Videos (n={})".format((y_true==0).sum())),
        (axes[1], 1, "red", "Rusuh Videos (n={})".format((y_true==1).sum())),
    ]:
        scores = y_score[y_true == label]
        ax.hist(scores, bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0.5, color="gray", ls="--", lw=2, label="threshold = 0.5")
        ax.set_xlabel("Anomaly Score (Rusuh probability)")
        ax.set_ylabel("Frequency")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
    plt.suptitle("Anomaly Score Distribution by Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Score distribution: {OUTPUT_DIR / 'score_distribution.png'}")

    # 5. Score Distribution (overlay)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    ax.hist(y_score[y_true==0], bins=bins, alpha=0.6, color="green", label="Normal/Damai", density=True)
    ax.hist(y_score[y_true==1], bins=bins, alpha=0.6, color="red", label="Rusuh", density=True)
    ax.axvline(0.5, color="gray", ls="--", lw=2, label="threshold = 0.5")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution Overlay", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Score overlay: {OUTPUT_DIR / 'score_overlay.png'}")

    # 6. ROC + Confusion Matrix combined (for report)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # ROC
    axes[0].plot(fpr, tpr, "b-", lw=2.5, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="blue")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right"); axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1])
    # CM
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Rusuh"],
                yticklabels=["Normal", "Rusuh"], ax=axes[1])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix", fontweight="bold")
    plt.suptitle("AttentionMIL Model Evaluation - Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Evaluation summary: {OUTPUT_DIR / 'evaluation_summary.png'}")

    # Save metrics to JSON
    results = {
        "model": "AttentionMIL",
        "input_dim": 1024,
        "hidden_units": 256,
        "test_samples": int(len(test_items)),
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "mcc": float(mcc),
        "false_positives": int(len(fp)),
        "false_negatives": int(len(fn)),
        "cm": cm.tolist(),
        "normal_test_count": int((y_true==0).sum()),
        "rusuh_test_count": int((y_true==1).sum()),
        "splits": {k: int(v) for k, v in splits.items()},
        "label_distribution": {str(k): int(v) for k, v in labels.items()},
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Metrics JSON: {OUTPUT_DIR / 'metrics.json'}")

    print(f"\n{'='*60}")
    print("All done! Results saved to:", OUTPUT_DIR)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
