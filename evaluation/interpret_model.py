"""
Model Interpretation: Attention weights, feature ablation, per-frame analysis.
UAS ML - AttentionMIL interpretability.
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

from core.mil_attention import AttentionMILModel

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})

OUTPUT_DIR = Path("reports/interpretation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = Path("features/final_dataset")
META_PATH = FEAT_DIR / "metadata.json"
MODEL_PATH = "models/mil_final.pt"
DEVICE = "cpu"


class AttentionMILWithGates(AttentionMILModel):
    """Wrapper that returns attention weights for interpretation."""
    def forward_with_attention(self, x):
        batch_size, n_segments, feat_dim = x.shape
        x_flat = x.view(-1, feat_dim)
        att_scores = self.attention(x_flat)
        att_weights = att_scores.view(batch_size, n_segments)
        att_weights = torch.softmax(att_weights, dim=1)
        bag_feat = torch.sum(x * att_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(bag_feat)
        return logits.squeeze(-1), att_weights.squeeze(0)


def load_model():
    model = AttentionMILWithGates(1024, 256, 0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def demo_attention_heatmap(model):
    """Show attention weights for sample videos from each class."""
    with open(META_PATH) as f:
        meta = json.load(f)
    test_items = [m for m in meta if m["split"] == "test"]

    # Pick examples: 3 normal, 3 rusuh
    normal_examples = [m for m in test_items if m["label"] == 0][:3]
    rusuh_examples = [m for m in test_items if m["label"] == 1][:3]
    examples = normal_examples + rusuh_examples

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (ax, item) in enumerate(zip(axes, examples)):
        feat = np.load(item["path"])
        label = item["label"]
        n_seg = min(16, feat.shape[0])
        feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)

        with torch.no_grad():
            logit, attn = model.forward_with_attention(feat_t)

        score = torch.sigmoid(logit).item()
        pred = "RUSUH" if score >= 0.5 else "NORMAL"

        ax.bar(range(n_seg), attn.numpy(), color="steelblue", alpha=0.8)
        ax.set_xlabel("Segment Index", fontsize=9)
        ax.set_ylabel("Attention Weight", fontsize=9)
        ax.set_title(f"{'RUSUH' if label else 'NORMAL'} | Pred: {pred} ({score:.3f})",
                     fontsize=10, fontweight="bold",
                     color="red" if label else "green")
        ax.set_xticks(range(n_seg))
        ax.set_xticklabels([str(i) for i in range(n_seg)], fontsize=7)
        ax.set_ylim(0, 1)

    plt.suptitle("Attention Weights per Video Segment\n(Bars = which segments matter most)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attention_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Attention heatmap: {OUTPUT_DIR / 'attention_weights.png'}")


def feature_ablation_analysis(model):
    """Simulate feature importance by ablating each segment and measuring score change."""
    with open(META_PATH) as f:
        meta = json.load(f)
    test_items = [m for m in meta if m["split"] == "test"]

    # Sample 100 test items for ablation
    np.random.seed(42)
    sample_items = np.random.choice(test_items, min(100, len(test_items)), replace=False)

    n_segments = 16
    impact_scores = np.zeros(n_segments)

    for item in sample_items:
        feat = np.load(item["path"])
        label = item["label"]
        n_seg = min(n_segments, feat.shape[0])

        feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)
        with torch.no_grad():
            logit_full, attn = model.forward_with_attention(feat_t)
        base_score = torch.sigmoid(logit_full).item()

        for i in range(n_seg):
            feat_abl = feat_t.clone()
            feat_abl[0, i, :] = 0  # zero out this segment
            with torch.no_grad():
                logit_abl, _ = model.forward_with_attention(feat_abl)
            ablated_score = torch.sigmoid(logit_abl).item()
            impact_scores[i] += abs(base_score - ablated_score)

    impact_scores /= len(sample_items)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["crimson" if s > np.mean(impact_scores) else "steelblue" for s in impact_scores]
    bars = ax.bar(range(n_segments), impact_scores, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(np.mean(impact_scores), color="gray", ls="--", lw=1.5,
               label=f"Mean impact = {np.mean(impact_scores):.4f}")
    ax.set_xlabel("Segment Index"); ax.set_ylabel("Avg Score Change")
    ax.set_title("Feature Ablation: Impact of Removing Each Segment on Score",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_segments))
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Ablation analysis: {OUTPUT_DIR / 'feature_ablation.png'}")

    return impact_scores


def per_video_breakdown(model):
    """Show per-video score evolution across segments."""
    with open(META_PATH) as f:
        meta = json.load(f)
    test_items = [m for m in meta if m["split"] == "test"]

    examples = [
        (next(m for m in test_items if m["label"] == 0), "green"),
        (next(m for m in test_items if m["label"] == 1), "red"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (item, color) in zip(axes, examples):
        feat = np.load(item["path"])
        n_seg = min(16, feat.shape[0])

        cumulative = []
        for i in range(1, n_seg + 1):
            feat_t = torch.FloatTensor(feat[:i]).unsqueeze(0)
            with torch.no_grad():
                if i < n_seg:
                    feat_padded = torch.zeros(1, 16, 1024)
                    feat_padded[0, :i, :] = feat_t
                    logit, _ = model.forward_with_attention(feat_padded)
                else:
                    logit, _ = model.forward_with_attention(feat_t)
            cumulative.append(torch.sigmoid(logit).item())

        ax.plot(range(1, n_seg + 1), cumulative, "o-", color=color, lw=2, markersize=8)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.7, label="threshold")
        ax.set_xlabel("Number of Segments Processed")
        ax.set_ylabel("Anomaly Score")
        ax.set_title(f"{'RUSUH' if item['label'] else 'NORMAL'} Video\nFinal: {cumulative[-1]:.3f}",
                     fontweight="bold", color=color)
        ax.set_ylim(-0.05, 1.05); ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Video Score Evolution (Cumulative)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_video_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Per-video breakdown: {OUTPUT_DIR / 'per_video_evolution.png'}")


def score_distribution_by_segments(model):
    """Show how the score changes as we add more segments."""
    with open(META_PATH) as f:
        meta = json.load(f)
    test_items = [m for m in meta if m["split"] == "test"]

    n_test = min(200, len(test_items))
    all_normal = [m for m in test_items if m["label"] == 0][:n_test//2]
    all_rusuh = [m for m in test_items if m["label"] == 1][:n_test//2]

    n_segments = 16
    normal_scores = np.zeros((len(all_normal), n_segments))
    rusuh_scores = np.zeros((len(all_rusuh), n_segments))
    normal_labels = np.ones(len(all_normal)) * 0
    rusuh_labels = np.ones(len(all_rusuh)) * 1

    for items, scores, labels in [(all_normal, normal_scores, normal_labels)]:
        # we need to iterate and fill
        pass

    # Actually simpler: iterate directly
    for i, item in enumerate(all_normal):
        feat = np.load(item["path"])
        n_seg = min(n_segments, feat.shape[0])
        for j in range(n_seg):
            feat_t = torch.FloatTensor(feat[:j+1]).unsqueeze(0)
            if j+1 < n_segments:
                feat_padded = torch.zeros(1, n_segments, 1024)
                feat_padded[0, :j+1, :] = feat_t
                logit, _ = model.forward_with_attention(feat_padded)
            else:
                logit, _ = model.forward_with_attention(feat_t)
            normal_scores[i, j] = torch.sigmoid(logit).item()

    for i, item in enumerate(all_rusuh):
        feat = np.load(item["path"])
        n_seg = min(n_segments, feat.shape[0])
        for j in range(n_seg):
            feat_t = torch.FloatTensor(feat[:j+1]).unsqueeze(0)
            if j+1 < n_segments:
                feat_padded = torch.zeros(1, n_segments, 1024)
                feat_padded[0, :j+1, :] = feat_t
                logit, _ = model.forward_with_attention(feat_padded)
            else:
                logit, _ = model.forward_with_attention(feat_t)
            rusuh_scores[i, j] = torch.sigmoid(logit).item()

    fig, ax = plt.subplots(figsize=(10, 6))

    segments = range(1, n_segments + 1)
    ax.errorbar(segments, normal_scores.mean(axis=0),
                yerr=normal_scores.std(axis=0), color="green",
                label="Normal/Damai", lw=2, marker="o", capsize=3)
    ax.errorbar(segments, rusuh_scores.mean(axis=0),
                yerr=rusuh_scores.std(axis=0), color="red",
                label="Rusuh", lw=2, marker="s", capsize=3)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.7, label="threshold")
    ax.set_xlabel("Number of Segments"); ax.set_ylabel("Mean Anomaly Score")
    ax.set_title("Score Convergence by Number of Segments", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Score convergence: {OUTPUT_DIR / 'score_convergence.png'}")

    return normal_scores, rusuh_scores


def main():
    print("=" * 60)
    print("MODEL INTERPRETATION - UAS Machine Learning")
    print("=" * 60)

    model = load_model()
    print(f"Model: AttentionMIL + AttentionWeights")

    # 1. Attention Weight Visualization
    print("\n--- Attention Weight Analysis ---")
    demo_attention_heatmap(model)

    # 2. Feature Ablation
    print("\n--- Feature Ablation Analysis ---")
    feature_ablation_analysis(model)

    # 3. Per-video score evolution
    print("\n--- Per-Video Breakdown ---")
    per_video_breakdown(model)

    # 4. Score convergence
    print("\n--- Score Convergence Analysis ---")
    score_distribution_by_segments(model)

    print(f"\n{'='*60}")
    print(f"All interpretation results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
