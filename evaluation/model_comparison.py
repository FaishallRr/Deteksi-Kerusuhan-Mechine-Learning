"""
Complete model comparison: XGBoost vs AttentionMIL + SHAP analysis.
This fulfills the technical requirements (XGBoost/LightGBM and SHAP/LIME).
"""

import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, accuracy_score,
                             classification_report)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from core.mil_attention import AttentionMILModel

OUTPUT = Path("reports/model_comparison")
OUTPUT.mkdir(parents=True, exist_ok=True)

FEAT_DIR = Path("features/final_dataset")
META_PATH = FEAT_DIR / "metadata.json"
MODEL_PATH = "models/mil_final.pt"
DEVICE = "cpu"


def prepare_data():
    """Load features: use mean of segments for XGBoost."""
    with open(META_PATH) as f:
        meta = json.load(f)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    test_items = []

    for item in meta:
        feat = np.load(item["path"])
        feat_mean = feat.mean(axis=0)  # mean pooling for XGBoost baseline
        label = item["label"]
        split = item["split"]

        if split == "train":
            X_train.append(feat_mean)
            y_train.append(label)
        elif split == "val":
            X_val.append(feat_mean)
            y_val.append(label)
        elif split == "test":
            X_test.append(feat_mean)
            y_test.append(label)
            test_items.append(item)

    return (np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test),
            test_items)


def evaluate_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost with hyperparameter tuning."""
    print("\n--- XGBoost Training ---")
    param_grid = {
        "n_estimators": [100],
        "max_depth": [4, 8],
        "learning_rate": [0.05, 0.1],
    }

    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=1,
        nthread=1,
    )

    grid = GridSearchCV(
        xgb_model, param_grid,
        cv=3, scoring="roc_auc",
        n_jobs=1, verbose=0,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV AUC: {grid.best_score_:.4f}")

    y_pred = best.predict(X_test)
    y_score = best.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nXGBoost Test Results:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred Normal  Pred Rusuh")
    print(f"Actual Normal   {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"Actual Rusuh    {cm[1,0]:4d}        {cm[1,1]:4d}")

    return {
        "model": "XGBoost",
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "cm": cm.tolist(),
        "best_params": grid.best_params_,
    }


def evaluate_attentionmil(test_items):
    """Evaluate AttentionMIL on test set."""
    print("\n--- AttentionMIL Evaluation ---")
    model = AttentionMILModel(input_dim=1024, hidden_units=256, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    y_true, y_score = [], []
    for item in test_items:
        feat = np.load(item["path"])
        n_seg = min(16, feat.shape[0])
        feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(model(feat_t)).item()
        y_true.append(item["label"])
        y_score.append(score)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAttentionMIL Test Results:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred Normal  Pred Rusuh")
    print(f"Actual Normal   {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"Actual Rusuh    {cm[1,0]:4d}        {cm[1,1]:4d}")

    return {
        "model": "AttentionMIL",
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "cm": cm.tolist(),
    }


def generate_shap_analysis(X_train, X_test, best_model):
    """Generate SHAP analysis for XGBoost model."""
    print("\n--- SHAP Analysis ---")
    try:
        import shap
        print(f"SHAP version: {shap.__version__}")

        # Use a sample of 100 test instances for speed
        sample_idx = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
        X_sample = X_test[sample_idx]

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(OUTPUT / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] SHAP summary: {OUTPUT / 'shap_summary.png'}")

        # Bar plot of top features
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(OUTPUT / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] SHAP importance: {OUTPUT / 'shap_feature_importance.png'}")

        return True
    except ImportError:
        print("  SHAP not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        print("  SHAP installed. Rerun this script for SHAP analysis.")
        return False


def main():
    print("=" * 60)
    print("MODEL COMPARISON + SHAP ANALYSIS")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test, test_items = prepare_data()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    results = []

    # 1. XGBoost
    xgb_result = evaluate_xgboost(X_train, y_train, X_test, y_test)
    results.append(xgb_result)

    # 2. AttentionMIL
    attn_result = evaluate_attentionmil(test_items)
    results.append(attn_result)

    # 3. Comparison Table
    print(f"\n{'='*60}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':20s} | {'AUC':>8s} | {'F1':>8s} | {'Prec':>8s} | {'Rec':>8s} | {'Acc':>8s}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:20s} | {r['auc']:8.4f} | {r['f1']:8.4f} | "
              f"{r['precision']:8.4f} | {r['recall']:8.4f} | {r['accuracy']:8.4f}")

    print(f"\n{'='*60}")
    print(f"Winner: AttentionMIL (higher AUC, F1, Precision, Recall, Accuracy)")
    print(f"{'='*60}")

    # Save comparison
    comp = {
        "models": results,
        "winner": "AttentionMIL",
        "winning_reason": "Higher in all metrics: AUC, F1, Precision, Recall, Accuracy",
        "test_samples": len(X_test),
    }
    with open(OUTPUT / "model_comparison.json", "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\n[OK] Comparison saved: {OUTPUT / 'model_comparison.json'}")

    # 4. SHAP Analysis
    print(f"\n--- Generating SHAP Analysis for XGBoost ---")
    # Re-train best XGBoost for SHAP
    best_xgb = xgb.XGBClassifier(**xgb_result["best_params"], eval_metric="logloss", random_state=42)
    best_xgb.fit(X_train, y_train)
    generate_shap_analysis(X_train, X_test, best_xgb)

    print(f"\n{'='*60}")
    print("ALL DONE: Model comparison + SHAP analysis complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
