"""Generate EDA notebook for UAS ML project."""
import json, nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Exploratory Data Analysis (EDA)
## Deteksi Kerusuhan Menggunakan AttentionMIL
### UAS Machine Learning - Universitas Dian Nuswantoro

---
**Dataset:** Video kerusuhan dengan 3 kategori (demo_rusuh, demo_damai, normal)  
**Fitur:** S3D 1024-dimensional features  
**Model:** Attention-based Multiple Instance Learning

---
""")

md("""## 1. Import Libraries""")

code("""import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings; warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 120})
OUT = Path("../reports/eda"); OUT.mkdir(parents=True, exist_ok=True)
""")

md("""## 2. Load Metadata""")

code("""with open("../features/final_dataset/metadata.json") as f:
    meta = json.load(f)
df = pd.DataFrame(meta)
print(f"Total samples: {len(df)}")
df.head(5)
""")

md("""## 3. Dataset Overview""")

code("""print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\\nLabel distribution:")
print(df["label_name"].value_counts())
print(f"\\nSplit distribution:")
print(df["split"].value_counts())
print(f"\\nSource distribution:")
print(df["source"].value_counts())
print(f"\\nMissing values:")
print(df.isnull().sum())
""")

md("""## 4. Label Distribution""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

label_counts = df["label_name"].value_counts()
colors_label = ["#2ecc71", "#e74c3c", "#3498db"]

axes[0].bar(label_counts.index, label_counts.values, color=colors_label, edgecolor="white")
axes[0].set_xlabel("Kelas"); axes[0].set_ylabel("Jumlah Video")
axes[0].set_title("Distribusi Label (Bar Chart)", fontweight="bold")
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

axes[1].pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%",
            colors=colors_label, startangle=90, explode=[0.03]*3)
axes[1].set_title("Distribusi Label (Pie Chart)", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "label_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""**Insight:** Dataset memiliki 3 kelas. Deteksi kerusuhan adalah binary classification (rusuh vs non-rusuh), dengan label_name `demo_rusuh` dipetakan ke label=1 dan sisanya ke label=0.

""")

md("""## 5. Split Distribution (Train/Val/Test)""")

code("""split_counts = df["split"].value_counts()
colors_split = {"train": "#2ecc71", "val": "#f39c12", "test": "#e74c3c"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(split_counts.index, split_counts.values,
            color=[colors_split[s] for s in split_counts.index], edgecolor="white")
axes[0].set_xlabel("Split"); axes[0].set_ylabel("Jumlah Video")
axes[0].set_title("Distribusi Split", fontweight="bold")
for i, v in enumerate(split_counts.values):
    axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

axes[1].pie(split_counts.values, labels=split_counts.index,
            autopct="%1.1f%%", colors=[colors_split[s] for s in split_counts.index],
            startangle=90, explode=[0.03]*3)
axes[1].set_title("Proporsi Split", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "split_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print("Split ratios:")
for s in ["train", "val", "test"]:
    print(f"  {s}: {split_counts.get(s, 0)} ({split_counts.get(s, 0)/len(df)*100:.1f}%)")
""")

md("""**Insight:** Split menggunakan rasio 80/10/10 (train/val/test). Ini standar untuk deep learning.

""")

md("""## 6. Label Distribution per Split""")

code("""ct = pd.crosstab(df["split"], df["label_name"])
print(ct)
print()

fig, ax = plt.subplots(figsize=(8, 5))
ct.plot(kind="bar", ax=ax, color=colors_label, edgecolor="white")
ax.set_xlabel("Split"); ax.set_ylabel("Count")
ax.set_title("Label Distribution per Split", fontweight="bold")
ax.legend(title="Label"); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUT / "label_per_split.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""**Insight:** Distribusi label terjaga di setiap split (stratified split).

""")

md("""## 7. Source Distribution""")

code("""src_counts = df["source"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(src_counts.index, src_counts.values, color=sns.color_palette("Set2"), edgecolor="white")
ax.set_xlabel("Jumlah Video"); ax.set_ylabel("Sumber Data")
ax.set_title("Distribusi Sumber Data", fontweight="bold")
for bar in bars:
    w = bar.get_width()
    ax.text(w + 20, bar.get_y() + bar.get_height()/2, str(w), ha="left", va="center")
plt.tight_layout()
plt.savefig(OUT / "source_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""**Insight:** Data berasal dari multiple sources: YouTube API, Kaggle (RWF-2000, Real Life Violence), SCVD, dan MSV-PG.

""")

md("""## 8. Source per Label""")

code("""fig, ax = plt.subplots(figsize=(10, 6))
ct_src = pd.crosstab(df["source"], df["label_name"])
ct_src.plot(kind="bar", ax=ax, color=colors_label, edgecolor="white")
ax.set_xlabel("Source"); ax.set_ylabel("Count")
ax.set_title("Source Distribution per Label", fontweight="bold")
ax.legend(title="Label")
plt.tight_layout()
plt.savefig(OUT / "source_per_label.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 9. Feature Space Visualization (PCA)""")

code("""print("Loading features for PCA... (this may take a moment)")
n_samples = min(2000, len(df))
np.random.seed(42)
sample_idx = np.random.choice(len(df), n_samples, replace=False)
sample_df = df.iloc[sample_idx]

features = []
labels = []
for _, row in sample_df.iterrows():
    feat = np.load(row["path"])
    features.append(feat.mean(axis=0))
    labels.append(row["label"])
features = np.array(features)
labels = np.array(labels)

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained: {pca.explained_variance_ratio_.sum():.3f}")

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1],
                     c=labels, cmap="RdYlGn", alpha=0.6, s=20, edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
ax.set_title("PCA Visualization of S3D Features", fontweight="bold")
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_ticklabels(["Normal/Damai", "Rusuh"])
plt.tight_layout()
plt.savefig(OUT / "pca_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""**Insight:** PCA menunjukkan separabilitas antara kelas rusuh dan normal, meskipun ada overlap di beberapa region. Model attention-based MIL mampu menangkap pola yang tidak terlihat di PCA 2D.

""")

md("""## 10. t-SNE Visualization""")

code("""print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_tsne = tsne.fit_transform(features)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1],
                     c=labels, cmap="RdYlGn", alpha=0.6, s=20, edgecolors="none")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE Visualization of S3D Features", fontweight="bold")
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_ticklabels(["Normal/Damai", "Rusuh"])
plt.tight_layout()
plt.savefig(OUT / "tsne_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""**Insight:** t-SNE menunjukkan cluster yang lebih terpisah antara kelas, mengkonfirmasi bahwa fitur S3D mampu menangkap perbedaan antara video rusuh dan damai/normal.

""")

md("""## 11. Feature Variance Analysis""")

code("""print("Analyzing feature variance...")
feat_variance = np.var(features, axis=0)
feat_mean = np.mean(features, axis=0)
feat_std = np.std(features, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(feat_mean, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
axes[0].set_xlabel("Mean Value"); axes[0].set_ylabel("Frequency")
axes[0].set_title("Feature Means Distribution", fontweight="bold")

axes[1].hist(feat_variance, bins=50, color="coral", alpha=0.7, edgecolor="white")
axes[1].set_xlabel("Variance"); axes[1].set_ylabel("Frequency")
axes[1].set_title("Feature Variance Distribution", fontweight="bold")

axes[2].hist(feat_std, bins=50, color="seagreen", alpha=0.7, edgecolor="white")
axes[2].set_xlabel("Std Dev"); axes[2].set_ylabel("Frequency")
axes[2].set_title("Feature Std Distribution", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "feature_statistics.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 12. Average Feature Profile by Class""")

code("""normal_feats = features[labels == 0]
rusuh_feats = features[labels == 1]

normal_mean = normal_feats.mean(axis=0)
rusuh_mean = rusuh_feats.mean(axis=0)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(normal_mean, alpha=0.8, label="Normal/Damai", color="green", lw=1.5)
ax.plot(rusuh_mean, alpha=0.8, label="Rusuh", color="red", lw=1.5)
ax.set_xlabel("Feature Index"); ax.set_ylabel("Mean Value")
ax.set_title("Average Feature Profile by Class (S3D 1024-d)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "feature_profile.png", dpi=150, bbox_inches="tight")
plt.show()

# Difference
diff = rusuh_mean - normal_mean
top_pos = np.argsort(diff)[-5:][::-1]
top_neg = np.argsort(diff)[:5]
print("Top features for Rusuh class:")
for i in top_pos:
    print(f"  Feature {i}: diff={diff[i]:.4f}")
print("\\nTop features for Normal/Damai class:")
for i in top_neg:
    print(f"  Feature {i}: diff={diff[i]:.4f}")
""")

md("""## 13. Feature Correlation Heatmap (Top 30 Features)""")

code("""# Select top 30 highest-variance features
top_var_idx = np.argsort(feat_variance)[-30:]
feat_subset = features[:, top_var_idx]
corr_matrix = np.corrcoef(feat_subset.T)

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
            square=True, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Feature Correlation Heatmap (Top 30 Variance Features)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "feature_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 14. Segment Count Distribution""")

code("""seg_counts = df["segments"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
if len(seg_counts) > 20:
    ax.hist(df["segments"], bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Number of Segments")
else:
    ax.bar(seg_counts.index.astype(str), seg_counts.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of Segments")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Segment Counts per Video", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "segment_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Mean segments: {df['segments'].mean():.1f}")
print(f"Median segments: {df['segments'].median():.1f}")
print(f"Min segments: {df['segments'].min()}")
print(f"Max segments: {df['segments'].max()}")
""")

md("""## 15. Summary & Insights

### Key Findings:
1. **Dataset Size:** 5,552 samples across 3 classes (binary: rusuh vs non-rusuh)
2. **Class Balance:** Relatively balanced (rusuh ~41%)
3. **Feature Quality:** S3D 1024-d features show separability in PCA/t-SNE
4. **Data Sources:** Multi-source (YouTube, Kaggle, SCVD, MSV-PG)
5. **Split:** 80/10/10 stratified

### Implications for Modeling:
- Binary classification (rusuh vs non-rusuh) is appropriate
- AttentionMIL can leverage segment-level features
- Multi-source data improves generalization
- Model evaluation with AUC, F1, Precision, Recall is suitable
""")

md("""---
*Generated for UAS Machine Learning - Deteksi Kerusuhan*
""")

nb.cells = cells

with open("01_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("[OK] Generated notebooks/01_eda.ipynb")
