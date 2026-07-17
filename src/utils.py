\"\"\"Utility functions for the project.\"\"\"
import sys, os, json, numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_metadata(path=\"features/final_dataset/metadata.json\"):
    with open(path) as f:
        return json.load(f)

def load_features(path):
    return np.load(path)

def get_split_stats(meta):
    splits = Counter(m[\"split\"] for m in meta)
    labels = Counter(m[\"label\"] for m in meta)
    return splits, labels

print(\"Utils loaded.\")
