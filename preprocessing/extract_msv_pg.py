"""
Extract S3D features directly from MSV-PG dataset TAR archives.
Bypasses video encoding — reads JPEG frames directly from TAR,
extracts features, and saves to features/final_dataset/.

Usage:
    C:\Python314\python.exe preprocessing/extract_msv_pg.py
"""

import sys
import tarfile
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
TAR_DIR = BASE_DIR / "data" / "demo_damai" / "msv_pg" / "MSV_PG Dataset"
FEAT_DIR = BASE_DIR / "features" / "final_dataset"

# Mapping: MSV-PG class -> our label
LABEL_MAP = {
    "lpg": "demo_damai",
    "natural": "normal",
}

# TAR files to process: (split, msv_class) -> our split, our label
TARS = [
    ("train", "lpg"),
    ("train", "natural"),
    ("val", "lpg"),
    ("val", "natural"),
]

MAX_SAMPLES = {
    ("train", "lpg"): 800,
    ("train", "natural"): 0,
    ("val", "lpg"): 100,
    ("val", "natural"): 0,
}

def decode_jpeg_from_tar(tar, member) -> np.ndarray:
    data = tar.extractfile(member).read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def is_jpg(name: str) -> bool:
    return name.endswith(".jpg") or name.endswith(".jpeg")

def extract_features_from_tar(tar_path: str, our_split: str, our_label: str, extractor, batch_size=8, max_samples=500):
    """Process one TAR file, extract features in batches, save to FEAT_DIR/our_split/"""
    out_dir = FEAT_DIR / our_split
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        samples = {}
        for m in tar:
            if m.isfile() and is_jpg(m.name):
                parts = m.name.split("/")
                if len(parts) < 4:
                    continue
                sample_id = parts[2]
                samples.setdefault(sample_id, []).append(m)

        sample_ids = list(samples.keys())
        pbar = tqdm(total=len(sample_ids), desc=f"{our_split}/{our_label}")

        for start in range(0, len(sample_ids), batch_size):
            if total >= max_samples:
                pbar.update(len(sample_ids) - start)
                break

            batch_ids = sample_ids[start:start + batch_size]
            batch_frames = []
            batch_names = []

            for sid in batch_ids:
                if total + len(batch_names) >= max_samples:
                    break
                members = samples[sid]
                members.sort(key=lambda m: m.name)
                selected = members[:16]
                if len(selected) < 16:
                    skipped += 1
                    continue
                frames = [decode_jpeg_from_tar(tar, m) for m in selected]
                batch_frames.append(frames)
                safe_id = sid.replace("sample", "").replace("_", "")
                batch_names.append(f"{our_label}_{safe_id}.npy")

            if not batch_frames:
                pbar.update(len(batch_ids))
                continue

            features = extractor.extract_batch(batch_frames)
            for i, name in enumerate(batch_names):
                np.save(out_dir / name, features[i])
            total += len(batch_frames)
            pbar.update(min(len(batch_ids), max_samples - (total - len(batch_frames))))

        pbar.close()
    return total, skipped


def main():
    from preprocessing.feature_extractor import TemporalFeatureExtractor

    print("Loading S3D feature extractor...")
    extractor = TemporalFeatureExtractor(architecture="s3d", device="cuda")
    print("Model ready.\n")

    grand_total = 0
    grand_skip = 0

    for split, msv_class in TARS:
        our_label = LABEL_MAP[msv_class]
        our_split = "train" if split == "train" else "val"

        max_s = MAX_SAMPLES.get((split, msv_class), 0)
        if max_s == 0:
            print(f"\nSKIP {split}/{msv_class} (max_samples=0)")
            continue

        tar_path = str(TAR_DIR / split / f"{msv_class}.tar.gz")
        if not Path(tar_path).exists():
            print(f"SKIP {tar_path} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split}/{msv_class} -> {our_split}/{our_label}")
        print(f"  Source: {tar_path} | Max: {max_s}")

        total, skipped = extract_features_from_tar(tar_path, our_split, our_label, extractor, max_samples=max_s)
        grand_total += total
        grand_skip += skipped

        print(f"  Extracted: {total} | Skipped (<16 frames): {skipped}")

    print(f"\n{'='*60}")
    print(f"Done! Total features extracted: {grand_total}")
    print(f"Total skipped: {grand_skip}")


if __name__ == "__main__":
    main()
