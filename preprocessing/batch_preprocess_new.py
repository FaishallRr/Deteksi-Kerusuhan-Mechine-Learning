"""
Optimized batch feature extraction for newly collected videos.
Uses extract_batch to process ALL segments from a video in a single forward pass.
"""
import sys
sys.path.insert(0, ".")

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import json


def extract_video_frames(video_path: Path, target_size=(224, 224), fps_sampling=2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % fps_sampling == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        count += 1
    cap.release()
    if len(frames) < 16:
        return None
    return np.array(frames)


def segment_frames(frames: np.ndarray, temporal_window=16):
    segments = []
    for i in range(0, len(frames) - temporal_window + 1, temporal_window):
        segments.append(frames[i:i + temporal_window])
    return segments


def process_dataset(metadata_path: str, features_dir: str, device="cuda"):
    from preprocessing.feature_extractor import TemporalFeatureExtractor

    metadata_path = Path(metadata_path)
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sources = {}
    for row in rows:
        src = row["source"]
        sources.setdefault(src, []).append(row)

    print(f"Loaded {len(rows)} videos from {len(sources)} sources")
    print(f"Device: {device}")

    extractor = TemporalFeatureExtractor("s3d", device)
    all_features_meta = []
    total_processed = 0
    total_skipped = 0

    for source_name, source_rows in sources.items():
        source_feat_dir = features_dir / source_name
        source_feat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Source: {source_name} ({len(source_rows)} videos)")
        print(f"{'='*60}")

        for row in tqdm(source_rows, desc=f"  {source_name[:35]}"):
            video_path = Path(row["file_path"])
            if not video_path.exists():
                total_skipped += 1
                continue

            feat_path = source_feat_dir / f"{row['video_id']}.npy"
            # Skip if already processed
            if feat_path.exists():
                try:
                    existing = np.load(feat_path)
                    all_features_meta.append({
                        "feature_file": str(feat_path),
                        "video_id": row["video_id"],
                        "num_segments": existing.shape[0] if existing.ndim > 1 else 1,
                        "feature_dim": existing.shape[-1] if existing.ndim > 0 else 1024,
                        "label": int(row["label"]),
                        "source": source_name,
                        "category": row.get("category", ""),
                    })
                    total_processed += 1
                    continue
                except Exception:
                    pass

            frames = extract_video_frames(video_path)
            if frames is None:
                total_skipped += 1
                continue

            segments = segment_frames(frames, temporal_window=16)
            if not segments:
                total_skipped += 1
                continue

            # Process segments in micro-batches to avoid OOM
            try:
                batch_size = 16  # max segments per forward pass
                all_feat = []
                for i in range(0, len(segments), batch_size):
                    batch_segs = segments[i:i + batch_size]
                    feat = extractor.extract_batch(batch_segs)
                    all_feat.append(feat)
                features = np.concatenate(all_feat, axis=0) if len(all_feat) > 1 else all_feat[0]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                print(f"\n  ERROR: {e}")
                total_skipped += 1
                continue
            except Exception as e:
                print(f"\n  ERROR: {e}")
                total_skipped += 1
                continue
            finally:
                torch.cuda.empty_cache()

            if features.size == 0:
                total_skipped += 1
                continue

            np.save(str(feat_path), features)

            all_features_meta.append({
                "feature_file": str(feat_path),
                "video_id": row["video_id"],
                "num_segments": features.shape[0] if features.ndim > 1 else 1,
                "feature_dim": features.shape[-1] if features.ndim > 0 else 1024,
                "label": int(row["label"]),
                "source": source_name,
                "category": row.get("category", ""),
            })
            total_processed += 1

        # Save per-source metadata
        source_meta = [m for m in all_features_meta if m["source"] == source_name]
        if source_meta:
            meta_path = source_feat_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(source_meta, f, indent=2)

    unified_path = features_dir / "metadata.json"
    with open(unified_path, "w") as f:
        json.dump(all_features_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BATCH PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped:   {total_skipped}")
    print(f"  Features:  {features_dir}/")

    label_counts = {}
    for m in all_features_meta:
        label_counts[m["label"]] = label_counts.get(m["label"], 0) + 1
    print(f"\n  Label distribution:")
    for lbl, count in sorted(label_counts.items()):
        print(f"    Label {lbl}: {count} videos")

    return all_features_meta


if __name__ == "__main__":
    import time
    start = time.time()
    process_dataset(
        metadata_path="dataset/unified_metadata.csv",
        features_dir="features/new_datasets",
        device="cuda",
    )
    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
