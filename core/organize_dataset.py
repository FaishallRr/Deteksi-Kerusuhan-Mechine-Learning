"""
Organize all collected videos into a unified training dataset.
Scans all sample_videos/ sources, deduplicates, creates unified metadata.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_METADATA = BASE_DIR / "dataset" / "unified_metadata.csv"

# Map source directories to their label/category info
# (source_dir, default_label, default_category, has_subdirs)
SOURCES = [
    # Facebook demo videos (newly collected)
    ("sample_videos/indonesia_v7/demo_damai", 0, "demo_damai", False),
    ("sample_videos/indonesia_v7/demo_rusuh", 1, "demo_rusuh", False),

    # YouTube v6 - anomaly
    ("sample_videos/indonesia_v6/anomaly", 1, "anomaly", "mixed"),
    # YouTube v6 - normal
    ("sample_videos/indonesia_v6/normal", 0, "normal", "mixed"),

    # Existing Facebook data
    ("sample_videos/indonesia_v4", 1, "anomaly", False),
    ("sample_videos/indonesia_v5", 1, "anomaly", False),

    # Demo Indonesia (existing)
    ("sample_videos/demo_indonesia", 0, "demo_damai", False),

    # CCTV sources (normal)
    ("sample_videos/cctv_indonesia_scraped", 0, "traffic", False),
    ("sample_videos/cctv_indonesia", 0, "traffic", False),

    # v3 data (already organized)
    ("sample_videos/indonesia_v3/anomaly", 1, "anomaly", True),
    ("sample_videos/indonesia_v3/normal", 0, "normal", True),

    # FP test set
    ("sample_videos/fp_test", 0, "fp_test", False),

    # Root test files
    ("sample_videos", None, None, False),
]


def scan_videos(source_dir: Path, default_label: int, default_category: str, has_subdirs):
    """Scan a directory and collect video files with metadata."""
    if not source_dir.exists():
        return []

    results = []
    video_exts = {".mp4", ".avi", ".webm", ".mkv"}

    if has_subdirs == "mixed":
        # Files directly in parent + files in subdirs
        for f in sorted(source_dir.iterdir()):
            if f.suffix.lower() in video_exts and not f.name.startswith("_"):
                results.append({
                    "file_path": str(f),
                    "video_id": f.stem,
                    "title": f.stem,
                    "category": default_category,
                    "label": default_label,
                    "source": source_dir.parent.name,
                })
        for subdir in sorted(source_dir.iterdir()):
            if not subdir.is_dir():
                continue
            category = subdir.name
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in video_exts and not f.name.startswith("_"):
                    results.append({
                        "file_path": str(f),
                        "video_id": f.stem,
                        "title": f.stem,
                        "category": category,
                        "label": default_label,
                        "source": source_dir.parent.name,
                    })

    elif has_subdirs:
        # Each subdirectory is a category
        for subdir in sorted(source_dir.iterdir()):
            if not subdir.is_dir():
                continue
            category = subdir.name
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in video_exts and not f.name.startswith("_"):
                    results.append({
                        "file_path": str(f),
                        "video_id": f.stem,
                        "title": f.stem,
                        "category": category,
                        "label": 1 if category == "anomaly" else 0,
                        "source": source_dir.parent.name if source_dir.parent.name != "sample_videos" else "root",
                    })
    else:
        for f in sorted(source_dir.iterdir()):
            if f.suffix.lower() in video_exts and not f.name.startswith("_"):
                label = default_label
                cat = default_category
                # Handle root-level test files
                if label is None:
                    if f.stem.startswith("anomaly") or f.stem.startswith("test_anomaly"):
                        label, cat = 1, "anomaly"
                    elif f.stem.startswith("normal") or f.stem.startswith("test_normal"):
                        label, cat = 0, "normal"
                    else:
                        continue
                results.append({
                    "file_path": str(f),
                    "video_id": f.stem,
                    "title": f.stem,
                    "category": cat,
                    "label": label,
                    "source": source_dir.parent.name if source_dir.parent.name != "sample_videos" else "root",
                })

    return results


def deduplicate(entries: list) -> list:
    """Remove duplicates based on video_id."""
    seen = set()
    unique = []
    for e in entries:
        key = e["video_id"]
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def run():
    all_entries = []
    source_counts = defaultdict(int)

    for rel_dir, label, category, has_subdirs in SOURCES:
        src_path = BASE_DIR / rel_dir
        entries = scan_videos(src_path, label, category, has_subdirs)

        # Label root files
        source_name = rel_dir.replace("/", "_").replace("\\", "_")
        for e in entries:
            e["source"] = source_name
        all_entries.extend(entries)
        source_counts[source_name] = len(entries)

    # Deduplicate
    all_entries = deduplicate(all_entries)

    # Sort
    all_entries.sort(key=lambda x: (x["label"], x["category"], x["video_id"]))

    # Save
    BASE_DIR.joinpath("dataset").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METADATA, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_path", "video_id", "title", "category", "label", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)

    # Print report
    print(f"{'='*60}")
    print(f"UNIFIED DATASET ORGANIZER")
    print(f"{'='*60}")
    print(f"\nSources scanned:")
    for name, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {name}: {count} videos")

    label_counts = defaultdict(int)
    cat_counts = defaultdict(int)
    for e in all_entries:
        label_counts[e["label"]] += 1
        cat_counts[(e["label"], e["category"])] += 1

    print(f"\nLabel distribution:")
    print(f"  Normal    (0): {label_counts[0]}")
    print(f"  Anomaly   (1): {label_counts[1]}")

    print(f"\nCategory breakdown:")
    for (label, cat), count in sorted(cat_counts.items()):
        lbl = "ANOMALY" if label == 1 else "NORMAL"
        print(f"  [{lbl}] {cat}: {count}")

    print(f"\nTotal unique videos: {len(all_entries)}")
    print(f"Output: {OUTPUT_METADATA}")


if __name__ == "__main__":
    run()
