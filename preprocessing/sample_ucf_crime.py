import sys
sys.path.insert(0, ".")

import shutil
from pathlib import Path
from typing import List
import random
import re


def sample_ucf_crime_frames(
    source_dir: str = "ucf_crime_raw",
    target_dir: str = "sample_videos/ucf_crime_frames",
    videos_per_class: int = 15,
    normal_count: int = 25,
    seed: int = 42,
):
    random.seed(seed)
    source = Path(source_dir)
    target = Path(target_dir)

    anomaly_classes = ["Fighting", "Assault", "Robbery", "Shooting", "Abuse", "Arson"]
    normal_classes = ["NormalVideos"]

    def get_video_ids(class_dir: Path) -> List[str]:
        ids = set()
        for f in class_dir.iterdir():
            m = re.match(r"(.+)_\d+\.png", f.name)
            if m:
                ids.add(m.group(1))
        return sorted(ids)

    total_copied = 0
    for split in ["Train", "Test"]:
        split_source = source / split
        split_target = target / split

        for cls in anomaly_classes:
            cls_source = split_source / cls
            if not cls_source.exists():
                continue
            all_ids = get_video_ids(cls_source)
            random.shuffle(all_ids)
            n = videos_per_class if split == "Train" else max(3, videos_per_class // 2)
            selected = all_ids[:n]

            cls_target = split_target / cls
            for vid_id in selected:
                cls_target.mkdir(parents=True, exist_ok=True)
                for f in cls_source.glob(f"{vid_id}_*.png"):
                    shutil.copy2(f, cls_target / f.name)
                    total_copied += 1

        if split == "Train":
            cls_source = split_source / "NormalVideos"
            all_ids = get_video_ids(cls_source)
            random.shuffle(all_ids)
            selected = all_ids[:normal_count]
            cls_target = split_target / "NormalVideos"
            for vid_id in selected:
                cls_target.mkdir(parents=True, exist_ok=True)
                for f in cls_source.glob(f"{vid_id}_*.png"):
                    shutil.copy2(f, cls_target / f.name)
                    total_copied += 1

    print(f"Total frames copied: {total_copied}")
    print(f"Structure:")
    for split in ["Train", "Test"]:
        split_dir = target / split
        if not split_dir.exists():
            continue
        for cls in sorted(split_dir.iterdir()):
            if cls.is_dir():
                vid_ids = set()
                for f in cls.iterdir():
                    m = re.match(r"(.+)_\d+\.png", f.name)
                    if m:
                        vid_ids.add(m.group(1))
                frames = len(list(cls.iterdir()))
                label = "NORMAL" if "Normal" in cls.name else "ANOMALY"
                print(f"  {split}/{cls.name}/: {len(vid_ids)} videos, {frames} frames [{label}]")


if __name__ == "__main__":
    sample_ucf_crime_frames()
