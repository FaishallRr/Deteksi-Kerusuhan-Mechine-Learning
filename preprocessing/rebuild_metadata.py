"""Rebuild metadata.json and train/val/test splits after adding MSV-PG features."""

import json
import random
import numpy as np
from pathlib import Path

random.seed(42)

FEAT_DIR = Path(__file__).resolve().parent.parent / "features" / "final_dataset"
OUT_META = FEAT_DIR / "metadata.json"
OUT_SPLITS = Path(__file__).resolve().parent.parent / "data" / "splits"

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def main():
    all_entries = []
    for split_dir in ["train", "val", "test"]:
        sp = FEAT_DIR / split_dir
        if not sp.exists():
            continue
        for fpath in sorted(sp.glob("*.npy")):
            name = fpath.stem
            if name.startswith("demo_rusuh_"):
                label, label_name = 1, "demo_rusuh"
            elif name.startswith("demo_damai_"):
                label, label_name = 0, "demo_damai"
            elif name.startswith("normal_"):
                label, label_name = 0, "normal"
            else:
                continue

            pref = name  # full stem
            source = "unknown"
            for s in ["rwf2000", "real_life_violence", "real_life_nonviolence",
                      "scvd", "indonesia", "msv_pg", "cctv"]:
                if s in pref.lower():
                    source = s
                    break
            # Detect YouTube by ID pattern (must contain at least one letter)
            if source == "unknown":
                import re
                suffix = pref.split("_", 2)[-1] if "_" in pref else pref
                if re.match(r'^[A-Za-z0-9_-]{11,}$', suffix) and re.search(r'[A-Za-z]', suffix):
                    source = "youtube"

            arr = np.load(fpath)
            segments = arr.shape[0] if arr.ndim > 1 else 1

            all_entries.append({
                "path": str(fpath),
                "label": label,
                "label_name": label_name,
                "split": split_dir,
                "source": source,
                "segments": segments,
            })

    print(f"Total feature files: {len(all_entries)}")

    rusuh = [e for e in all_entries if e["label_name"] == "demo_rusuh"]
    damai = [e for e in all_entries if e["label_name"] == "demo_damai"]
    normal = [e for e in all_entries if e["label_name"] == "normal"]

    # MSV-PG files have numeric-only suffixes, matched as "unknown" source
    damai_new = [e for e in damai if e["source"] in ("msv_pg", "unknown")]
    damai_old = [e for e in damai if e["source"] not in ("msv_pg", "unknown")]

    print(f"  demo_rusuh: {len(rusuh)}")
    print(f"  demo_damai (old sources): {len(damai_old)}")
    print(f"  demo_damai (msv_pg/unknown): {len(damai_new)}")
    print(f"  normal: {len(normal)}")

    def split_class(items):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    r_train, r_val, r_test = split_class(rusuh)
    do_train, do_val, do_test = split_class(damai_old)
    dn_train, dn_val, dn_test = split_class(damai_new)
    n_train, n_val, n_test = split_class(normal)

    print(f"\n  rusuh:      train={len(r_train):4d} val={len(r_val):4d} test={len(r_test):4d}")
    print(f"  damai_old:  train={len(do_train):4d} val={len(do_val):4d} test={len(do_test):4d}")
    print(f"  damai_new:  train={len(dn_train):4d} val={len(dn_val):4d} test={len(dn_test):4d}")
    print(f"  normal:     train={len(n_train):4d} val={len(n_val):4d} test={len(n_test):4d}")

    for items, split in [(r_train, "train"), (r_val, "val"), (r_test, "test"),
                          (do_train, "train"), (do_val, "val"), (do_test, "test"),
                          (dn_train, "train"), (dn_val, "val"), (dn_test, "test"),
                          (n_train, "train"), (n_val, "val"), (n_test, "test")]:
        for item in items:
            item["split"] = split

    final_metadata = all_entries

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2)
    print(f"\nSaved metadata.json")

    OUT_SPLITS.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        rows = [e for e in final_metadata if e["split"] == split_name]
        csv_path = OUT_SPLITS / f"{split_name}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("path,label,source\n")
            for r in rows:
                f.write(f"{r['path']},{r['label']},{r['source']}\n")
        rusuh_n = sum(1 for e in rows if e["label"] == 1)
        damai_n = sum(1 for e in rows if e["label"] == 0)
        print(f"  {split_name}.csv: {len(rows):4d} rows (rusuh={rusuh_n}, damai={damai_n})")


if __name__ == "__main__":
    main()
