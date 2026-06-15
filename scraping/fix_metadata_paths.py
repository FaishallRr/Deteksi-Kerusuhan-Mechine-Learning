import csv

csv_path = "sample_videos/indonesia/metadata.csv"
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        old_path = row["file_path"]
        parts = old_path.replace("\\", "/").split("/")
        # Old: sample_videos/indonesia/anomaly/fighting/xxx.mp4
        # New: sample_videos/indonesia/anomaly/xxx.mp4
        # Remove subcategory (index 3), keep everything else
        if len(parts) >= 5:
            parts.pop(3)
        row["file_path"] = "/".join(parts)
        rows.append(row)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Fixed {len(rows)} paths")
