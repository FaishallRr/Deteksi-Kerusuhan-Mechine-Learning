"""
Batch download Facebook Reels from URL files.
Usage: python scraping/download_fb_batch.py
"""
import subprocess
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import time

BASE_DIR = Path(__file__).resolve().parent.parent
FFMPEG_PATH = str(BASE_DIR / "bin" / "ffmpeg.exe")
OUTPUT_DIR = BASE_DIR / "sample_videos" / "indonesia_v7"

URL_FILES = {
    "demo_damai": BASE_DIR / "scraping" / "fb_urls_demo_damai.txt",
    "demo_rusuh": BASE_DIR / "scraping" / "fb_urls_demo_rusuh.txt",
}


def read_urls(filepath: Path) -> list:
    if not filepath.exists():
        return []
    urls = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def download_video(url: str, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", FFMPEG_PATH,
        "-o", str(output_path / "%(id)s.%(ext)s"),
        "--no-playlist",
        "--no-overwrites",
        "--quiet",
        url,
    ]

    # Try with cookies first, then public
    for attempt in ["cookies", "public"]:
        cmd2 = cmd.copy()
        if attempt == "cookies":
            cmd2 += ["--cookies-from-browser", "chrome"]
        try:
            result = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass

    return False


def get_downloaded_id(output_path: Path) -> str:
    files = list(output_path.glob("*"))
    if files:
        for f in files:
            if f.suffix in (".mp4", ".webm", ".mkv", ".avi"):
                return f.stem
    return ""


def run():
    metadata = []
    total_downloaded = 0
    total_urls = 0

    for category, url_file in URL_FILES.items():
        urls = read_urls(url_file)
        if not urls:
            print(f"  SKIP: {url_file.name} (empty/not found)")
            continue

        total_urls += len(urls)
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{category}] {len(urls)} URLs")
        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] {url[:50]}...", end=" ")
            success = download_video(url, category_dir)

            if success:
                video_id = url.split("/")[-1].replace("/", "")
                downloaded_files = list(category_dir.glob("*"))
                for f in downloaded_files:
                    if f.suffix in (".mp4", ".webm", ".mkv"):
                        label = 0 if category == "demo_damai" else 1
                        metadata.append({
                            "file_path": str(f),
                            "video_id": f.stem,
                            "title": f.stem,
                            "category": category,
                            "label": label,
                            "source": "Facebook",
                            "channel": "",
                            "duration_sec": 0,
                            "keyword": category,
                            "downloaded_at": datetime.now().isoformat(),
                        })
                        total_downloaded += 1
                print("OK")
            else:
                print("FAILED")
            time.sleep(2)

    # Save metadata
    csv_path = OUTPUT_DIR / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file_path", "video_id", "title", "category", "label",
            "source", "channel", "duration_sec", "keyword", "downloaded_at",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\n{'='*50}")
    print(f"FACEBOOK BATCH DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"  URLs processed: {total_urls}")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Metadata: {csv_path}")


if __name__ == "__main__":
    run()
