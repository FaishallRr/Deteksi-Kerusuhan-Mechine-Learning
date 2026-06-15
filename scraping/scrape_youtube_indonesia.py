import subprocess
import json
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime


class YouTubeScraperIndonesia:
    def __init__(self, output_dir: str = "sample_videos/indonesia"):
        self.output_dir = Path(output_dir)
        self.metadata = []
        self.downloaded_ids = set()

        self.anomaly_keywords = {
            "fighting": [
                "tawuran CCTV Indonesia",
                "tawuran pelajar terekam CCTV",
                "perkelahian massal CCTV",
                "tawuran jalanan Indonesia",
            ],
            "robbery": [
                "perampokan terekam CCTV Indonesia",
                "perampokan toko CCTV",
                "perampokan bersenjata CCTV",
                "pencurian motor CCTV",
            ],
            "riot": [
                "kerusuhan Indonesia CCTV",
                "demo ricuh terekam CCTV",
                "bentrok massa CCTV",
                "anarkis demo Indonesia CCTV",
            ],
            "criminal": [
                "begal CCTV Indonesia",
                "jambret terekam CCTV",
                "penjambretan CCTV",
                "kejahatan jalanan CCTV Indonesia",
                "pengeroyokan terekam CCTV",
            ],
        }

        self.normal_keywords = {
            "traffic": [
                "CCTV lalu lintas jalan raya Indonesia",
                "suasana jalan raya Jakarta CCTV",
                "arus lalu lintas kota CCTV Indonesia",
                "kemacetan jalan tol CCTV",
            ],
            "market": [
                "aktivitas pasar tradisional Indonesia",
                "suasana pasar pagi hari",
                "kegiatan jual beli pasar tradisional",
                "CCTV pasar tradisional ramai",
            ],
            "street": [
                "suasana jalan desa CCTV",
                "aktivitas warga di pusat kota",
                "kegiatan sehari-hari masyarakat Indonesia",
                "jalan protokol kota besar Indonesia",
                "suasana terminal bus CCTV",
            ],
        }

    def search_videos(self, keyword: str, max_results: int = 30) -> list:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            f"ytsearch{max_results}:{keyword}",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            videos = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        videos.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return videos
        except subprocess.TimeoutExpired:
            print(f"  [!] Timeout searching: {keyword}")
            return []
        except Exception as e:
            print(f"  [!] Error searching: {e}")
            return []

    def download_video(self, video: dict, output_path: Path) -> bool:
        video_id = video.get("id", "")
        if video_id in self.downloaded_ids:
            return False

        url = f"https://www.youtube.com/watch?v={video_id}"
        duration = video.get("duration", 0)

        if duration and (duration < 30 or duration > 600):
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        template = str(output_path.parent / f"{video_id}.%(ext)s")

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "best[height<=480][ext=mp4]",
            "--max-filesize", "50M",
            "-o", template,
            "--no-warnings",
            "--no-playlist",
            url,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            downloaded = list(output_path.parent.glob(f"{video_id}.*"))
            if downloaded:
                self.downloaded_ids.add(video_id)
                return True
        except subprocess.TimeoutExpired:
            print(f"    [!] Download timeout: {video_id}")
        except Exception as e:
            print(f"    [!] Download error: {e}")
        return False

    def get_video_title(self, video: dict) -> str:
        title = video.get("title", "unknown")
        title = re.sub(r'[^\w\s\-]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()[:80]
        return title

    def scrape_category(self, keywords: dict, label: int, max_per_keyword: int = 25):
        total_downloaded = 0
        for category, kw_list in keywords.items():
            category_dir = self.output_dir / ("anomaly" if label == 1 else "normal") / category
            category_dir.mkdir(parents=True, exist_ok=True)

            for keyword in kw_list:
                if total_downloaded >= max_per_keyword * len(kw_list):
                    break

                print(f"  Searching: '{keyword}'...")
                videos = self.search_videos(keyword, max_results=max_per_keyword)
                print(f"    Found {len(videos)} videos")

                for v in videos:
                    if total_downloaded >= max_per_keyword * len(kw_list):
                        break

                    duration = v.get("duration", 0)
                    if duration and (duration < 30 or duration > 600):
                        continue

                    title = self.get_video_title(v)
                    print(f"    Downloading: {title[:50]}...", end=" ")

                    success = self.download_video(v, category_dir)
                    if success:
                        total_downloaded += 1
                        self.metadata.append({
                            "file_path": str(category_dir / f"{v.get('id', '')}.mp4"),
                            "video_id": v.get("id", ""),
                            "title": title,
                            "category": category,
                            "label": label,
                            "source": "YouTube",
                            "duration_sec": duration,
                            "keyword": keyword,
                            "downloaded_at": datetime.now().isoformat(),
                        })
                        print("[OK]")
                    else:
                        print("[SKIP]")

                    time.sleep(1)

        return total_downloaded

    def save_metadata(self):
        csv_path = self.output_dir / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "file_path", "video_id", "title", "category", "label",
                "source", "duration_sec", "keyword", "downloaded_at",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metadata)
        print(f"\nMetadata saved: {csv_path}")
        print(f"Total videos: {len(self.metadata)}")
        return csv_path

    def run(self, max_per_keyword: int = 25):
        print("=" * 60)
        print("YOUTUBE SCRAPER — DATA INDONESIA")
        print("=" * 60)

        print(f"\n[1/2] Scraping ANOMALY videos...")
        anomaly_count = self.scrape_category(
            self.anomaly_keywords, label=1, max_per_keyword=max_per_keyword
        )
        print(f"  Total anomaly: {anomaly_count}")

        print(f"\n[2/2] Scraping NORMAL videos...")
        normal_count = self.scrape_category(
            self.normal_keywords, label=0, max_per_keyword=max_per_keyword
        )
        print(f"  Total normal: {normal_count}")

        csv_path = self.save_metadata()

        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"  Anomaly: {anomaly_count}")
        print(f"  Normal:  {normal_count}")
        print(f"  Total:   {anomaly_count + normal_count}")
        print(f"  Metadata: {csv_path}")
        print()

        return self.metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sample_videos/indonesia")
    parser.add_argument("--max-per-keyword", type=int, default=20)
    args = parser.parse_args()

    scraper = YouTubeScraperIndonesia(output_dir=args.output)
    scraper.run(max_per_keyword=args.max_per_keyword)
