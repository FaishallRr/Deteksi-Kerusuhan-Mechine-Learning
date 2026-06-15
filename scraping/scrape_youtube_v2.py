import subprocess
import json
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime


EXCLUDE_TITLE_PATTERNS = [
    r"tvOne", r"iNews", r"LintasiNews", r"Kabar\s", r"REDAKSI",
    r"Buser", r"Police\s*Line", r"LaporPolisi", r"\bBERITA\b", r"\bBerita\b",
    r"kompas", r"detikcom", r"kumparan", r"liputan6", r"tribun",
    r"Kompilasi", r"Pagi\s", r"Malam\s", r"Siang\s", r"Sore\s",
    r"\d{4}", r"Prioritas Indonesia", r"Jakarta Today",
    r"Seputaran\s*News", r"Buletin", r"Lintas\s",
    r"AKIP\s*tvOne", r"BIM\s*\d{4}", r"LIM\s*\d{4}", r"LIP\s*\d{4}",
    r"86\s*$", r"NET\s*\d", r"BicaraDigital", r"tips$",
    r"PEMBELAJARAN", r"Materi\s", r"Belajar\s", r"Animasi\s",
    r"EFEK SUARA", r"SOUND EFFECT",
    r"BIPA", r"EKONOMI", r"Pasar\s*Modern",
]

EXCLUDE_CHANNEL_PATTERNS = [
    r"tvOneNews", r"iNews", r"Kompas", r"detikcom", r"Tribun",
    r"Official\s*iNews", r"METRO\s*TV", r"TV\s*One",
    r"Buser\s*Indonesia", r"LaporPolisi",
]


def title_matches_exclude(title: str) -> bool:
    for pat in EXCLUDE_TITLE_PATTERNS:
        if re.search(pat, title, re.IGNORECASE):
            return True
    return False


def channel_matches_exclude(channel: str) -> bool:
    for pat in EXCLUDE_CHANNEL_PATTERNS:
        if re.search(pat, channel, re.IGNORECASE):
            return True
    return False


class YouTubeScraperV2:
    def __init__(self, output_dir: str = "sample_videos/indonesia_v2"):
        self.output_dir = Path(output_dir)
        self.metadata = []
        self.downloaded_ids = set()

        self.anomaly_keywords = {
            "fighting": [
                "tawuran CCTV",
                "tawuran pelajar",
                "tawuran jalanan",
                "perkelahian massal",
                "bentrokan massa",
                "carok maut CCTV",
            ],
            "robbery": [
                "perampokan CCTV",
                "perampokan minimarket",
                "perampokan toko emas",
                "perampokan bank",
                "pencurian motor CCTV",
                "copet CCTV",
            ],
            "riot": [
                "demo ricuh",
                "kerusuhan massa",
                "bentrok warga",
                "anarkis demo",
                "kerusuhan tahanan",
                "huruhara",
            ],
            "criminal": [
                "begal CCTV",
                "begal motor",
                "jambret CCTV",
                "pengeroyokan CCTV",
                "kejahatan jalanan",
            ],
        }

        self.normal_keywords = {
            "traffic": [
                "CCTV lalu lintas",
                "CCTV jalan tol",
                "CCTV simpang",
                "pantauan CCTV jalan raya",
                "arus kendaraan CCTV",
            ],
            "market": [
                "suasana pasar tradisional",
                "pasar ramai",
                "aktivitas jual beli pasar",
                "pasar pagi hari",
                "CCTV pasar",
            ],
            "street": [
                "suasana jalan kota",
                "aktivitas warga kota",
                "jalan protokol",
                "suasana terminal",
                "CCTV alun alun",
                "keramaian kota",
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
            return []
        except Exception:
            return []

    def download_video(self, video: dict, output_path: Path) -> bool:
        video_id = video.get("id", "")
        if video_id in self.downloaded_ids:
            return False

        url = f"https://www.youtube.com/watch?v={video_id}"
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
        except Exception:
            pass
        return False

    def is_valid_video(self, video: dict, min_dur: int, max_dur: int) -> tuple:
        title = video.get("title", "")
        channel = video.get("channel", "") or video.get("uploader", "") or ""
        duration = video.get("duration", 0)

        if duration and (duration < min_dur or duration > max_dur):
            return False, f"duration {duration}s (not in {min_dur}-{max_dur})"

        if channel_matches_exclude(channel):
            return False, f"news channel: {channel[:40]}"

        if title_matches_exclude(title):
            return False, f"news title: {title[:60]}"

        return True, "ok"

    def clean_title(self, title: str) -> str:
        title = re.sub(r'[^\w\s\-]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()[:80]
        return title

    def scrape_category(self, keywords: dict, label: int, max_per_keyword: int = 25):
        total_downloaded = 0
        min_dur = 15 if label == 1 else 30
        max_dur = 90 if label == 1 else 300

        for category, kw_list in keywords.items():
            category_dir = self.output_dir / ("anomaly" if label == 1 else "normal")
            category_dir.mkdir(parents=True, exist_ok=True)

            for keyword in kw_list:
                if total_downloaded >= max_per_keyword * len(kw_list):
                    break

                print(f"  Searching: '{keyword}'...")
                videos = self.search_videos(keyword, max_results=max_per_keyword + 10)

                valid = []
                for v in videos:
                    ok, reason = self.is_valid_video(v, min_dur, max_dur)
                    if ok:
                        valid.append(v)

                print(f"    Found {len(videos)}, valid after filter: {len(valid)}")

                if valid:
                    print(f"    Sample titles:")
                    for v in valid[:5]:
                        print(f"      - {self.clean_title(v.get('title',''))[:60]} ({v.get('duration',0)}s)")

                for v in valid:
                    if total_downloaded >= max_per_keyword * len(kw_list):
                        break

                    title = self.clean_title(v.get("title", ""))
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
                            "channel": v.get("channel", "") or v.get("uploader", "") or "",
                            "duration_sec": v.get("duration", 0),
                            "keyword": keyword,
                            "downloaded_at": datetime.now().isoformat(),
                        })
                        print("OK")
                    else:
                        print("SKIP")

                    time.sleep(1.5)

        return total_downloaded

    def save_metadata(self):
        csv_path = self.output_dir / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "file_path", "video_id", "title", "category", "label",
                "source", "channel", "duration_sec", "keyword", "downloaded_at",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metadata)
        print(f"\nMetadata saved: {csv_path}")
        print(f"Total videos: {len(self.metadata)}")
        return csv_path

    def run(self, max_per_keyword: int = 25):
        print("=" * 60)
        print("YOUTUBE SCRAPER V2 — DATA INDONESIA (FILTERED)")
        print("=" * 60)
        print(f"Exclude patterns: {len(EXCLUDE_TITLE_PATTERNS)} title + {len(EXCLUDE_CHANNEL_PATTERNS)} channel")
        print(f"Anomaly duration: 15-90s | Normal duration: 30-300s")
        print()

        print(f"[1/2] Scraping ANOMALY videos...")
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
        print(f"SCRAPING V2 COMPLETE")
        print(f"{'='*60}")
        print(f"  Anomaly: {anomaly_count}")
        print(f"  Normal:  {normal_count}")
        print(f"  Total:   {anomaly_count + normal_count}")
        print(f"  Metadata: {csv_path}")
        print()

        return self.metadata, csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sample_videos/indonesia_v2")
    parser.add_argument("--max-per-keyword", type=int, default=20)
    args = parser.parse_args()

    scraper = YouTubeScraperV2(output_dir=args.output)
    scraper.run(max_per_keyword=args.max_per_keyword)
