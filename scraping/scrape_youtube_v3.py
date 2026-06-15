import subprocess
import json
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime


# Less aggressive exclusions — removed \d{4}, Pagi/Malam/Siang/Sore, etc.
# These blocked too many valid videos
EXCLUDE_TITLE_PATTERNS = [
    r"tvOne", r"iNews", r"LintasiNews", r"Kabar\s", r"REDAKSI",
    r"Buser", r"Police\s*Line", r"LaporPolisi", r"\bBERITA\b", r"\bBerita\b",
    r"kompas", r"detikcom", r"kumparan", r"liputan6", r"tribun",
    r"Kompilasi", r"Prioritas Indonesia", r"Jakarta Today",
    r"Seputaran\s*News", r"Buletin", r"Lintas\s",
    r"AKIP\s*tvOne", r"BIM\s*\d{4}", r"LIM\s*\d{4}", r"LIP\s*\d{4}",
    r"86\s*$", r"NET\s*\d", r"BicaraDigital",
    r"PEMBELAJARAN", r"Materi\s", r"Belajar\s", r"Animasi\s",
    r"BIPA", r"EKONOMI",
]

# Extra exclusions specifically for the NORMAL category
# (tutorials, music videos, reviews, etc.)
NORMAL_EXCLUDE_TITLE_PATTERNS = [
    r"\bCara\b", r"\bTutorial\b", r"\bPanduan\b", r"\bTips\b", r"\bReview\b",
    r"\bTest\b", r"\bUnboxing\b",
    r"Official\s*Music\s*Video", r"Official\s*Lyric", r"\bLagu\b", r"\bSong\b",
    r"\bOST\b", r"\bLyric", r"\bMV\b", r"\bMusic\s*Video\b",
]

EXCLUDE_CHANNEL_PATTERNS = [
    r"tvOneNews", r"iNews", r"Kompas", r"detikcom", r"Tribun",
    r"Official\s*iNews", r"METRO\s*TV", r"TV\s*One",
    r"Buser\s*Indonesia", r"LaporPolisi",
]


def title_matches_exclude(title: str, patterns: list = None) -> bool:
    if patterns is None:
        patterns = EXCLUDE_TITLE_PATTERNS
    for pat in patterns:
        if re.search(pat, title, re.IGNORECASE):
            return True
    return False


def channel_matches_exclude(channel: str) -> bool:
    for pat in EXCLUDE_CHANNEL_PATTERNS:
        if re.search(pat, channel, re.IGNORECASE):
            return True
    return False


# Known Indonesian CCTV live channels to scrape for normal data
# Channel: CCTV PENGAWAS LALU LINTAS DISHUB KOTA PONTIANAK
# Contains 24/7 live streams of traffic cameras in Pontianak
KNOWN_CCTV_CHANNELS = [
    # DISHUB Pontianak — 24/7 raw traffic cam live streams (BEST!)
    "https://www.youtube.com/@CCTVDISHUBKOTAPONTIANAK/streams",
    # NTMC POLRI — traffic monitoring streams (pre-recorded, good quality)
    "https://www.youtube.com/@NTMCPOLRIOFFICIAL/streams",
]

# Max seconds to download from a live stream
MAX_LIVE_DOWNLOAD_SEC = 90


class YouTubeScraperV3:
    def __init__(self, output_dir: str = "sample_videos/indonesia_v3"):
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
                "CCTV lalu lintas live",
                "CCTV jalan tol live",
                "CCTV simpang live",
                "arus kendaraan CCTV live",
                "pantauan CCTV lalu lintas",
                "live traffic Indonesia CCTV",
                "DISHUB CCTV live",
                "DISHUB Pontianak CCTV",
                "DISHUB kota CCTV",
                "NTMC Polri live streaming",
                "NTMC POLRI CCTV",
            ],
            "market": [
                "suasana pasar tradisional",
                "pasar ramai",
                "aktivitas jual beli pasar",
                "pasar pagi hari",
                "CCTV pasar",
                "suasana pasar tradisional Indonesia",
            ],
            "street": [
                "suasana jalan kota",
                "aktivitas warga kota",
                "jalan protokol",
                "suasana terminal",
                "CCTV alun alun",
                "keramaian kota",
                "jalan raya Indonesia",
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

    def get_channel_videos(self, channel_url: str, max_results: int = 30) -> list:
        """Fetch videos/streams from a specific YouTube channel."""
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--playlist-end", str(max_results),
            channel_url,
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
        except Exception as e:
            print(f"    Channel error: {e}")
            return []

    def download_video(self, video: dict, output_path: Path, is_live: bool = False) -> bool:
        video_id = video.get("id", "")
        if video_id in self.downloaded_ids:
            return False

        url = f"https://www.youtube.com/watch?v={video_id}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_path = str(Path(__file__).resolve().parent.parent / "bin" / "ffmpeg.exe")

        # Use --download-sections for live streams to get a short clip
        # For non-live, use --max-filesize to limit downloads
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "best[height<=480][ext=mp4]",
            "-o", str(output_path.parent / f"{video_id}.%(ext)s"),
            "--no-warnings",
            "--no-playlist",
            "--no-live-from-start",
            "--ffmpeg-location", ffmpeg_path,
        ]
        if is_live:
            cmd += ["--download-sections", "*0-30"]  # 30 second clip
        else:
            cmd += ["--max-filesize", "30M"]
        cmd.append(url)

        timeout = 120 if is_live else 120

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            downloaded = list(output_path.parent.glob(f"{video_id}.*"))
            if downloaded:
                self.downloaded_ids.add(video_id)
                return True
        except subprocess.TimeoutExpired:
            downloaded = list(output_path.parent.glob(f"{video_id}.*"))
            if downloaded:
                file_size = downloaded[0].stat().st_size
                if file_size > 500_000:
                    self.downloaded_ids.add(video_id)
                    return True
        except Exception:
            pass
        return False

    def is_valid_video(
        self,
        video: dict,
        min_dur: int,
        max_dur: int,
        label: int = 1,
    ) -> tuple:
        title = video.get("title", "")
        channel = video.get("channel", "") or video.get("uploader", "") or ""
        duration = video.get("duration")  # None for live streams

        # Duration check: None = live stream (only accepted for normal)
        if duration is not None:
            if duration < min_dur or duration > max_dur:
                return False, f"duration {duration}s (not in {min_dur}-{max_dur})"
        else:
            # Live stream — only allow for normal (label=0)
            if label != 0:
                return False, "live stream (rejected for anomaly)"

        if channel_matches_exclude(channel):
            return False, f"news channel: {channel[:40]}"

        if title_matches_exclude(title):
            return False, f"news title: {title[:60]}"

        if label == 0 and title_matches_exclude(title, NORMAL_EXCLUDE_TITLE_PATTERNS):
            return False, f"non-CCTV content: {title[:60]}"

        return True, "ok"

    def clean_title(self, title: str) -> str:
        title = re.sub(r'[^\w\s\-]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()[:80]
        return title

    def download_and_record(
        self,
        video: dict,
        category_dir: Path,
        category: str,
        label: int,
        keyword: str,
        is_live: bool = False,
    ) -> bool:
        title = self.clean_title(video.get("title", ""))
        print(f"    Downloading: {title[:50]}...", end=" ")

        success = self.download_video(video, category_dir, is_live=is_live)
        if success:
            ch = (
                video.get("channel") or
                video.get("uploader") or
                video.get("playlist_channel") or
                ""
            )
            self.metadata.append({
                "file_path": str(category_dir / f"{video.get('id', '')}.mp4"),
                "video_id": video.get("id", ""),
                "title": title,
                "category": category,
                "label": label,
                "source": "YouTube",
                "channel": ch,
                "duration_sec": video.get("duration", 0),
                "keyword": keyword,
                "downloaded_at": datetime.now().isoformat(),
            })
            print("OK")
            return True
        else:
            print("SKIP")
            return False

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
                    ok, reason = self.is_valid_video(v, min_dur, max_dur, label=label)
                    if ok:
                        valid.append(v)

                print(f"    Found {len(videos)}, valid after filter: {len(valid)}")

                if valid:
                    print(f"    Sample titles:")
                    for v in valid[:5]:
                        dur = v.get("duration", "LIVE" if v.get("duration") is None else v.get("duration", 0))
                        print(f"      - {self.clean_title(v.get('title',''))[:60]} ({dur}s)")

                for v in valid:
                    if total_downloaded >= max_per_keyword * len(kw_list):
                        break
                    is_live = v.get("duration") is None
                    if self.download_and_record(v, category_dir, category, label, keyword, is_live=is_live):
                        total_downloaded += 1
                    time.sleep(1.5)

        return total_downloaded

    def scrape_channel_streams(self, label: int = 0, max_per_channel: int = 20):
        """Scrape known CCTV channel URLs for normal data."""
        total_downloaded = 0
        category_dir = self.output_dir / "normal"
        category_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [CCTV] Fetching streams from {len(KNOWN_CCTV_CHANNELS)} known channels...")
        for channel_url in KNOWN_CCTV_CHANNELS:
            if total_downloaded >= max_per_channel * 3:
                break
            print(f"    Channel: {channel_url}")
            channel_videos = self.get_channel_videos(channel_url, max_results=30)

            # The flat-playlist output from channel streams may not have
            # "channel" field directly; fall back to playlist_channel
            for v in channel_videos:
                ch = v.get("channel") or v.get("playlist_channel") or "CCTV Channel"
                v["channel"] = ch
                v["uploader"] = ch

            valid = []
            for v in channel_videos:
                ok, reason = self.is_valid_video(v, min_dur=10, max_dur=600, label=0)
                if ok:
                    valid.append(v)

            print(f"    Found {len(channel_videos)}, valid: {len(valid)}")
            for v in valid[:5]:
                print(f"      - {self.clean_title(v.get('title',''))[:60]}")

            for v in valid:
                if total_downloaded >= max_per_channel * 3:
                    break
                is_live = v.get("duration") is None
                if self.download_and_record(v, category_dir, "traffic", 0, "cctv channel", is_live=is_live):
                    total_downloaded += 1
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
        print("YOUTUBE SCRAPER V3 — DATA INDONESIA (LIVE CCTV SUPPORT)")
        print("=" * 60)
        print(f"Exclude patterns: {len(EXCLUDE_TITLE_PATTERNS)} title + {len(NORMAL_EXCLUDE_TITLE_PATTERNS)} normal-extra + {len(EXCLUDE_CHANNEL_PATTERNS)} channel")
        print(f"Known CCTV channels: {len(KNOWN_CCTV_CHANNELS)} channel URLs")
        print(f"Anomaly duration: 15-90s | Normal duration: 30-300s (live ok)")
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
        print(f"  Total normal (search): {normal_count}")

        print(f"\n  ---> Scraping KNOWN CCTV CHANNELS for normal data...")
        cctv_count = self.scrape_channel_streams(label=0)
        print(f"  Total normal (cctv): {cctv_count}")

        csv_path = self.save_metadata()

        print(f"\n{'='*60}")
        print(f"SCRAPING V3 COMPLETE")
        print(f"{'='*60}")
        print(f"  Anomaly: {anomaly_count}")
        print(f"  Normal:  {normal_count + cctv_count} (search:{normal_count} + cctv:{cctv_count})")
        print(f"  Total:   {anomaly_count + normal_count + cctv_count}")
        print(f"  Metadata: {csv_path}")
        print()

        return self.metadata, csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sample_videos/indonesia_v3")
    parser.add_argument("--max-per-keyword", type=int, default=20)
    args = parser.parse_args()

    scraper = YouTubeScraperV3(output_dir=args.output)
    scraper.run(max_per_keyword=args.max_per_keyword)
