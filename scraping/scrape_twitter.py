import subprocess
import json
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "sample_videos" / "twitter"
ANOMALY_DIR = OUTPUT_DIR / "anomaly"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

MAX_VIDEOS_PER_ACCOUNT = 30
MAX_TOTAL_VIDEOS = 60

EXCLUDE_AUTHOR_PATTERNS = [
    r"kompas", r"tribun", r"detikcom", r"liputan6",
    r"tvone", r"indosiar", r"berita", r"news",
]

EXCLUDE_TEXT_PATTERNS = [
    r"berita", r"news", r"lagu", r"lirik", r"music",
    r"tutorial", r"belajar", r"review",
]

KNOWN_ACCOUNTS = [
    "https://x.com/TMCPoldaMetro",
    "https://x.com/DivHumas_Polri",
]

TWEET_URLS_FILE = BASE_DIR / "scraping" / "twitter_urls.txt"

SEARCH_KEYWORDS = [
    "tawuran",
    "perampokan",
    "demo ricuh",
    "kerusuhan",
    "begal",
]


def author_matches_exclude(author: str) -> bool:
    for pat in EXCLUDE_AUTHOR_PATTERNS:
        if re.search(pat, author, re.IGNORECASE):
            return True
    return False


def text_matches_exclude(text: str) -> bool:
    for pat in EXCLUDE_TEXT_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def clean_text(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[^\w\s\-.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()[:max_len]
    return text


class TwitterScraper:
    def __init__(self):
        self.downloaded_ids = set()
        self.metadata = []
        self._load_existing()

    def _load_existing(self):
        if METADATA_FILE.exists():
            with open(METADATA_FILE, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.downloaded_ids.add(row.get("video_id", ""))

    def scrape_account(self, account_url: str) -> int:
        """Scrape all video tweets from a Twitter account."""
        downloaded = 0

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--playlist-end", str(MAX_VIDEOS_PER_ACCOUNT),
            account_url,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    video_info = json.loads(line)
                except json.JSONDecodeError:
                    continue

                tweet_id = video_info.get("id", "")
                if tweet_id in self.downloaded_ids:
                    continue

                title = video_info.get("title", "") or ""
                url = video_info.get("webpage_url", "")
                webpage_url = video_info.get("webpage_url", url)

                author = ""
                if "/status/" in webpage_url:
                    author = webpage_url.split("/")[3]
                if not author and account_url:
                    author = account_url.rstrip("/").split("/")[-1]

                if author_matches_exclude(author):
                    continue
                if text_matches_exclude(title):
                    continue

                out_path = ANOMALY_DIR / f"{tweet_id.replace('/', '_')}.mp4"
                success = self._download_tweet(webpage_url, out_path)
                if success:
                    downloaded += 1
                    self.downloaded_ids.add(tweet_id)
                    self.metadata.append({
                        "file_path": str(out_path.relative_to(BASE_DIR)),
                        "video_id": tweet_id,
                        "title": clean_text(title) or tweet_id,
                        "category": "anomaly",
                        "label": 1,
                        "source": "Twitter",
                        "channel": author,
                        "duration_sec": "",
                        "keyword": account_url,
                        "downloaded_at": datetime.now().isoformat(),
                    })
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"    Error scraping {account_url}: {e}")

        return downloaded

    def _download_tweet(self, tweet_url: str, output_path: Path) -> bool:
        """Download video from a single tweet."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "best[height<=480][ext=mp4]",
            "-o", str(output_path.parent / f"{output_path.stem}.%(ext)s"),
            "--no-warnings",
            "--no-playlist",
            "--max-filesize", "30M",
            "--socket-timeout", "10",
            tweet_url,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if output_path.exists():
                return True
            for f in output_path.parent.glob(f"{output_path.stem}.*"):
                if f.suffix in (".mp4", ".webm", ".mkv"):
                    return True
            return False
        except Exception:
            return False

    def download_from_url_file(self) -> int:
        """Download tweets from URL list file."""
        if not TWEET_URLS_FILE.exists():
            print(f"  URL file not found: {TWEET_URLS_FILE}")
            return 0

        downloaded = 0
        with open(TWEET_URLS_FILE) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        print(f"  Found {len(urls)} URLs in file")
        for url in urls:
            if downloaded >= MAX_TOTAL_VIDEOS:
                break
            tweet_id_match = re.search(r'/status/(\d+)', url)
            if not tweet_id_match:
                continue
            tweet_id = tweet_id_match.group(1)
            if tweet_id in self.downloaded_ids:
                continue

            author_match = re.search(r'(?:twitter|x)\.com/(\w+)', url)
            author = author_match.group(1) if author_match else ""

            print(f"    Downloading tweet {tweet_id}...", end=" ")
            sys.stdout.flush()
            out_path = ANOMALY_DIR / f"tw_{tweet_id}.mp4"
            success = self._download_tweet(url, out_path)
            if success:
                downloaded += 1
                self.downloaded_ids.add(tweet_id)
                self.metadata.append({
                    "file_path": str(out_path.relative_to(BASE_DIR)),
                    "video_id": tweet_id,
                    "title": tweet_id,
                    "category": "anomaly",
                    "label": 1,
                    "source": "Twitter",
                    "channel": author,
                    "duration_sec": "",
                    "keyword": url,
                    "downloaded_at": datetime.now().isoformat(),
                })
                print("OK")
            else:
                print("FAILED")

        return downloaded

    def run(self):
        total_downloaded = 0

        print("=" * 50)
        print("TWITTER/X VIDEO SCRAPER")
        print("=" * 50)

        # Step 1: Download from URL file
        print("\n[1/2] Downloading from URL file...")
        dl = self.download_from_url_file()
        total_downloaded += dl
        print(f"  Downloaded: {dl}")

        # Step 2: Scrape known accounts
        print("\n[2/2] Scraping known accounts...")
        for i, account_url in enumerate(KNOWN_ACCOUNTS):
            if total_downloaded >= MAX_TOTAL_VIDEOS:
                break
            remaining = MAX_TOTAL_VIDEOS - total_downloaded
            print(f"\n  Scraping: {account_url}")
            dl = self.scrape_account(account_url)
            total_downloaded += dl
            print(f"  Downloaded: {dl} (total: {total_downloaded})")

        # Save metadata
        self._save_metadata()

        print(f"\n{'=' * 50}")
        print(f"TOTAL DOWNLOADED: {total_downloaded}")
        print(f"Metadata: {METADATA_FILE}")
        if TWEET_URLS_FILE.exists():
            print(f"\nTip: Add more tweet URLs to:")
            print(f"  {TWEET_URLS_FILE}")
            print(f"  (one URL per line, lines starting with # are ignored)")
        print(f"\nNote: Twitter search requires manual URL collection.")
        print(f"To find tweets, search on Google:")
        print(f"  site:x.com tawuran")

    def _save_metadata(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        exists = METADATA_FILE.exists()
        with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "file_path", "video_id", "title", "category", "label",
                "source", "channel", "duration_sec", "keyword",
                "downloaded_at",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            for row in self.metadata:
                writer.writerow(row)


if __name__ == "__main__":
    TwitterScraper().run()
