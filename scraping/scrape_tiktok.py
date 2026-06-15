import asyncio
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "sample_videos" / "tiktok"
ANOMALY_DIR = OUTPUT_DIR / "anomaly"
NORMAL_DIR = OUTPUT_DIR / "normal"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
MAX_VIDEOS_PER_KEYWORD = 20
MAX_TOTAL_VIDEOS = 80

EXCLUDE_AUTHOR_PATTERNS = [
    r"tvone", r"tvOne", r"buser", r"indosiar", r"kompas",
    r"liputan6", r"tribun", r"detikcom", r"kumparan",
    r"polres", r"polsek", r"polresta", r"polrestabes",
    r"infopublik", r"berita", r"news", r"official",
    r"radio", r"polda", r"pemkot", r"dishub",
]

EXCLUDE_DESC_PATTERNS = [
    r"berita", r"news", r"tvone", r"indosiar", r"kompas",
    r"liputan6", r"tribunnews", r"detikcom",
    r"lagu", r"lirik", r"music", r"musik",
    r"tutorial", r"belajar", r"review",
    r"promosi", r"iklan", r"endorse",
]

ANOMALY_KEYWORDS = [
    "tawuran",
    "perampokan",
    "begal",
    "jambret",
    "kerusuhan",
    "demo ricuh",
    "pengeroyokan",
    "kejahatan jalanan",
    "kriminal",
    "pencurian motor",
    "perkelahian",
    "bentrokan",
    "copet",
]

NORMAL_KEYWORDS = [
    "cctv lalu lintas",
    "suasana jalan",
    "aktivitas warga",
    "pasar tradisional",
    "jalan raya",
    "keramaian kota",
]


def author_matches_exclude(author: str) -> bool:
    for pat in EXCLUDE_AUTHOR_PATTERNS:
        if re.search(pat, author, re.IGNORECASE):
            return True
    return False


def desc_matches_exclude(desc: str) -> bool:
    for pat in EXCLUDE_DESC_PATTERNS:
        if re.search(pat, desc, re.IGNORECASE):
            return True
    return False


def clean_text(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[^\w\s\-.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()[:max_len]
    return text


class TikTokScraper:
    def __init__(self):
        self.downloaded_ids = set()
        self.metadata = []
        self._load_existing()

    def _load_existing(self):
        if METADATA_FILE.exists():
            with open(METADATA_FILE, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.downloaded_ids.add(row.get("video_id", ""))

    async def process_search(
        self, api, keyword: str, label: int, max_count: int
    ) -> int:
        """Search TikTok and download matching videos. Returns count downloaded."""
        downloaded = 0
        obj_type = "item"

        try:
            async for item in api.search.search_type(
                search_term=keyword, obj_type=obj_type, count=max_count * 2
            ):
                if downloaded >= max_count:
                    break

                item_dict = item.as_dict
                vid_id = str(item_dict.get("id", ""))
                if vid_id in self.downloaded_ids:
                    continue

                author = item_dict.get("author", {}).get("uniqueId", "")
                desc = item_dict.get("desc", "") or ""
                create_time = item_dict.get("createTime", 0)
                duration = item_dict.get("video", {}).get("duration", 0)
                play_count = item_dict.get("stats", {}).get("playCount", 0)

                if author_matches_exclude(author):
                    continue

                if label == 1:
                    if duration < 15 or duration > 90:
                        continue
                    if desc_matches_exclude(desc):
                        continue
                else:
                    if duration < 30 or duration > 300:
                        continue

                out_dir = ANOMALY_DIR if label == 1 else NORMAL_DIR
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{vid_id}.mp4"

                safe_author = author.encode('ascii', 'ignore').decode()
                print(f"    Downloading @{safe_author} ({duration}s)...", end=" ")
                sys.stdout.flush()
                try:
                    video_bytes = await item.bytes()
                    with open(out_path, "wb") as f:
                        f.write(video_bytes)
                except Exception as e:
                    err = str(e).encode('ascii', 'ignore').decode()
                    print(f"FAILED: {err}")
                    continue

                self.downloaded_ids.add(vid_id)
                downloaded += 1

                clean_desc = clean_text(desc)
                self.metadata.append({
                    "file_path": str(out_path.relative_to(BASE_DIR)),
                    "video_id": vid_id,
                    "title": clean_desc or vid_id,
                    "category": "anomaly" if label == 1 else "normal",
                    "label": label,
                    "source": "TikTok",
                    "channel": author,
                    "duration_sec": duration,
                    "keyword": keyword,
                    "play_count": play_count,
                    "downloaded_at": datetime.now().isoformat(),
                })
                print(f"OK ({len(video_bytes) / 1024:.0f} KB)")

        except Exception as e:
            print(f"    Search error for '{keyword}': {type(e).__name__}: {e}")

        return downloaded

    async def run(self):
        from TikTokApi import TikTokApi

        print("Initializing TikTok API (this may take a moment)...")
        api = TikTokApi()
        try:
            await api.create_sessions(headless=True, num_sessions=1)
            print("Sessions created.\n")

            total_downloaded = 0

            # --- Scrape anomaly ---
            print("=" * 50)
            print("SCRAPING ANOMALY VIDEOS")
            print("=" * 50)
            for keyword in ANOMALY_KEYWORDS:
                if total_downloaded >= MAX_TOTAL_VIDEOS:
                    break
                remaining = MAX_TOTAL_VIDEOS - total_downloaded
                count = min(MAX_VIDEOS_PER_KEYWORD, remaining)
                print(f"\nSearching: '{keyword}' (target: {count})")
                dl = await self.process_search(api, keyword, label=1, max_count=count)
                total_downloaded += dl
                print(f"  Downloaded: {dl} (total: {total_downloaded})")
                await asyncio.sleep(2)

            # --- Save metadata ---
            self._save_metadata()
            print(f"\n{'=' * 50}")
            print(f"TOTAL DOWNLOADED: {total_downloaded} anomaly videos")
            if total_downloaded == 0:
                print("(!) No videos downloaded. TikTok may be rate-limiting.")
            print(f"Metadata saved to: {METADATA_FILE}")

        finally:
            await api.close_sessions()

    def _save_metadata(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        exists = METADATA_FILE.exists()
        with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "file_path", "video_id", "title", "category", "label",
                "source", "channel", "duration_sec", "keyword",
                "play_count", "downloaded_at",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            for row in self.metadata:
                writer.writerow(row)


if __name__ == "__main__":
    asyncio.run(TikTokScraper().run())
