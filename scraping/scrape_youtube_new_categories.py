import sys
sys.path.insert(0, ".")
from scraping.scrape_youtube_v4 import YouTubeScraperV4

scraper = YouTubeScraperV4(output_dir="sample_videos/indonesia_v6")
total = 0

# Run per-category to avoid timeout issues
categories = [
    ("[1/5] DEMO RUSUH", scraper.demo_violent_keywords, 1, 12),
    ("[2/5] DEMO RUSUH CCTV/DRONE", {"demo_rusuh_cctv": scraper.demo_violent_keywords.get("demo_rusuh_cctv", [])}, 1, 15),
    ("[3/5] DEMO DAMAI", scraper.demo_peaceful_keywords, 0, 12),
    ("[4/5] DEMO DAMAI CCTV/DRONE", {"demo_damai_cctv": scraper.demo_peaceful_keywords.get("demo_damai_cctv", [])}, 0, 15),
    ("[5/5] HARD NEGATIVE", scraper.hard_negative_keywords, 0, 12),
]

for name, keywords, label, max_kw in categories:
    print(f"\n{'='*60}")
    print(name)
    print(f"{'='*60}")
    c = scraper.scrape_category(keywords, label=label, max_per_keyword=max_kw)
    print(f"  Downloaded: {c}")
    total += c

csv_path = scraper.save_metadata()
print(f"\nTotal new YouTube downloads: {total}")
print(f"Metadata: {csv_path}")
