"""
Download sajam weapon images from Bing Image Search.
More reliable than Google — avoids bot detection.
"""
from pathlib import Path
from bing_image_downloader import downloader

OUTPUT_DIR = Path("dataset/sajam_google_images")

QUERIES = {
    "celurit": [
        "celurit Madura panjang",
        "celurit tawuran",
        "clurit Madura tradisional",
        "celurit sabit panjang",
        "clurit lengkung tradisional",
        "celurit pandai besi Madura",
        "celurit baja tradisional",
    ],
    "golok": [
        "golok Betawi",
        "golok Banten besar",
        "golok pandai besi Indonesia",
        "golok panjang tradisional",
        "golok clurit Indonesia",
    ],
    "kapak": [
        "kapak perang tradisional Indonesia",
        "kapak kampak tradisional",
        "kapak besar Indonesia",
        "kapak pandai besi",
    ],
    "pedang": [
        "pedang tradisional Indonesia",
        "pedang keris panjang",
        "pedang lurus tradisional",
        "pedang samurai Indonesia",
    ],
    "pisau": [
        "pisau dapur besar Indonesia",
        "pisau belati tradisional",
        "pisau lipat besar",
        "pisau kampak tradisional",
    ],
    "sabit": [
        "sabit petani Indonesia",
        "sabit panjang rumput",
        "sabit clurit pertanian",
        "sabit tradisional sawah",
    ],
}


def run():
    total_expected = 0

    for weapon, kw_list in QUERIES.items():
        weapon_dir = OUTPUT_DIR / weapon
        weapon_dir.mkdir(parents=True, exist_ok=True)

        for kw in kw_list:
            print(f"\n  [{weapon}] Searching: '{kw}'...")
            try:
                downloader.download(
                    kw,
                    limit=30,
                    output_dir=str(OUTPUT_DIR),
                    adult_filter_off=False,
                    force_replace=False,
                    timeout=30,
                )
                print(f"    Done.")
            except Exception as e:
                print(f"    Error: {e}")
            total_expected += 30

    print(f"\n{'='*50}")
    print(f"BING IMAGE SCRAPING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total keywords: {total_expected} expected images")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
