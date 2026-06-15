"""Test search for demo CCTV/drone footage on YouTube."""
import subprocess, json, sys

KEYWORDS = [
    "demo damai drone footage Indonesia",
    "demo rusuh CCTV footage",
    "demo dari ketinggian Indonesia",
    "demo tampak atas drone",
    "aksi demo udara 2024",
    "demo buruh damai udara",
    "demo mahasiswa drone view",
    "demo ricuh dari atas helikopter",
]

for kw in KEYWORDS:
    cmd = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--dump-json",
           "--no-warnings", "--playlist-end", "5", f"ytsearch5:{kw}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
        print(f"\n'{kw}' -> {len(lines)} results")
        for line in lines[:3]:
            v = json.loads(line)
            title = v.get("title", "?")[:60]
            dur = v.get("duration", "LIVE")
            ch = v.get("channel", "")[:25]
            print(f"  [{dur}s] {title} | {ch}")
    except Exception as e:
        print(f"\n'{kw}' -> ERROR: {e}")
