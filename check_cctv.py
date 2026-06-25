import urllib.request, ssl, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

from cctv_sources import CCTV_SEMARANG

total = len(CCTV_SEMARANG)
online = 0
offline = 0

print(f"Mengecek koneksi {total} CCTV Semarang...\n")

for name, owner, url in CCTV_SEMARANG:
    try:
        req = urllib.request.Request(url, method='GET', headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        data = resp.read(500)
        is_hls = data.startswith(b'#EXTM3U')
        if is_hls:
            online += 1
            print(f"  [+] {name}")
        else:
            offline += 1
            print(f"  [!] {name} - BAD RESPONSE")
    except Exception as e:
        offline += 1
        err = type(e).__name__
        print(f"  [x] {name} - {err}")

print(f"\n--- SUMMARY ---")
print(f"Total CCTV: {total}")
print(f"Online: {online}")
print(f"Offline: {offline}")
