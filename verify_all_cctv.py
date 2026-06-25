import urllib.request, ssl, json, time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Load all streams from the raw data
with open('cctv_raw_data.json', 'r') as f:
    all_streams = json.load(f)

print(f"Testing {len(all_streams)} CCTV streams...\n")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
    'Origin': 'https://pantausemar.semarangkota.go.id',
    'Referer': 'https://pantausemar.semarangkota.go.id/',
}

working = []
not_working = []

for i, (name, owner, url, status) in enumerate(all_streams):
    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        data = resp.read(500)
        is_hls = data.startswith(b'#EXTM3U')
        if is_hls:
            working.append((name, owner, url))
            print(f"  OK  [{i+1:2d}/{len(all_streams)}] {name}")
        else:
            not_working.append((name, owner, url, 'BAD-RESPONSE'))
            print(f"  BAD [{i+1:2d}/{len(all_streams)}] {name} - not HLS")
    except Exception as e:
        err = type(e).__name__
        code = getattr(e, 'code', '?') if hasattr(e, 'code') else '?'
        not_working.append((name, owner, url, f'{err} {code}'))
        print(f"  FAIL [{i+1:2d}/{len(all_streams)}] {name} - {err} {code}")

print(f"\n=== RESULTS ===")
print(f"Working: {len(working)}/{len(all_streams)}")
print(f"Not working: {len(not_working)}/{len(all_streams)}")

if working:
    print(f"\n=== WORKING CCTV ({len(working)}) ===")
    for name, owner, url in working:
        print(f"  {name:35s} | {url}")

if not_working:
    print(f"\n=== NOT WORKING ({len(not_working)}) ===")
    for name, owner, url, reason in not_working:
        print(f"  {name:35s} | {reason}")

# Save results
with open('cctv_verified.json', 'w') as f:
    json.dump({'working': working, 'not_working': not_working}, f, indent=2)
print(f"\nResults saved to cctv_verified.json")
