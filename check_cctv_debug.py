import urllib.request, ssl, re, json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
ctx.set_ciphers('DEFAULT')

# Test a working camera vs failing one
tests = [
    ("TITIK NOL (known working)", "https://livepantau.semarangkota.go.id/dc23feef-8cbd-4f67-8c46-382c691d5f09/index.m3u8"),
    ("THAMRIN PANDANARAN (failing)", "https://livepantau.semarangkota.go.id/e910f4f2-a77d-4cf8-8341-aa2e6c6b65c9/index.m3u8"),
]

for label, url in tests:
    print(f"\n=== {label} ===")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        print(f"Status: {resp.status} {resp.reason}")
        data = resp.read(500)
        print(f"Is HLS: {data.startswith(b'#EXTM3U')}")
        print(f"Preview: {data[:150]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(f"Response headers: {dict(e.headers)}")
        body = e.read(500)
        print(f"Body: {body[:300]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

# Try to scrape Pantau Semar website for updated CCTV list
print("\n\n=== Scraping Pantau Semar website ===")
url = 'https://pantausemar.semarangkota.go.id/'
try:
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    })
    resp = urllib.request.urlopen(req, timeout=30, context=ctx)
    html = resp.read().decode('utf-8', errors='replace')
    print(f"Status: {resp.status}, HTML size: {len(html)} bytes")
    
    # Find all UUIDs
    uuids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', html)
    print(f"\nFound {len(uuids)} UUIDs in page:")
    for u in uuids:
        print(f"  https://livepantau.semarangkota.go.id/{u}/index.m3u8")
    
    # Look for m3u8 in scripts/data
    m3u8_pattern = re.findall(r'https?://[^"\'\\s<>]+\.m3u8[^"\'\\s<>]*', html)
    print(f"\nFound {len(m3u8_pattern)} m3u8 URLs in page:")
    for m in m3u8_pattern[:20]:
        print(f"  {m}")
    
    # Look for JSON data
    json_pattern = re.findall(r'\[[^]]+stream[^]]+\]', html, re.IGNORECASE)
    print(f"\nJSON-like stream data: {len(json_pattern)} matches")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
