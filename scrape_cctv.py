import urllib.request, ssl, re, json, os, time, concurrent.futures

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print("Fetching Pantau Semar website...")
req = urllib.request.Request('https://pantausemar.semarangkota.go.id/', headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
})
resp = urllib.request.urlopen(req, timeout=30, context=ctx)
html = resp.read().decode('utf-8', errors='replace')
print(f"Got HTML: {len(html)} bytes")

# Extract all UUIDs
all_uuids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', html)
print(f"Total UUIDs found: {len(all_uuids)}")

# De-duplicate while preserving order
seen = set()
unique_uuids = []
for u in all_uuids:
    if u not in seen:
        seen.add(u)
        unique_uuids.append(u)
print(f"Unique UUIDs: {len(unique_uuids)}")

# Find camera names in the HTML - look for structured data
# Try to find name-uuid mapping
name_pattern = re.findall(r'([A-Za-z0-9\s\-/]+(?:360|Wide|PTZ|Mall|2|01|02))', html)
print(f"Potential camera names found in HTML")

# Look for JS variable assignments or JSON data
js_data = re.findall(r'var\s+\w+\s*=\s*(\[[^\]]+\])', html)
print(f"JS array assignments: {len(js_data)}")

# Look for data in script tags
script_pattern = re.findall(r'<script[^>]*src="([^"]+)"', html)
print(f"External scripts: {len(script_pattern)}")
for s in script_pattern[:20]:
    print(f"  {s}")

# Try to find the main app JS file
app_js_urls = [s for s in script_pattern if 'app' in s.lower() or 'main' in s.lower() or 'bundle' in s.lower()]
print(f"\nApp JS candidates: {app_js_urls}")
