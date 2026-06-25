import urllib.request, ssl, json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

origin_hdrs = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Origin': 'http://localhost:8501',
    'Referer': 'http://localhost:8501/',
}

with open('cctv_verified.json', 'r') as f:
    data = json.load(f)

all_ok = True
for name, owner, master_url in data['working']:
    base = master_url[:master_url.rindex('/')+1]
    print(f'[{name:35s}] ', end='', flush=True)
    
    try:
        # 1. Master
        req = urllib.request.Request(master_url, headers=origin_hdrs)
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        master = resp.read().decode()
        
        sub_url = None
        for line in master.splitlines():
            line = line.strip()
            if line.endswith('.m3u8') and not line.startswith('#'):
                sub_url = line if '://' in line else base + line
                break
        
        if not sub_url:
            print('NO SUB-STREAM')
            all_ok = False
            continue
        
        # 2. Sub + segment test
        req = urllib.request.Request(sub_url, headers=origin_hdrs)
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        sub = resp.read().decode()
        
        seg_found = False
        for sl in sub.splitlines():
            sl = sl.strip()
            if sl.endswith('.mp4') and not sl.startswith('#'):
                seg_url = sl if '://' in sl else base + sl
                try:
                    req2 = urllib.request.Request(seg_url, headers=origin_hdrs)
                    resp2 = urllib.request.urlopen(req2, timeout=10, context=ctx)
                    d = resp2.read()
                    status = resp2.status
                    if status == 200 and len(d) > 1000:
                        print(f'OK ({len(d)//1024}KB)')
                    else:
                        print(f'PARTIAL (HTTP {status} {len(d)}B)')
                    seg_found = True
                except:
                    pass
                break
        
        if not seg_found:
            print('SEGMENT 404')
            all_ok = False
    except Exception as e:
        code = e.code if hasattr(e, 'code') else type(e).__name__
        print(f'FAIL {code}')
        all_ok = False

result = 'ALL OK' if all_ok else 'SOME FAILED'
print(f'\n\nOverall: {result}')
