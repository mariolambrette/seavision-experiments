import io, json, random, statistics as st
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from fathomnet.api import images
from fathomnet.dto import GeoImageConstraints

OUT = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_manifest.json"
PNG = ["MBARI", "Schmidt Ocean Institute", "Joost Daniels",
       "Universidad de Costa Rica and Schmidt Ocean Institute", "OET"]
CAP, PAGE, SAMPLE, ENC = 50, 500, 400, 15
random.seed(0)

def enumerate_inst(code):
    out, off = [], 0
    while True:
        page = images.find(GeoImageConstraints(
            ownerInstitutionCodes=[code], limit=PAGE, offset=off))
        if not page:
            break
        for im in page:
            out.append((im.uuid, im.url,
                        [b.concept for b in (im.boundingBoxes or []) if b.concept]))
        off += len(page)
        if len(page) < PAGE:
            break
        if off % 5000 == 0:
            print(f"    {code}: {off}")
    return out

inv = {}
for code in PNG:
    print(f"enumerating {code}...")
    inv[code] = enumerate_inst(code)
    print(f"  {len(inv[code]):,} images")

allimgs = [(c, u, url, cs) for c, v in inv.items() for (u, url, cs) in v]
print(f"\ntotal images in PNG block: {len(allimgs):,}")

freq = {}
for _, _, _, cs in allimgs:
    for c in cs:
        freq[c] = freq.get(c, 0) + 1
print(f"concepts: {len(freq):,}")

# greedy cap: rarest-concept-first so thin classes survive
order = sorted(allimgs, key=lambda r: min([freq[c] for c in r[3]], default=10**9))
quota, kept = {}, []
for row in order:
    cs = row[3]
    if not cs or any(quota.get(c, 0) < CAP for c in cs):
        kept.append(row)
        for c in cs:
            quota[c] = quota.get(c, 0) + 1
kept_by_inst = {}
for c, u, url, cs in kept:
    kept_by_inst[c] = kept_by_inst.get(c, 0) + 1
print(f"kept under cap {CAP}: {len(kept):,} images "
      f"({len(kept)/len(allimgs):.1%})")
print("  by institution:", kept_by_inst)

def head(url):
    try:
        r = requests.head(url, timeout=20, allow_redirects=True)
        return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0

def reencode(url):
    try:
        raw = requests.get(url, timeout=60).content
        buf = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert("RGB").save(
            buf, "JPEG", quality=95, optimize=True)
        return len(raw), buf.tell()
    except Exception:
        return None

report, tot = {}, {"unc_raw": 0, "unc_jpg": 0, "cap_raw": 0, "cap_jpg": 0}
for code in PNG:
    rows = inv[code]
    if not rows:
        continue
    samp = random.sample(rows, min(SAMPLE, len(rows)))
    with ThreadPoolExecutor(max_workers=12) as ex:
        sizes = [s for s in ex.map(head, [r[1] for r in samp]) if s > 0]
    mean_raw = st.mean(sizes) if sizes else 0

    esamp = random.sample(rows, min(ENC, len(rows)))
    with ThreadPoolExecutor(max_workers=4) as ex:
        pairs = [p for p in ex.map(reencode, [r[1] for r in esamp]) if p]
    ratio = (sum(b for _, b in pairs) / sum(a for a, _ in pairs)) if pairs else 1.0

    n_all, n_cap = len(rows), kept_by_inst.get(code, 0)
    report[code] = {
        "images_all": n_all, "images_capped": n_cap,
        "mean_bytes": int(mean_raw), "jpeg_ratio": round(ratio, 3),
        "uncapped_raw_gb": round(mean_raw * n_all / 1e9, 1),
        "uncapped_jpeg_gb": round(mean_raw * ratio * n_all / 1e9, 1),
        "capped_raw_gb": round(mean_raw * n_cap / 1e9, 1),
        "capped_jpeg_gb": round(mean_raw * ratio * n_cap / 1e9, 1)}
    tot["unc_raw"] += mean_raw * n_all
    tot["unc_jpg"] += mean_raw * ratio * n_all
    tot["cap_raw"] += mean_raw * n_cap
    tot["cap_jpg"] += mean_raw * ratio * n_cap
    print(f"\n{code}: {report[code]}")

print("\n=== PNG BLOCK TOTALS (GB) ===")
print(f"  uncapped, keep PNG   : {tot['unc_raw']/1e9:>8.1f}")
print(f"  uncapped, JPEG q95   : {tot['unc_jpg']/1e9:>8.1f}")
print(f"  capped {CAP}, keep PNG  : {tot['cap_raw']/1e9:>8.1f}")
print(f"  capped {CAP}, JPEG q95  : {tot['cap_jpg']/1e9:>8.1f}")
print("  (JPEG institutions add ~94 GB on top, uncapped)")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"report": report, "cap": CAP,
               "kept_uuids": [r[1] for r in kept]}, f)
print("\nwritten:", OUT)
