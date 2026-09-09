import json, os, statistics as st
from concurrent.futures import ThreadPoolExecutor
import requests
from fathomnet.api import boundingboxes, images, darwincore
from fathomnet.dto import BoundingBoxConstraintsDTO, GeoImageConstraints

OUT = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_sizes.json"
N = 120

def pct(v, p):
    v = sorted(v); return v[int(p * (len(v) - 1))] if v else None

def head(url):
    try:
        r = requests.head(url, timeout=20, allow_redirects=True)
        return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0

rows, total_bytes = [], 0
for code in darwincore.find_owner_institution_codes():
    try:
        nbox = boundingboxes.count(BoundingBoxConstraintsDTO(
            ownerInstitutionCodes=[code])).count
        sample = images.find(GeoImageConstraints(
            ownerInstitutionCodes=[code], limit=N))
    except Exception as e:
        rows.append({"institution": code, "error": str(e)}); continue
    if not sample:
        rows.append({"institution": code, "boxes": nbox, "sample": 0}); continue

    with ThreadPoolExecutor(max_workers=8) as ex:
        sizes = [s for s in ex.map(head, [im.url for im in sample]) if s > 0]
    bpi = st.mean([len(im.boundingBoxes or []) for im in sample]) or 1
    exts = {}
    for im in sample:
        e = os.path.splitext(im.url.split("?")[0])[1].lower()
        exts[e] = exts.get(e, 0) + 1
    med = pct(sizes, .5) or 0
    est_imgs = nbox / bpi
    est_gb = med * est_imgs / 1e9
    total_bytes += med * est_imgs
    rows.append({"institution": code, "boxes": nbox,
                 "boxes_per_image": round(bpi, 2),
                 "est_images": int(est_imgs), "sample": len(sizes),
                 "median_bytes": med, "p90_bytes": pct(sizes, .9),
                 "ext": exts, "est_gb": round(est_gb, 1)})

rows.sort(key=lambda r: -r.get("est_gb", 0))
for r in rows:
    print(r)
print("\nESTIMATED TOTAL GB:", round(total_bytes / 1e9, 1))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print("written:", OUT)
