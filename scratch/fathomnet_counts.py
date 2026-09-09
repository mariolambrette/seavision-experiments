import json, os, statistics as st
from concurrent.futures import ThreadPoolExecutor
import requests
from fathomnet.api import boundingboxes, images, darwincore
from fathomnet.dto import BoundingBoxConstraintsDTO, GeoImageConstraints

OUT = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_counts.json"
VER = dict(includeVerified=True, includeUnverified=False)
res = {}

print("totals...")
res["boxes_all"] = boundingboxes.count_all().count
res["images_all"] = images.count_all().count
res["boxes_verified"] = boundingboxes.count(
    BoundingBoxConstraintsDTO(**VER)).count
print(res)

print("per-concept (unfiltered)...")
bc = boundingboxes.count_total_by_concept()
res["by_concept"] = sorted(
    [{"concept": c.concept, "boxes": c.count} for c in bc],
    key=lambda d: -d["boxes"])
print(f"  {len(res['by_concept'])} concepts; "
      f"top 10: {[(d['concept'], d['boxes']) for d in res['by_concept'][:10]]}")

print("institutions...")
insts = darwincore.find_owner_institution_codes()
def inst_count(code):
    try:
        n = boundingboxes.count(BoundingBoxConstraintsDTO(
            ownerInstitutionCodes=[code], **VER)).count
    except Exception as e:
        n = f"ERR {e}"
    return {"institution": code, "boxes_verified": n}
with ThreadPoolExecutor(max_workers=8) as ex:
    res["by_institution"] = list(ex.map(inst_count, insts))
res["by_institution"].sort(
    key=lambda d: -(d["boxes_verified"] if isinstance(d["boxes_verified"], int) else -1))
print(f"  {len(insts)} institutions; top 10:")
for d in res["by_institution"][:10]:
    print("   ", d)

print("sampling 300 verified images for size...")
sample = images.find(GeoImageConstraints(limit=300, **VER))
res["sample_n"] = len(sample)

def head(im):
    try:
        r = requests.head(im.url, timeout=20, allow_redirects=True)
        return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0
with ThreadPoolExecutor(max_workers=8) as ex:
    sizes = [s for s in ex.map(head, sample) if s > 0]

boxes, dims = [], []
for im in sample:
    if im.width and im.height:
        dims.append((im.width, im.height))
    for b in (im.boundingBoxes or []):
        if b.width and b.height:
            boxes.append(min(b.width, b.height))

def pct(v, p):
    v = sorted(v); return v[int(p * (len(v) - 1))] if v else None

res["frame_bytes"] = {
    "n": len(sizes), "median": pct(sizes, .5),
    "p10": pct(sizes, .1), "p90": pct(sizes, .9),
    "mean": int(st.mean(sizes)) if sizes else None}
res["frame_dims_median"] = (pct([d[0] for d in dims], .5),
                            pct([d[1] for d in dims], .5))
res["box_short_side_px"] = {
    "n": len(boxes), "median": pct(boxes, .5),
    "p10": pct(boxes, .1), "p90": pct(boxes, .9),
    "frac_under_64": round(sum(b < 64 for b in boxes) / len(boxes), 4) if boxes else None,
    "boxes_per_image": round(len(boxes) / max(len(sample), 1), 2)}

if res["frame_bytes"]["mean"]:
    gb = res["frame_bytes"]["mean"] * res["images_all"] / 1e9
    res["projected_frames_gb_all_images"] = round(gb, 1)

print(json.dumps({k: v for k, v in res.items()
                  if k not in ("by_concept", "by_institution")}, indent=2))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("written:", OUT)
