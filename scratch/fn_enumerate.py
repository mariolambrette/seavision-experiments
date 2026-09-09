import csv, json, os
from collections import Counter, defaultdict
from fathomnet.api import images, darwincore
from fathomnet.dto import GeoImageConstraints

OUTDIR = r"D:\marineai\classification-experiments\sw\scratch\fn_manifest"
FINAL  = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_concept_final.csv"
PAGE = 500

action, aphia = {}, {}
for r in csv.DictReader(open(FINAL, encoding="utf-8")):
    action[r["concept"]] = r["action"]
    aphia[r["concept"]] = r["aphia_id"]
# marine organism: emit the crop, no category, flag the image
action["marine organism"] = "unlabelled"

def enumerate_inst(code):
    safe = "".join(c if c.isalnum() else "_" for c in code)
    path = os.path.join(OUTDIR, f"{safe}.jsonl")
    if os.path.exists(path):
        print(f"  {code}: already done, skipping")
        return path
    tmp, off, n_img, n_keep, n_box = path + ".part", 0, 0, 0, 0
    with open(tmp, "w", encoding="utf-8") as fh:
        while True:
            page = images.find(GeoImageConstraints(
                ownerInstitutionCodes=[code], limit=PAGE, offset=off))
            if not page:
                break
            for im in page:
                n_img += 1
                boxes = []
                for b in (im.boundingBoxes or []):
                    if b.rejected or not b.concept:
                        continue
                    act = action.get(b.concept)
                    if act not in ("class", "unlabelled"):
                        continue
                    boxes.append({
                        "uuid": b.uuid, "concept": b.concept, "action": act,
                        "aphia_id": aphia.get(b.concept) or None,
                        "x": b.x, "y": b.y, "w": b.width, "h": b.height,
                        "review_state": str(b.reviewState) if b.reviewState else None,
                        "reviewer": b.reviewer, "observer": b.observer,
                        "alt_concept": b.altConcept, "group_of": b.groupOf,
                        "occluded": b.occluded, "truncated": b.truncated,
                    })
                if not boxes:
                    continue
                n_keep += 1
                n_box += len(boxes)
                fh.write(json.dumps({
                    "uuid": im.uuid, "url": im.url, "institution": code,
                    "width": im.width, "height": im.height,
                    "lat": im.latitude, "lon": im.longitude,
                    "depth_m": im.depthMeters, "altitude": im.altitude,
                    "timestamp": im.timestamp, "imaging_type": im.imagingType,
                    "sha256": im.sha256, "media_type": im.mediaType,
                    "contributors_email": im.contributorsEmail,
                    "temperature_c": im.temperatureCelsius,
                    "salinity": im.salinity, "oxygen_ml_l": im.oxygenMlL,
                    "boxes": boxes,
                }) + "\n")
            off += len(page)
            if len(page) < PAGE:
                break
            if off % 10000 == 0:
                print(f"    {code}: {off:,} seen, {n_keep:,} kept")
    os.replace(tmp, path)
    print(f"  {code}: {n_img:,} images -> {n_keep:,} kept, {n_box:,} boxes")
    return path

paths = []
for code in darwincore.find_owner_institution_codes():
    print(f"enumerating {code}...")
    try:
        paths.append(enumerate_inst(code))
    except Exception as e:
        print(f"  FAILED {code}: {e}  (re-run to resume)")

print("\n=== MANIFEST SUMMARY ===")
tot_img = tot_box = 0
per_inst, per_action = Counter(), Counter()
for p in paths:
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        tot_img += 1
        tot_box += len(d["boxes"])
        per_inst[d["institution"]] += 1
        for b in d["boxes"]:
            per_action[b["action"]] += 1
print(f"images to download: {tot_img:,}")
print(f"boxes (crops)     : {tot_box:,}   {dict(per_action)}")
print("\nper institution:")
for k, v in per_inst.most_common():
    print(f"  {k:<52} {v:>8,}")
