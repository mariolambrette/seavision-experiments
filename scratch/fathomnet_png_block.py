import json, time
from concurrent.futures import ThreadPoolExecutor
from fathomnet.api import boundingboxes
from fathomnet.dto import BoundingBoxConstraintsDTO

OUT = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_png_block.json"
PNG = ["MBARI", "Schmidt Ocean Institute", "Joost Daniels",
       "Universidad de Costa Rica and Schmidt Ocean Institute", "OET"]

print("global per-concept counts...")
glob = {c.concept: c.count for c in boundingboxes.count_total_by_concept()}
print(f"  {len(glob)} concepts")

def png_count(concept):
    for attempt in range(3):
        try:
            return concept, boundingboxes.count(BoundingBoxConstraintsDTO(
                concept=concept, ownerInstitutionCodes=PNG)).count
        except Exception:
            time.sleep(2 ** attempt)
    return concept, None

print(f"counting {len(glob)} concepts within the PNG block (a few minutes)...")
with ThreadPoolExecutor(max_workers=8) as ex:
    png = dict(ex.map(png_count, glob.keys()))

errs = [c for c, v in png.items() if v is None]
rows = [{"concept": c, "png": png[c], "total": glob[c],
         "unique": png[c] == glob[c] and png[c] > 0}
        for c in glob if png[c] is not None]

present = [r for r in rows if r["png"] > 0]
uniq = [r for r in present if r["unique"]]
shared = [r for r in present if not r["unique"]]

print(f"\nerrors: {len(errs)}")
print(f"concepts present in PNG block : {len(present)}")
print(f"  unique to it (lost if dropped): {len(uniq)}"
      f"  boxes={sum(r['png'] for r in uniq):,}")
print(f"  also elsewhere               : {len(shared)}"
      f"  boxes={sum(r['png'] for r in shared):,}")

print("\ntop 30 concepts UNIQUE to the PNG block:")
for r in sorted(uniq, key=lambda r: -r["png"])[:30]:
    print(f"  {r['png']:>8,}  {r['concept']}")

print("\nsingletons/small classes unique to it (<20 boxes):",
      sum(1 for r in uniq if r["png"] < 20))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "errors": errs}, f, indent=2)
print("\nwritten:", OUT)
