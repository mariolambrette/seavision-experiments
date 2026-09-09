import json
from concurrent.futures import ThreadPoolExecutor
from fathomnet.api import boundingboxes, worms

OUT = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_concept_resolve.json"
counts = {c.concept: c.count for c in boundingboxes.count_total_by_concept()}
print(f"{len(counts)} concepts")

def resolve(name):
    try:
        n = worms.get_info(name)
        if n and n.aphiaId:
            return {"concept": name, "boxes": counts[name], "ok": True,
                    "aphia_id": n.aphiaId, "rank": n.rank, "matched": n.name}
    except Exception:
        pass
    return {"concept": name, "boxes": counts[name], "ok": False}

with ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(ex.map(resolve, counts.keys()))

ok = [r for r in rows if r["ok"]]
bad = sorted([r for r in rows if not r["ok"]], key=lambda r: -r["boxes"])
print(f"\nresolved  : {len(ok):>5}   boxes={sum(r['boxes'] for r in ok):,}")
print(f"UNRESOLVED: {len(bad):>5}   boxes={sum(r['boxes'] for r in bad):,}")
print(f"  of those, >=20 boxes: {sum(1 for r in bad if r['boxes'] >= 20)}")

ranks = {}
for r in ok:
    ranks[r["rank"]] = ranks.get(r["rank"], 0) + 1
print("\nranks:", dict(sorted(ranks.items(), key=lambda kv: -kv[1])))

print("\ntop 40 unresolved:")
for r in bad[:40]:
    print(f"  {r['boxes']:>8,}  {r['concept']}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print("\nwritten:", OUT)
