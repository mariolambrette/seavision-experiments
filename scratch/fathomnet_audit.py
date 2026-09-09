import csv, re
from collections import Counter

CSV = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_concept_review.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

GEAR = re.compile(r"equipment|sampler|manipulator|rov|vehicle|cable|rope|marker|"
                  r"lander|trap|net|frame|light|laser|thruster|arm|tool|sled|"
                  r"platform|mooring|sensor|camera|elevator|benchmark|transect",
                  re.I)

cls = [r for r in rows if r["action"] == "class"]
susp = [r for r in cls if r["concept"][:1].islower()
        and r["rank"] in ("Genus", "Species", "Subspecies")]
print(f"auto-classed: {len(cls)}")
print(f"SUSPECT (lowercase concept -> genus/species): {len(susp)}")
for r in sorted(susp, key=lambda r: -int(r["boxes"])):
    print(f"  {int(r['boxes']):>7,}  {r['concept']:<34} -> "
          f"{r['resolved_name']} ({r['rank']}, {r['kingdom']})")

rev = [r for r in rows if r["action"] == "review"]
print(f"\nreview rows: {len(rev)}")
print("  by note:", dict(Counter(r["notes"] for r in rev)))
gear = [r for r in rev if GEAR.search(r["concept"])]
print(f"\nlikely equipment/non-organism in review ({len(gear)}):")
for r in sorted(gear, key=lambda r: -int(r["boxes"]))[:40]:
    print(f"  {int(r['boxes']):>7,}  {r['concept']}")

rest = [r for r in rev if r not in gear and not r["suggestion"]]
print(f"\nno suggestion, not equipment ({len(rest)}) - the real manual work:")
for r in sorted(rest, key=lambda r: -int(r["boxes"]))[:40]:
    print(f"  {int(r['boxes']):>7,}  {r['concept']:<38} {r['notes']}")
