import csv, requests
UA = {"User-Agent": "SeaVision/1.0"}
CSV = r"D:\marineai\classification-experiments\sw\taxon_maps\fathomnet_concepts.csv"

for aid, what in [(2, "Animalia (kingdom)"), (1821, "Chordata (phylum)"),
                  (159795, "Lutjanus campechanus"), (206059, "Lethrinus (genus)")]:
    for attempt in range(2):
        try:
            r = requests.get("https://www.marinespecies.org/rest/"
                             f"AphiaClassificationByAphiaID/{aid}",
                             headers=UA, timeout=20)
            print(f"  {aid:>8}  {what:<26} HTTP {r.status_code}"
                  f"  {len(r.content)} bytes")
            break
        except Exception as e:
            if attempt:
                print(f"  {aid:>8}  {what:<26} {type(e).__name__}")

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r["action"] == "class" and r["rank"] in ("Kingdom", "")]
print(f"\nclass rows at Kingdom rank or with no rank: {len(rows)}")
for r in rows:
    print(f"  {int(r['boxes']):>7,}  {r['concept']:<30} "
          f"{r['rank'] or '(blank)'}  {r['aphia_id']}")
