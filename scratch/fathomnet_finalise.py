import csv, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests

CSV = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_concept_review.csv"
OUT = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_concept_final.csv"
UA = {"User-Agent": "SeaVision/1.0"}
MIN_BOXES = 20

FIX = {
    "sea fan": "Alcyonacea", "feather star": "Crinoidea",
    "black coral": "Antipatharia", "squat lobster": "Galatheoidea",
    "hydroid": "Hydrozoa", "salp": "Salpida", "sea snail": "Gastropoda",
    "octopus": "Octopoda", "sea slug": "Nudibranchia", "barnacle": "Cirripedia",
    "ctenophore": "Ctenophora", "hermit crab": "Paguroidea",
    "lobster": "Nephropidae", "chimaera": "Chimaeriformes",
    "chiton": "Polyplacophora", "cuttlefish": "Sepiida",
    "Anemone": "Actiniaria",
}

def worms(name):
    for a in range(3):
        try:
            r = requests.get(
                "https://www.marinespecies.org/rest/AphiaRecordsByName/"
                f"{requests.utils.quote(name)}?like=false&marine_only=false",
                headers=UA, timeout=30)
            if r.status_code != 200: return None
            recs = r.json()
            acc = [x for x in recs if x.get("status") == "accepted"]
            return (acc or recs)[0] if recs else None
        except Exception:
            time.sleep(2 ** a)
    return None

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
with ThreadPoolExecutor(max_workers=4) as ex:
    fixed = dict(zip(FIX, ex.map(worms, FIX.values())))

print("corrections:")
for c, rec in fixed.items():
    if rec is None:
        print(f"  !! {c}: {FIX[c]} did not resolve"); continue
    print(f"  {c:<16} -> {rec['valid_name']} ({rec['valid_AphiaID']}, {rec['rank']})")
    for r in rows:
        if r["concept"] == c:
            r["resolved_name"] = rec["valid_name"]
            r["aphia_id"] = str(rec["valid_AphiaID"])
            r["rank"] = rec["rank"]
            r["notes"] = "vernacular corrected"

# post-merge totals per AphiaID, over everything not dropped for policy reasons
POLICY = {"diatom", "not an organism"}
merged = defaultdict(int)
for r in rows:
    if r["aphia_id"] and r["notes"] not in POLICY and r["action"] != "unlabelled":
        merged[r["aphia_id"]] += int(r["boxes"])

revived, still_thin, kept, dropped = [], 0, 0, 0
for r in rows:
    aid = r["aphia_id"]
    if r["action"] == "drop" and aid and r["notes"] not in POLICY:
        if merged[aid] >= MIN_BOXES:
            revived.append((r["concept"], int(r["boxes"]), r["resolved_name"],
                            merged[aid]))
            r["action"] = "class"
        else:
            still_thin += 1
    if r["action"] == "class": kept += int(r["boxes"])
    elif r["action"] == "drop": dropped += int(r["boxes"])

revived.sort(key=lambda x: -x[1])
print(f"\nrows revived by post-merge threshold: {len(revived)}"
      f"  boxes recovered: {sum(x[1] for x in revived):,}")
for c, n, name, tot in revived[:30]:
    print(f"  {n:>6,}  {c:<34} -> {name} (class total {tot:,})")
print(f"\nstill below {MIN_BOXES} after merge: {still_thin} rows")
print(f"\nboxes kept as classes: {kept:,}")
print(f"boxes dropped        : {dropped:,}")
print(f"distinct categories  : "
      f"{len({r['aphia_id'] for r in rows if r['action']=='class' and r['aphia_id']})}")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("\nwritten:", OUT)
