import csv, os, re, json
from collections import Counter

RAW = r"N:\marineai\dataset\raw\ozfish"
CROPS = os.path.join(RAW, "crops")
META = os.path.join(RAW, "crop_metadata.csv")
OUT = r"D:\marineai\classification-experiments\sw\scratch\ozfish_namecheck.json"

SUFFIX = re.compile(r"^(?P<base>.+?\.(?:avi|mp4|mpeg)\..+?\.png)-\d+-\d+\.png$", re.I)

files = os.listdir(CROPS)
base, nomatch = {}, []
for f in files:
    m = SUFFIX.match(f)
    if m:
        base.setdefault(m["base"], []).append(f)
    else:
        nomatch.append(f)

print(f"files: {len(files):,}")
print(f"suffix pattern did NOT match: {len(nomatch):,}")
for f in nomatch[:10]:
    print("   ", f)

dupes = {k: v for k, v in base.items() if len(v) > 1}
print(f"\ndistinct base names: {len(base):,}")
print(f"base names claimed by >1 file: {len(dupes):,}")
for k, v in list(dupes.items())[:10]:
    print(f"   {k}\n      {v}")

with open(META, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
print(f"\ncrop_metadata.csv: {len(rows):,} rows; columns: {list(rows[0].keys())}")
csv_names = [r["file_name"] for r in rows]
cn, bn = set(csv_names), set(base)
print(f"  distinct file_name in csv: {len(cn):,}")
print(f"  csv names with no file on disk : {len(cn - bn):,}")
print(f"  files on disk not in csv       : {len(bn - cn):,}")
for n in list(cn - bn)[:5]:
    print("   missing:", n)
for n in list(bn - cn)[:5]:
    print("   extra  :", n)

# is the counter a global row index?
idx = []
for f in files:
    m = re.search(r"-(\d+)-(\d+)\.png$", f)
    if m:
        idx.append((int(m.group(1)), int(m.group(2))))
first = sorted(i for i, _ in idx)
second = Counter(j for _, j in idx)
print(f"\ncounter 1: min={first[0]} max={first[-1]} distinct={len(set(first)):,}")
print(f"counter 2: {dict(second)}")

json.dump({"n_files": len(files), "n_base": len(base),
           "nomatch": nomatch[:200], "dupes": {k: v for k, v in list(dupes.items())[:200]},
           "csv_missing": list(cn - bn)[:200], "disk_extra": list(bn - cn)[:200]},
          open(OUT, "w", encoding="utf-8"), indent=2)
print("\nwritten:", OUT)
