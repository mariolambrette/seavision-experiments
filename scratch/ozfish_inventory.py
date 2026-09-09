import csv, os, re, json
from collections import Counter

RAW = r"N:\marineai\dataset\raw\ozfish"
CROPS = os.path.join(RAW, "crops")
OUT = r"D:\marineai\classification-experiments\sw\scratch\ozfish_inventory.json"

PAT = re.compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\.(?P<frame>\d+)"
    r"\.(?P<x0>-?\d+)\.(?P<y0>-?\d+)\.(?P<x1>-?\d+)\.(?P<y1>-?\d+)\.png$",
    re.IGNORECASE)

print("csv/xlsx files found under", RAW)
for dp, _, fns in os.walk(RAW):
    for fn in fns:
        if fn.lower().endswith((".csv", ".xlsx", ".txt")):
            p = os.path.join(dp, fn)
            print(f"  {os.path.getsize(p)/1e6:>9.2f} MB  {p}")

print("\nscanning crops directory...")
files = os.listdir(CROPS)
exts = Counter(os.path.splitext(f)[1].lower() for f in files)
print(f"  {len(files):,} entries; extensions: {dict(exts)}")

parsed, unparsed = {}, []
for f in files:
    m = PAT.match(f)
    if m:
        parsed[f] = m.groupdict()
    elif os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"):
        unparsed.append(f)
print(f"  parsed: {len(parsed):,}   UNPARSED images: {len(unparsed):,}")
for f in unparsed[:15]:
    print("   ", f)

surveys = Counter(re.match(r"^([A-Za-z]+)", v["vid"]).group(1)
                  for v in parsed.values())
deps = {v["vid"] for v in parsed.values()}
cams = Counter(v["cam"] for v in parsed.values())
conts = Counter(v["ext"].lower() for v in parsed.values())
print(f"\n  surveys: {dict(surveys)}")
print(f"  deployments: {len(deps):,}   cameras: {dict(cams)}"
      f"   containers: {dict(conts)}")

zero = [f for f in parsed if os.path.getsize(os.path.join(CROPS, f)) == 0]
print(f"  zero-byte files: {len(zero)}")

res = {"n_files": len(files), "n_parsed": len(parsed),
       "unparsed": unparsed, "surveys": dict(surveys),
       "n_deployments": len(deps), "zero_byte": zero}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("\nwritten:", OUT)
