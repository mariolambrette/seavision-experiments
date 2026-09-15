import json, os, shutil
IMG = r"N:\marineai\dataset\collated\images"
QUAR = r"N:\marineai\dataset\collated\images_orphaned"
FILES = [r"N:\marineai\dataset\collated\seavision.json",
         r"N:\marineai\dataset\collated\seavision_fathomnet.json"]

keep = set()
for p in FILES:
    print("loading", p, flush=True)
    d = json.load(open(p, encoding="utf-8"))
    keep.update(im["file_name"] for im in d["images"])
    del d
print(f"referenced: {len(keep):,}")

on_disk = os.listdir(IMG)
orphans = [f for f in on_disk if f not in keep]
print(f"on disk: {len(on_disk):,}   orphans: {len(orphans):,}")
if orphans:
    os.makedirs(QUAR, exist_ok=True)
    for i, f in enumerate(orphans):
        shutil.move(os.path.join(IMG, f), os.path.join(QUAR, f))
        if i % 10000 == 0:
            print(f"  moved {i:,}")
    print("moved to", QUAR)
