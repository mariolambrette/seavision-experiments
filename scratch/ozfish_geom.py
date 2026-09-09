import os, sys, random, statistics as st
from collections import Counter
from PIL import Image

SW = r"D:\marineai\classification-experiments\sw"
sys.path.insert(0, os.path.join(SW, "converters"))
import common as C
import ozfish_data as O

cfg = C.load_config(os.path.join(SW, "configs", "collate_ozfish.yaml"),
                    O.EXTRA_REQUIRED)
root = os.path.join(cfg["source_root"], cfg["image_root_rel"])
random.seed(0)

rows = [(r, p) for r, p in O.load_rows(cfg) if p]
samp = random.sample(rows, 400)

dw, dh, rw, rh, exact, bysurvey = [], [], [], [], 0, {}
edge = 0
for r, p in samp:
    try:
        with Image.open(O.src_for(root, r.file_name)) as im:
            aw, ah = im.size
    except Exception as e:
        print("unreadable:", r.file_name, e); continue
    pw, ph = p["w"], p["h"]
    if p["x0"] < 0 or p["y0"] < 0:
        edge += 1
    dw.append(aw - pw); dh.append(ah - ph)
    if pw: rw.append(aw / pw)
    if ph: rh.append(ah / ph)
    if (aw, ah) == (pw, ph): exact += 1
    bysurvey.setdefault(p["survey"], []).append((aw / pw if pw else 0,
                                                 ah / ph if ph else 0))

print(f"sampled {len(dw)}   exact matches: {exact}   negative-origin boxes: {edge}")
print(f"\nwidth  delta (actual - filename): {dict(Counter(dw).most_common(8))}")
print(f"height delta (actual - filename): {dict(Counter(dh).most_common(8))}")
print(f"\nwidth  ratio: median {st.median(rw):.4f}  min {min(rw):.4f}  max {max(rw):.4f}")
print(f"height ratio: median {st.median(rh):.4f}  min {min(rh):.4f}  max {max(rh):.4f}")

print("\nratio by survey:")
for sv in sorted(bysurvey):
    v = bysurvey[sv]
    print(f"  {sv}: n={len(v):>4}  w {st.median(x for x, _ in v):.4f}"
          f"   h {st.median(y for _, y in v):.4f}")

print("\n10 examples (filename box -> actual):")
for r, p in samp[:10]:
    with Image.open(O.src_for(root, r.file_name)) as im:
        aw, ah = im.size
    print(f"  {p['survey']} x0={p['x0']:>5} y0={p['y0']:>5}  "
          f"box {p['w']:>4}x{p['h']:<4} -> actual {aw:>4}x{ah:<4}  "
          f"d=({aw-p['w']:>+4},{ah-p['h']:>+4})")
