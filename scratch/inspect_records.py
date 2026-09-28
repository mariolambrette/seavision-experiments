#!/usr/bin/env python3
"""Pull named records out of one or more shard sets and look at them.

Written for two records that failed level 3 of the square verification, but
kept general because "a verifier flagged something and I need to see it" keeps
happening and guessing from the metadata has a poor record in this project.

It does two things. It prints every stored field for each key in each set,
which often ends the question on its own -- if both flagged records turn out
to be `margin_reduced`, or both `padded`, that is the answer and no image is
needed. And it writes a side-by-side PNG so the remaining cases are settled by
looking rather than by argument.

    python scratch/inspect_records.py \\
        --sets "D:/marineai/classification-experiments/shards/crops" \\
               "D:/marineai/classification-experiments/shards/square-m00" \\
               "D:/marineai/classification-experiments/shards/square-m10" \\
        --keys fn-1c3356cb22 fn-6648fad6ff \\
        --out "D:/marineai/_audit/inspect.png"

The box recorded in `box_in_square` is drawn on each square, so a square whose
animal is in the wrong place is obvious: the outline will sit on empty seabed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile

from PIL import Image, ImageDraw

INTERESTING = ["source", "geometry", "margin_key", "tier", "pad_fraction",
               "shift", "square_side_native", "square_origin",
               "box_in_square", "frame_size", "width", "height",
               "native_width", "native_height", "category_name",
               "byte_copied", "crop_provenance"]


def pull(set_dir, keys):
    """-> {key: (image_bytes, meta)} -- one pass, stops once all are found."""
    found = {}
    tars = sorted(f for f in os.listdir(set_dir) if f.endswith(".tar"))
    for t in tars:
        with tarfile.open(os.path.join(set_dir, t)) as tf:
            for info in tf:
                key, _, ext = info.name.partition(".")
                if key not in keys:
                    continue
                payload = tf.extractfile(info).read()
                slot = found.setdefault(key, [None, None])
                if ext == "json":
                    slot[1] = json.loads(payload)
                else:
                    slot[0] = payload
        if all(v[0] and v[1] for v in found.values()) and \
                len(found) == len(keys):
            break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", default="inspect.png")
    ap.add_argument("--tile", type=int, default=340)
    args = ap.parse_args()

    keys = set(args.keys)
    grids = {}
    for d in args.sets:
        if not os.path.isdir(d):
            sys.exit(f"{d} is not a directory")
        label = os.path.basename(os.path.normpath(d))
        print(f"\n=== {label} ===")
        got = pull(d, keys)
        grids[label] = got
        for k in args.keys:
            if k not in got:
                print(f"  {k}: NOT IN THIS SET")
                continue
            m = got[k][1] or {}
            print(f"  {k}")
            for f in INTERESTING:
                if f in m and m[f] not in (None, {}, []):
                    print(f"      {f:<20} {m[f]}")

    cell, pad = args.tile, 8
    label_h = 22
    cols, rows = len(args.keys), len(grids)
    W = pad + cols * (cell + pad)
    H = pad + rows * (cell + label_h + pad)
    sheet = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)

    for r, (label, got) in enumerate(grids.items()):
        for c, k in enumerate(args.keys):
            x = pad + c * (cell + pad)
            y = pad + r * (cell + label_h + pad)
            dr.text((x + 3, y + 4), f"{label}  {k}", fill=(0, 0, 0))
            box = Image.new("RGB", (cell, cell), (25, 25, 25))
            if k in got and got[k][0]:
                with Image.open(io.BytesIO(got[k][0])) as im:
                    im = im.convert("RGB")
                    m = got[k][1] or {}
                    bis = m.get("box_in_square")
                    side = m.get("square_side_native")
                    if bis and side:
                        # draw the recorded box, in the stored image's own
                        # pixels -- if the animal is not inside it, the
                        # metadata and the pixels disagree and that is the
                        # whole question
                        s = im.width / float(side)
                        d2 = ImageDraw.Draw(im)
                        d2.rectangle([bis[0] * s, bis[1] * s,
                                      (bis[0] + bis[2]) * s,
                                      (bis[1] + bis[3]) * s],
                                     outline=(255, 60, 60), width=3)
                    im.thumbnail((cell, cell), Image.LANCZOS)
                    box.paste(im, ((cell - im.width) // 2,
                                   (cell - im.height) // 2))
            sheet.paste(box, (x, y + label_h))
            dr.rectangle([x, y + label_h, x + cell, y + label_h + cell],
                         outline=(120, 120, 120), width=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    sheet.save(args.out, "PNG")
    print(f"\nwrote {args.out}")
    print("The red outline is the box the metadata claims. If the animal sits")
    print("inside it in every row, the geometry is right and the level 3")
    print("difference is a measurement artefact. If it sits outside in the")
    print("square rows only, the square is cut from the wrong place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
