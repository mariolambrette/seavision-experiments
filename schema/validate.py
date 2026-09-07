#!/usr/bin/env python3
"""Validate a SeaVision collation against the schema and check integrity."""

import argparse, collections, json, os, sys
import jsonschema

SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "seavision_schema.json")


def check(path, schema_path=SCHEMA, tol=1e-6):
    errors, warnings = [], []
    data = json.load(open(path, encoding="utf-8"))
    schema = json.load(open(schema_path, encoding="utf-8"))

    # 1. structure and unknown keys
    v = jsonschema.Draft202012Validator(schema)
    for e in sorted(v.iter_errors(data), key=lambda x: list(x.path))[:50]:
        errors.append(f"schema: {'/'.join(str(p) for p in e.path)}: {e.message}")

    imgs = {i["id"]: i for i in data.get("images", [])}
    cats = {c["id"] for c in data.get("categories", [])}
    dsets = {d["id"] for d in data.get("datasets", [])}
    lics = {l["id"] for l in data.get("licenses", [])}

    # 2. unique ids and file names
    for name, ids in [("image", [i["id"] for i in data.get("images", [])]),
                      ("annotation", [a["id"] for a in data.get("annotations", [])])]:
        dup = [k for k, n in collections.Counter(ids).items() if n > 1]
        if dup:
            errors.append(f"duplicate {name} ids: {dup[:10]}")
    fn = [i["file_name"] for i in data.get("images", [])]
    dup = [k for k, n in collections.Counter(fn).items() if n > 1]
    if dup:
        errors.append(f"duplicate file_names: {dup[:10]}")

    # 3. referential integrity
    for i in data.get("images", []):
        if i["dataset_id"] not in dsets:
            errors.append(f"image {i['id']}: dataset_id {i['dataset_id']} not in datasets")
    for d in data.get("datasets", []):
        if d.get("license_id") is not None and d["license_id"] not in lics:
            errors.append(f"dataset {d['id']}: license_id {d['license_id']} not in licenses")
    for a in data.get("annotations", []):
        if a["image_id"] not in imgs:
            errors.append(f"annotation {a['id']}: image_id {a['image_id']} does not exist")
        if a["category_id"] not in cats:
            errors.append(f"annotation {a['id']}: category_id {a['category_id']} does not exist")

    # 4. geometry
    for a in data.get("annotations", []):
        im = imgs.get(a["image_id"])
        if not im:
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            errors.append(f"annotation {a['id']}: degenerate bbox {a['bbox']}")
        if x < 0 or y < 0 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
            warnings.append(f"annotation {a['id']}: bbox outside image bounds")
        if abs(a["area"] - w * h) > tol * max(1.0, w * h):
            warnings.append(f"annotation {a['id']}: area {a['area']} != w*h {w*h}")

    # 5. grouping sanity
    for i in data.get("images", []):
        missing = set(i["groups"]) - set(i["group_sources"])
        if missing:
            errors.append(f"image {i['id']}: groups {sorted(missing)} have no group_sources entry")

    # 6. counts
    counts = {
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "categories": len(data.get("categories", [])),
        "datasets": len(data.get("datasets", [])),
        "empty_images": len(set(imgs) - {a["image_id"] for a in data.get("annotations", [])}),
        "images_with_unlabelled_animal":
            sum(1 for i in data.get("images", []) if i.get("has_unlabelled_animal")),
    }
    return errors, warnings, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--schema", default=SCHEMA)
    ap.add_argument("--expect", help="JSON dict of expected counts, e.g. '{\"images\":2667}'")
    args = ap.parse_args()

    errors, warnings, counts = check(args.json_path, args.schema)

    print("counts:")
    for k, v in counts.items():
        print(f"  {k:<32} {v}")

    if args.expect:
        for k, want in json.loads(args.expect).items():
            got = counts.get(k)
            mark = "OK  " if got == want else "FAIL"
            print(f"  {mark} {k}: got {got}, expected {want}")
            if got != want:
                errors.append(f"count mismatch: {k} got {got}, expected {want}")

    for w in warnings[:30]:
        print("WARN ", w)
    if len(warnings) > 30:
        print(f"WARN  ... and {len(warnings)-30} more")
    for e in errors[:30]:
        print("ERROR", e)
    if len(errors) > 30:
        print(f"ERROR ... and {len(errors)-30} more")

    print("\nVALID" if not errors else f"\nINVALID: {len(errors)} error(s)")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
