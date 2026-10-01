#!/usr/bin/env python3
"""Fetch per-image position, depth and time from the FathomNet API. READ-ONLY
with respect to the collation; writes one resumable JSONL file.

Why: the URL groups need a check in the direction that matters -- whether
one deployment has been SPLIT across two groups. Positions give that
(two groups at the same place on the same day are a candidate split), and
the API holds positions the collation does not (SEFSC lat/lon are null in
our records but present in the API). The tags carry no dive id and the
upload/DarwinCore records are bulk batches ("GFISHER 01 of 13"), so
positions are the best independent evidence available.

Batched through images.find_by_uuid_in_list, so ~430k frames is a few
thousand calls, not 430k. Resumable: every batch is appended and flushed,
and a re-run skips uuids already written, including ones the API did not
return (recorded as missing, so they are not requested forever).

    python scratch/fathomnet_fetch_positions.py ^
        --groups D:\\marineai\\scratch\\fathomnet_groups\\fathomnet_groups.csv.gz ^
        --out    D:\\marineai\\scratch\\fathomnet_positions.jsonl

Start with --limit 2000 to check the batch size works and see the rate,
then run without it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time

DEFAULT_INST = ["NOAA NMFS SEFSC", "MBARI", "NOAA Ocean Exploration"]
KEEP = ["uuid", "url", "latitude", "longitude", "depthMeters", "timestamp",
        "imagingType", "valid", "width", "height"]


def as_dict(obj):
    for m in ("to_dict", "dict", "model_dump"):
        f = getattr(obj, m, None)
        if callable(f):
            try:
                return f()
            except Exception:                            # noqa: BLE001
                pass
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def fetch(images, batch, tries=5):
    delay = 2
    for attempt in range(1, tries + 1):
        try:
            return images.find_by_uuid_in_list(batch)
        except Exception as exc:                         # noqa: BLE001
            print(f"    ! batch failed ({attempt}/{tries}): {exc!r}")
            if attempt == tries:
                raise
            time.sleep(delay)
            delay *= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--institutions", nargs="*", default=DEFAULT_INST)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many uuids (trial run)")
    args = ap.parse_args()

    from fathomnet.api import images

    want, inst_of = [], {}
    with gzip.open(args.groups, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uu = row["fathomnet_image_uuid"]
            if row["owner_institution"] in args.institutions and \
                    uu and uu not in inst_of:
                inst_of[uu] = row["owner_institution"]
                want.append(uu)
    want.sort()                                   # deterministic order

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["uuid"])
                except (ValueError, KeyError):
                    pass                          # a torn last line
    todo = [u for u in want if u not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(want):,} frames wanted, {len(done):,} already fetched, "
          f"{len(todo):,} to do, batch {args.batch}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0, got, missing, with_pos = time.time(), 0, 0, 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            res = fetch(images, batch) or []
            seen = set()
            for dto in res:
                d = as_dict(dto)
                uu = d.get("uuid")
                if uu not in inst_of:
                    continue
                seen.add(uu)
                rec = {k: d.get(k) for k in KEEP}
                rec["institution"] = inst_of[uu]
                out.write(json.dumps(rec, default=str) + "\n")
                got += 1
                with_pos += rec["latitude"] is not None
            for uu in batch:
                if uu not in seen:
                    out.write(json.dumps({"uuid": uu, "missing": True,
                                          "institution": inst_of[uu]}) + "\n")
                    missing += 1
            out.flush()
            n = i + len(batch)
            if (i // args.batch) % 50 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n:,}/{len(todo):,}  {n / max(el, 1e-9):.0f}/s  "
                      f"returned {got:,}  with position {with_pos:,}  "
                      f"missing {missing:,}", flush=True)
            time.sleep(args.sleep)

    print(f"\ndone: returned {got:,}, with position {with_pos:,}, "
          f"missing {missing:,}\nwrote {args.out}")


if __name__ == "__main__":
    main()
