#!/usr/bin/env python3
"""
Download the OzFish frames from Pawsey.

    https://storage.pawsey.org.au/public/m/FDFML/frames      ~45k frames
    https://storage.pawsey.org.au/public/m/FDFML/metadata    frame_metadata.csv

Why all of them rather than a designed subset: square expansion needs the
parent frame, and the first experiment's geometry result is backbone-dependent
in a way that matters. Square is DINOv2's WORST geometry (0.72 vs 0.79/0.80 at
k=20) but the CLIP family's BEST -- BioCLIP 2 scores 0.79 with square against
0.69 distort and 0.62 letterbox. Four of the six backbones in plan 5.3 are
CLIP-family, and square is what brings BioCLIP 2 level with DINOv2 at all. So
denying square expansion to the strongest source in the collation would
handicap most of the backbone list.

WHAT THIS CHECKS, AND WHY
The SEAMAPD21 attempt (WP4) accumulated 65 "downloaded" part files before
anyone noticed every one was a 280-byte HTML error page. A downloader that
only checks whether a file arrived will do that again. So every file is
verified by MAGIC BYTES and a minimum size before it counts as downloaded,
and anything that fails is reported by category rather than silently retried
into a pile of junk.

    python fetch_ozfish_frames.py --raw-root "D:/marineai/dataset/raw/ozfish"
    python fetch_ozfish_frames.py --raw-root "..." --verify-only
    python fetch_ozfish_frames.py --raw-root "..." --limit 20      # smoke test
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import sys
import threading
import time
from collections import Counter

FRAMES_URL = "https://storage.pawsey.org.au/public/m/FDFML/frames"
METADATA_URL = "https://storage.pawsey.org.au/public/m/FDFML/metadata"
FRAME_META = "frame_metadata.csv"

IMAGE_EXT = (".png", ".jpg", ".jpeg")
MAGIC = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpeg"}
MIN_BYTES = 1024          # SEAMAPD21's error pages were 280 bytes

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def check_file(path):
    """-> (ok, reason). A file is good only if it is big enough AND starts
    with an image magic number. 'It exists' is not a check."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, f"unstatable: {exc}"
    if size < MIN_BYTES:
        return False, f"too small ({size} bytes)"
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError as exc:
        return False, f"unreadable: {exc}"
    for magic in MAGIC:
        if head.startswith(magic):
            return True, ""
    if head[:1] in (b"<", b"{"):
        return False, "looks like HTML or JSON, not an image"
    return False, f"unrecognised magic {head[:4]!r}"


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

def load_frame_names(raw_root, metadata_url, name_column, session):
    """Frame file names, from the local metadata CSV if present else Pawsey."""
    local = os.path.join(raw_root, FRAME_META)
    if not os.path.exists(local):
        url = f"{metadata_url.rstrip('/')}/{FRAME_META}"
        log(f"  {FRAME_META} not found locally; fetching {url}")
        os.makedirs(raw_root, exist_ok=True)
        r = session.get(url, timeout=120)
        r.raise_for_status()
        if r.content[:1] in (b"<",):
            sys.exit(f"{url} returned HTML, not a CSV. Check the URL before "
                     f"starting a 45,000-file download against it.")
        with open(local, "wb") as fh:
            fh.write(r.content)
        log(f"  saved {local}")

    with open(local, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{local} is empty")

    cols = list(rows[0].keys())
    if name_column:
        if name_column not in cols:
            sys.exit(f"--name-column {name_column!r} not in {cols}")
        col = name_column
    else:
        # Pick the column whose values look like image file names. Guessing a
        # column name would be the same mistake as guessing a URL.
        scored = [(sum(1 for r in rows[:500]
                       if str(r.get(c, "")).lower().endswith(IMAGE_EXT)), c)
                  for c in cols]
        scored.sort(reverse=True)
        if scored[0][0] == 0:
            sys.exit(f"No column in {local} looks like image filenames. "
                     f"Columns: {cols}. Pass --name-column explicitly.")
        col = scored[0][1]
        log(f"  filename column detected: {col!r} "
            f"({scored[0][0]}/{min(500, len(rows))} sampled rows match)")

    names, dupes = [], 0
    seen = set()
    for r in rows:
        n = str(r.get(col, "")).strip()
        if not n:
            continue
        if n in seen:
            dupes += 1
            continue
        seen.add(n)
        names.append(n)
    if dupes:
        log(f"  {dupes:,} duplicate filenames in the metadata, ignored")
    return names


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def fetch_one(session, url, dest, tries=4, timeout=120):
    """-> (status, detail). Downloads to .part then renames, so an interrupted
    run never leaves a truncated file that a later run mistakes for done."""
    part = dest + ".part"
    delay = 2
    for attempt in range(1, tries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return "missing", "HTTP 404"
                r.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        if chunk:
                            fh.write(chunk)
            ok, why = check_file(part)
            if not ok:
                os.remove(part)
                if attempt == tries:
                    return "bad", why
            else:
                os.replace(part, dest)
                return "ok", ""
        except Exception as exc:                          # noqa: BLE001
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass
            if attempt == tries:
                return "failed", f"{type(exc).__name__}: {exc}"
        time.sleep(delay)
        delay *= 2
    return "failed", "retries exhausted"


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

FRAME_PAT = __import__("re").compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\.(?P<frame>\d+)"
    r"\.(?P<imgext>png|jpg|jpeg)$", __import__("re").I)


def candidates(base, name):
    """Plausible URLs for one frame, most likely first.

    The first round of probing tested seven SUBDIRECTORY layouts under
    storage.pawsey.org.au and all seven returned 404. That was the wrong
    hypothesis space: open-AIMS/ozfish issue #1 shows the real portal is
    data.pawsey.org.au/public/?path=/FDFML/frames, which is a MEDIAFLUX
    instance, not a static file server. So the host was wrong, not the path.

    Mediaflux serves public objects from a download endpoint rather than from
    the browse path, and the exact form varies by deployment -- hence testing
    several rather than asserting one.
    """
    m = FRAME_PAT.match(name)
    b = base.rstrip("/")
    out = [("storage: flat", f"{b}/{name}")]
    if m:
        vid = m["vid"] + "_" + m["cam"].upper()
        video = f"{vid}.{m['ext'].lower()}"
        survey = __import__("re").match(r"^([A-Za-z]+)", m["vid"]).group(1)
        out += [
            ("storage: by video",    f"{b}/{video}/{name}"),
            ("storage: by survey",   f"{b}/{survey}/{name}"),
        ]
    D = "https://data.pawsey.org.au"
    rel = f"FDFML/frames/{name}"
    out += [
        ("mediaflux: /download/public",  f"{D}/download/public/{rel}"),
        ("mediaflux: /public/download",  f"{D}/public/download/{rel}"),
        ("mediaflux: /download?path",    f"{D}/download?path=/{rel}"),
        ("mediaflux: /public?path",      f"{D}/public/?path=/{rel}"),
        ("mediaflux: /data/public",      f"{D}/data/public/{rel}"),
    ]
    return out


def cmd_list(args, session):
    """Ask the object store what its keys ACTUALLY are.

    Seven path layouts and three hosts have now failed, which means the
    hypothesis being tested is wrong rather than incompletely enumerated. The
    crops already on disk are named X.png-1-1.png rather than X.png, because
    the portal renamed them on the way out -- so the store's keys may simply
    not be frame_metadata.csv's file_name, and no URL built from that column
    will ever resolve.

    This asks for a listing instead of constructing a path, in the three
    dialects such a store might speak.
    """
    base = args.frames_url.rstrip("/")
    root, prefix = base.rsplit("/public/", 1) if "/public/" in base else (base, "")
    root = root + "/public" if "/public/" in base else base

    attempts = [
        ("S3 v2",  root, {"list-type": "2", "prefix": prefix, "max-keys": "20"}),
        ("S3 v1",  root, {"prefix": prefix, "max-keys": "20"}),
        ("Swift",  root, {"prefix": prefix, "format": "json", "limit": "20"}),
        ("bare",   base, {}),
    ]
    for label, url, params in attempts:
        print(f"\n--- {label}: {url}  {params}")
        try:
            r = session.get(url, params=params, timeout=90)
            ctype = r.headers.get("content-type", "?")
            print(f"    HTTP {r.status_code}   {ctype}   {len(r.content):,} bytes")
            body = r.text[:2500]
            if r.status_code == 200 and body.strip():
                print("    ---- first 2,500 characters ----")
                for line in body.splitlines()[:40]:
                    print("    " + line[:160])
        except Exception as exc:                          # noqa: BLE001
            print(f"    ERR {type(exc).__name__}: {exc}")

    print("\nWhat to look for: <Key>, <Name>, or a JSON 'name' field. Those are")
    print("the real object paths. Send me a handful and I will build the URL")
    print("from what the store says rather than from what the metadata implies.")
    return 0


def cmd_probe(args, session):
    """Find the real URL layout by testing one file, and prove the method
    works by testing a CROP whose layout we already know succeeds."""
    import requests                                       # noqa: F401

    frames = load_frame_names(args.raw_root, args.metadata_url,
                              args.name_column, session)
    name = frames[0]
    print(f"\nprobing with frame: {name}")
    print("-" * 74)

    found = []
    for label, url in candidates(args.frames_url, name):
        try:
            r = session.get(url, timeout=60, stream=True)
            head = r.raw.read(8, decode_content=True) if r.status_code == 200 else b""
            r.close()
            is_img = any(head.startswith(m) for m in MAGIC)
            print(f"  {r.status_code:<5} {'IMAGE' if is_img else '     '}  "
                  f"{label:<28} {url}")
            if r.status_code == 200 and is_img:
                found.append((label, url))
        except Exception as exc:                          # noqa: BLE001
            print(f"  ERR   {'':5}  {label:<28} {type(exc).__name__}: {exc}")

    # Control: does the CROPS product answer at its documented flat path? If
    # it does not either, the base URLs are not direct file paths at all and
    # no amount of subdirectory guessing will help.
    crop_meta = os.path.join(args.raw_root, "crop_metadata.csv")
    if os.path.exists(crop_meta):
        with open(crop_meta, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        col = next((c for c in rows[0]
                    if str(rows[0].get(c, "")).lower().endswith(IMAGE_EXT)), None)
        if col:
            cname = str(rows[0][col]).strip()
            base = args.frames_url.rstrip("/").rsplit("/", 1)[0]
            D = "https://data.pawsey.org.au"
            controls = [
                ("storage: flat",               f"{base}/crops/{cname}"),
                ("mediaflux: /download/public", f"{D}/download/public/FDFML/crops/{cname}"),
                ("mediaflux: /public/download", f"{D}/public/download/FDFML/crops/{cname}"),
            ]
            print("\ncontrol -- the CROPS you already hold, which definitely exist.")
            print("Whichever form serves a crop is the form that will serve a frame.")
            hit = False
            for label, curl in controls:
                try:
                    r = session.get(curl, timeout=60, stream=True)
                    head = r.raw.read(8, decode_content=True) if r.status_code == 200 else b""
                    r.close()
                    ok = any(head.startswith(m) for m in MAGIC)
                    hit = hit or ok
                    print(f"  {r.status_code:<5} {'IMAGE' if ok else '     '}  "
                          f"{label:<28} {curl}")
                except Exception as exc:                  # noqa: BLE001
                    print(f"  ERR   {'':5}  {label:<28} {type(exc).__name__}: {exc}")
            if not hit:
                print("\n  No form serves a crop either. These are not direct file")
                print("  URLs, and no path guessing will help -- the archive came")
                print("  through the portal as a zip and the frames must too.")
                print("  Note issue #1: that zip 'dies part way through' for other")
                print("  people, with no maintainer answer and no mirror.")

    print()
    if found:
        print("USE THIS LAYOUT:")
        for label, url in found:
            print(f"  {label}: {url}")
        print("\nTell me which one and I will wire it into the downloader.")
    else:
        print("No layout returned an image. Do not start a bulk download.")
    return 0 if found else 1


def main():
    ap = argparse.ArgumentParser(description="Download OzFish frames")
    ap.add_argument("--raw-root", required=True,
                    help="e.g. D:/marineai/dataset/raw/ozfish")
    ap.add_argument("--frames-url", default=FRAMES_URL)
    ap.add_argument("--metadata-url", default=METADATA_URL)
    ap.add_argument("--name-column", default=None,
                    help="column in frame_metadata.csv holding the file name; "
                         "auto-detected if omitted")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW downloads -- use 20 first")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-check what is already on disk and exit")
    ap.add_argument("--probe", action="store_true",
                    help="find the real URL layout by testing one file")
    ap.add_argument("--list", dest="do_list", action="store_true",
                    help="ask the object store for its actual keys")
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("requests is required: pip install requests")

    out_dir = os.path.join(args.raw_root, "frames")
    os.makedirs(out_dir, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "SeaVision-collation (University of Exeter)"

    if args.do_list:
        sys.exit(cmd_list(args, session))

    if args.probe:
        sys.exit(cmd_probe(args, session))

    log("reading metadata")
    names = load_frame_names(args.raw_root, args.metadata_url,
                             args.name_column, session)
    log(f"  {len(names):,} frames listed")

    # ---- what is already good on disk ---------------------------------
    log("checking what is already on disk")
    have, broken = set(), []
    for n in names:
        p = os.path.join(out_dir, n)
        if os.path.exists(p):
            ok, why = check_file(p)
            if ok:
                have.add(n)
            else:
                broken.append((n, why))
    log(f"  {len(have):,} already present and valid")
    if broken:
        log(f"  {len(broken):,} present but INVALID -- these will be re-fetched")
        for n, why in broken[:10]:
            log(f"      {n}: {why}")

    if args.verify_only:
        log("")
        log(f"verify-only: {len(have):,} valid, {len(broken):,} invalid, "
            f"{len(names) - len(have) - len(broken):,} absent")
        sys.exit(1 if broken else 0)

    todo = [n for n in names if n not in have]
    if args.limit:
        todo = todo[:args.limit]
    log(f"  {len(todo):,} to download")
    if not todo:
        log("nothing to do")
        return

    stats = Counter()
    problems = []
    t0 = time.time()
    done = 0

    def work(n):
        url = f"{args.frames_url.rstrip('/')}/{n}"
        return n, fetch_one(session, url, os.path.join(out_dir, n))

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (status, detail) in ex.map(work, todo):
            stats[status] += 1
            done += 1
            if status != "ok":
                problems.append((n, status, detail))
            if done % 250 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el if el else 0
                left = (len(todo) - done) / rate if rate else 0
                log(f"  {done:,}/{len(todo):,}  {rate:.1f}/s  "
                    f"~{left/60:.0f} min left  {dict(stats)}")

    log("")
    log(f"downloaded {stats['ok']:,}   missing {stats['missing']:,}   "
        f"bad content {stats['bad']:,}   failed {stats['failed']:,}")

    if problems:
        by_kind = Counter(s for _, s, _ in problems)
        log(f"\n{len(problems):,} problem(s): {dict(by_kind)}")
        for n, s, d in problems[:20]:
            log(f"    {s:<8} {n}  {d}")
        log("\nRe-run to retry; valid files are skipped. If 'bad content' is")
        log("large, STOP -- that is the SEAMAPD21 failure and the URL is wrong.")

    log("")
    log("=" * 70)
    log("IMPORTANT -- seed the NAS before the next sync runs.")
    log("raw/ syncs NAS -> D: with /MIR, so a pull would DELETE these frames,")
    log("which exist only on D:. Copy them up first, with /E and never /MIR:")
    log("")
    log(f'  robocopy "{out_dir}" '
        f'"N:\\marineai\\dataset\\raw\\ozfish\\frames" /E /MT:32 /R:2 /W:5')
    log("=" * 70)


if __name__ == "__main__":
    main()