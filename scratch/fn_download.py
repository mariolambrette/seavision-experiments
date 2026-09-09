import io, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image

MANIFEST = r"D:\marineai\classification-experiments\sw\scratch\fn_manifest"
OUT      = r"N:\marineai\dataset\raw\fathomnet\images"
FAILLOG  = r"D:\marineai\classification-experiments\sw\scratch\fn_download_failures.jsonl"
WORKERS, QUALITY = 12, 95

# JPEG institutions first: cheap, and they carry the reef-fish block
ORDER = ["NOAA NMFS SEFSC", "NOAA Ocean Exploration", "Ocean Networks Canada",
         "CENCOOS", "POSCO", "UniOfPlym", "UW NSF-OOI CSSF", "UW NSF-OOI WHOI",
         "ONCS", "OET", "Universidad de Costa Rica and Schmidt Ocean Institute",
         "Joost Daniels", "MBARI", "Schmidt Ocean Institute"]

_local = threading.local()
def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "SeaVision/1.0 (research; University of Exeter)"
        a = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", a); s.mount("http://", a)
        _local.s = s
    return _local.s

lock = threading.Lock()
stats = {"done": 0, "skip": 0, "fail": 0, "bytes_in": 0, "bytes_out": 0}

def fetch(rec, outdir):
    dest = os.path.join(outdir, rec["uuid"] + ".jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        with lock: stats["skip"] += 1
        return
    url = rec["url"]
    for attempt in range(4):
        try:
            r = session().get(url, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** attempt) + random.random() * 2); continue
            r.raise_for_status()
            raw = r.content
            break
        except Exception as e:
            if attempt == 3:
                with lock:
                    stats["fail"] += 1
                    with open(FAILLOG, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"uuid": rec["uuid"], "url": url,
                                             "error": str(e)[:300]}) + "\n")
                return
            time.sleep((2 ** attempt) + random.random() * 2)
    else:
        with lock:
            stats["fail"] += 1
            with open(FAILLOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"uuid": rec["uuid"], "url": url,
                                     "error": "exhausted retries on HTTP status"}) + "\n")
        return

    ext = os.path.splitext(url.split("?")[0])[1].lower()
    tmp = dest + ".part"
    try:
        if ext in (".jpg", ".jpeg"):
            with open(tmp, "wb") as fh:      # already JPEG: no generation loss
                fh.write(raw)
        else:
            im = Image.open(io.BytesIO(raw))
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGB")
            elif im.mode != "RGB":
                im = im.convert("RGB")
            im.save(tmp, "JPEG", quality=QUALITY, optimize=True)
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        with lock:
            stats["fail"] += 1
            with open(FAILLOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"uuid": rec["uuid"], "url": url,
                                     "error": "encode: " + str(e)[:300]}) + "\n")
        return

    with lock:
        stats["done"] += 1
        stats["bytes_in"] += len(raw)
        stats["bytes_out"] += os.path.getsize(dest)
        n = stats["done"] + stats["skip"]
        if n % 1000 == 0:
            print(f"    {n:,} ({stats['done']:,} new, {stats['skip']:,} skipped, "
                  f"{stats['fail']:,} failed)  "
                  f"{stats['bytes_out']/1e9:.1f} GB written", flush=True)

def safe(code):
    return "".join(c if c.isalnum() else "_" for c in code)

for code in ORDER:
    path = os.path.join(MANIFEST, safe(code) + ".jsonl")
    if not os.path.exists(path):
        print(f"skip {code}: no manifest"); continue
    outdir = os.path.join(OUT, safe(code))
    os.makedirs(outdir, exist_ok=True)
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"\n=== {code}: {len(recs):,} images -> {outdir}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda r: fetch(r, outdir), recs))
    print(f"  {code} finished in {(time.time()-t0)/60:.1f} min", flush=True)

print(f"\nDONE  new {stats['done']:,}  skipped {stats['skip']:,}  "
      f"failed {stats['fail']:,}")
print(f"  fetched {stats['bytes_in']/1e9:.1f} GB, wrote {stats['bytes_out']/1e9:.1f} GB")
print(f"  failures logged to {FAILLOG}")
