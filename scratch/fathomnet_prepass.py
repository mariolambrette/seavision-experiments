import csv, json, re, time
from concurrent.futures import ThreadPoolExecutor
import requests

SRC = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_concept_resolve.json"
OUT = r"D:\marineai\classification-experiments\sw\scratch\fathomnet_concept_review.csv"
UA = {"User-Agent": "SeaVision/1.0 (research; University of Exeter)"}
REST = "https://www.marinespecies.org/rest/AphiaRecordsByName/{}?like=false&marine_only=false"

DROP_EXACT = {"Detritus", "Nano plankton", "detritus"}
UNLABELLED = {"marine organism"}
VERNACULAR = {
    "bony fish": "Actinopterygii", "sponge": "Porifera",
    "brittle star": "Ophiuroidea", "urchin": "Echinoidea",
    "sea star": "Asteroidea", "stony coral": "Scleractinia",
    "coral": "Anthozoa", "crab": "Brachyura", "shrimp": "Caridea",
    "actiniarian": "Actiniaria", "anemone+": "Actiniaria",
    "anemone": "Actiniaria", "soft coral": "Alcyonacea",
    "sea fan": "Alcyonacea", "bivalve": "Bivalvia",
    "sea cucumber": "Holothuroidea", "zoanthid": "Zoantharia",
    "medusae": "Medusozoa", "medusa": "Medusozoa",
    "benthic annelid": "Annelida", "squid": "Decapodiformes",
    "larvacean": "Appendicularia", "pyrosome": "Pyrosomatida",
    "lobed ctenophore": "Lobata", "corallimorph": "Corallimorpharia",
    "sea squirt": "Ascidiacea", "crustacean": "Crustacea",
    "octopus": "Octopoda", "physonect siphonophore": "Physonectae",
    "calycophoran siphonophore": "Calycophorae",
}
SP = re.compile(r"^(?P<base>.+?)\s+(?:sp|spp|cf|aff)\.?\s*[A-Za-z0-9]*$", re.I)
CODE = re.compile(r"^(?P<name>[A-Z]{4,})-\d+$")

cache = {}
def worms(name):
    if not name: return None
    if name in cache: return cache[name]
    rec = None
    for attempt in range(3):
        try:
            r = requests.get(REST.format(requests.utils.quote(name)),
                             headers=UA, timeout=30)
            if r.status_code == 204: break
            recs = r.json()
            acc = [x for x in recs if x.get("status") == "accepted"]
            rec = (acc or recs)[0] if recs else None
            break
        except Exception:
            time.sleep(2 ** attempt)
    cache[name] = rec
    return rec

def split_code(c):
    m = CODE.match(c)
    if not m: return None
    s = m["name"].capitalize()
    for i in range(4, len(s)):          # guess the genus/species boundary
        yield f"{s[:i]} {s[i:].lower()}"

rows = json.load(open(SRC, encoding="utf-8"))
out = []

def handle(r):
    c, n = r["concept"], r["boxes"]
    base = dict(concept=c, boxes=n, action="", method="",
                resolved_name="", aphia_id="", rank="", kingdom="",
                suggestion="", notes="")
    if c in DROP_EXACT:
        return {**base, "action": "drop", "method": "policy",
                "notes": "not an organism"}
    if c in UNLABELLED:
        return {**base, "action": "unlabelled", "method": "policy",
                "notes": "sets has_unlabelled_animal, emits no category"}

    rec = None; method = ""
    if r["ok"]:
        rec, method = worms(r.get("matched") or c), "fathomnet+worms"
    if rec is None:
        rec, method = worms(c), "worms direct"
    if rec is None and c.lower() in VERNACULAR:
        rec, method = worms(VERNACULAR[c.lower()]), "vernacular suggestion"
        base["suggestion"] = VERNACULAR[c.lower()]
    if rec is None:
        m = SP.match(c)
        if m:
            rec, method = worms(m["base"]), "sp. suffix stripped"
            base["suggestion"] = m["base"]
    if rec is None and CODE.match(c):
        for cand in split_code(c):
            rec = worms(cand)
            if rec:
                method, base["suggestion"] = "noaa code split", cand
                break
    if rec is None:
        return {**base, "action": "review", "method": "unresolved"}

    kingdom = rec.get("kingdom") or ""
    phylum = rec.get("phylum") or ""
    rank = rec.get("rank") or ""
    res = {**base, "method": method,
           "resolved_name": rec.get("valid_name") or rec.get("scientificname") or "",
           "aphia_id": rec.get("valid_AphiaID") or rec.get("AphiaID") or "",
           "rank": rank, "kingdom": kingdom}
    if phylum == "Bacillariophyta" or rec.get("class") == "Bacillariophyceae":
        return {**res, "action": "drop", "notes": "diatom"}
    if kingdom and kingdom != "Animalia":
        return {**res, "action": "review", "notes": f"non-Animalia ({kingdom})"}
    if not rank:
        return {**res, "action": "review", "notes": "no rank on WoRMS record"}
    if method != "worms direct" and method != "fathomnet+worms":
        return {**res, "action": "review", "notes": "auto-suggested, confirm"}
    return {**res, "action": "class"}

with ThreadPoolExecutor(max_workers=4) as ex:
    out = list(ex.map(handle, rows))

tally = {}
for r in out:
    tally[r["action"]] = tally.get(r["action"], 0) + 1
print("actions:", tally)
rev = sorted([r for r in out if r["action"] == "review"], key=lambda r: -r["boxes"])
print(f"\nrows needing your eyes: {len(rev)}  "
      f"({sum(1 for r in rev if r['boxes'] >= 20)} with >=20 boxes)")
print("top 30:")
for r in rev[:30]:
    print(f"  {r['boxes']:>7,}  {r['concept']:<38} {r['notes']} "
          f"{('-> ' + r['suggestion']) if r['suggestion'] else ''}")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(sorted(out, key=lambda r: (r["action"], -r["boxes"])))
print("\nwritten:", OUT)
