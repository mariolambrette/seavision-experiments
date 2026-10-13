#!/usr/bin/env python3
"""Exploratory UMAP maps of the WP9 embeddings, as interactive HTML.

EXPLORATORY ONLY. UMAP keeps local neighbourhoods and distorts everything
else: gaps between clusters, their sizes and their relative positions depend
on the settings and the seed. Nothing here is a result or a decision input;
quantitative questions (WP10 above all) are answered by their own packages.

Two subsets, never mixed (decided 13 October 2026):
  dev    the development split -- free to inform later choices;
  main   the main split -- descriptive only (methods figure E1); anything a
         main-set map suggests must be checked on the dev set before it
         changes a decision.

Points: species-rank crops only, at most --cap per species (so no species,
e.g. FathomNet's red snapper, swamps the map), plus a capped background
sample per source. The SAME points are embedded by every backbone and
geometry, so maps can be compared point for point. Square geometries hold no
FishWIO and no background, so those points are simply absent there.

Per map: L2-normalise -> PCA to 50 dimensions -> UMAP (n_neighbors 30,
min_dist 0.1, fixed seed, so single-threaded and reproducible). In the
viewer every map is rotated and scaled onto the first (Procrustes on the
shared points; reflection allowed, since a UMAP's handedness is arbitrary)
so switching backbone does not spin the picture. Orientation is cosmetic;
this changes nothing about the structure.

Steps (each writes to --work and is skipped if its output exists):
  labels   one row per crops/background record: key, set, source, split,
           deployment, category, rank, species/genus/family/order, short side
  sample   the capped key list per subset
  embed    UMAP coordinates per (subset, backbone, geometry)
  html     one self-contained HTML viewer per subset

    python scratch\\umap_explore.py all --subset dev
    python scratch\\umap_explore.py all --subset main
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract.preprocess import SHARD_SET          # noqa: E402
from extract.readout_spec import readout_arrays   # noqa: E402

TAR_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{6})\.tar$")
DEFAULT_BACKBONES = ["bioclip25_huge", "dinov3_vitl16", "dinov2_vitl14_reg"]
GEOMETRIES = ["letterbox", "distort", "square_m00", "square_m10"]
LABEL_FIELDS = ["key", "set", "source", "split", "deployment", "category_id",
                "rank", "species", "genus", "family", "order", "short_side"]


# ------------------------------------------------------------------ labels
def load_categories(paths):
    cats = {}
    for p in paths:
        print(f"  categories from {os.path.basename(p)} ...", flush=True)
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        for c in doc["categories"]:
            lin = c.get("lineage") or {}
            rank = str(c.get("rank", "")).lower()
            # taxonomy from the lineage, never from the name (WP6a)
            cats[c["id"]] = (rank, c.get("name") if rank == "species" else "",
                             lin.get("genus") or "", lin.get("family") or "",
                             lin.get("order") or "")
        del doc
    return cats


def read_split(split_dir, set_name):
    out = {}
    with gzip.open(os.path.join(split_dir, f"split_{set_name}.csv.gz"), "rt",
                   newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["key"]] = (r["source"], r["split"], r["group"])
    return out


def scan_json(tar_path):
    """key -> record json, reading only the .json members."""
    out = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".json"):
                out[m.name.rsplit(".", 1)[0]] = json.loads(
                    tf.extractfile(m).read())
    return out


def cmd_labels(a, cfg):
    out = os.path.join(a.work, "labels.csv.gz")
    if os.path.exists(out):
        print(f"labels: {out} exists, skipping")
        return
    cats = load_categories(a.coco)
    rows, missing = [], Counter()
    for set_name in ("crops", "background"):
        split = read_split(a.split, set_name)
        d = os.path.join(cfg["shards"], set_name)
        tars = sorted(f for f in os.listdir(d) if TAR_RE.match(f))
        for i, t in enumerate(tars, 1):
            print(f"  {set_name} [{i}/{len(tars)}] {t}", flush=True)
            for key, m in scan_json(os.path.join(d, t)).items():
                if key not in split:
                    missing[set_name] += 1
                    continue
                src, spl, grp = split[key]
                w, h = m.get("native_width"), m.get("native_height")
                short = min(w, h) if w and h else ""
                cid = m.get("category_id")
                rank = sp = ge = fa = od = ""
                if cid is not None and cid in cats:
                    rank, sp, ge, fa, od = cats[cid]
                rows.append([key, set_name, src, spl, grp,
                             "" if cid is None else cid, rank, sp, ge, fa, od,
                             short])
    if missing:
        sys.exit(f"keys absent from the split files: {dict(missing)}")
    os.makedirs(a.work, exist_ok=True)
    tmp = out + ".part"
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(LABEL_FIELDS)
        w.writerows(rows)
    os.replace(tmp, out)
    print(f"labels: {len(rows):,} records -> {out}")


def read_labels(work):
    with gzip.open(os.path.join(work, "labels.csv.gz"), "rt", newline="",
                   encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ sample
def cmd_sample(a, _cfg):
    out = os.path.join(a.work, a.subset, "sample.csv")
    if os.path.exists(out):
        print(f"sample: {out} exists, skipping")
        return
    rows = [r for r in read_labels(a.work) if r["split"] == a.subset]
    rng = random.Random(a.seed)
    by_sp = defaultdict(list)
    bg = defaultdict(list)
    for r in rows:
        if r["set"] == "crops" and r["rank"] == "species" and r["species"]:
            by_sp[r["species"]].append(r)
        elif r["set"] == "background":
            bg[r["source"]].append(r)
    pick = []
    for sp in sorted(by_sp):
        rs = sorted(by_sp[sp], key=lambda r: r["key"])
        pick += rs if len(rs) <= a.cap else rng.sample(rs, a.cap)
    for src in sorted(bg):
        rs = sorted(bg[src], key=lambda r: r["key"])
        pick += rs if len(rs) <= a.bg_cap else rng.sample(rs, a.bg_cap)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        w.writerows(pick)
    n_bg = sum(r["set"] == "background" for r in pick)
    print(f"sample {a.subset}: {len(pick):,} points ({len(by_sp)} species, "
          f"cap {a.cap}; {n_bg:,} background, cap {a.bg_cap} per source) "
          f"-> {out}")


def read_sample(work, subset):
    with open(os.path.join(work, subset, "sample.csv"), newline="",
              encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------- embed
def locate(emb_root, set_name, wanted):
    """key -> (folder, row) for the wanted keys of one set."""
    where = {}
    for stem in sorted(os.listdir(emb_root)):
        d = os.path.join(emb_root, stem)
        if stem.endswith(".part") or not os.path.exists(
                os.path.join(d, "done.json")):
            continue
        with open(os.path.join(d, "keys.txt"), encoding="utf-8") as fh:
            for i, k in enumerate(fh.read().splitlines()):
                if k in wanted:
                    where[k] = (d, i)
    return where


def load_vectors(cfg, backbone, size, geometry, arrays, keys_by_set):
    """-> (keys in order, [n, d] float32 matrix, L2-normalised per half)."""
    root = os.path.join(cfg["out"], backbone, f"{size}px", geometry)
    keys, blocks = [], []
    for set_name, wanted in keys_by_set.items():
        sdir = os.path.join(root, set_name)
        if not wanted or not os.path.isdir(sdir):
            continue
        where = locate(sdir, set_name, set(wanted))
        by_dir = defaultdict(list)
        for k in wanted:
            if k in where:
                by_dir[where[k][0]].append((where[k][1], k))
        for d, items in sorted(by_dir.items()):
            items.sort()
            rows = np.array([i for i, _ in items])
            parts = []
            for n in arrays:
                x = np.load(os.path.join(d, f"{n}.npy"), mmap_mode="r")
                x = np.asarray(x[rows], dtype=np.float32)
                x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
                parts.append(x)
            blocks.append(np.concatenate(parts, axis=1))
            keys += [k for _, k in items]
    return keys, np.concatenate(blocks, axis=0)


def align(ref, xy):
    """Rotate (or reflect) and scale `xy` onto `ref`, fitted on the points
    both maps hold, applied to every point of `xy`. Cosmetic only: it changes
    orientation and scale, never which points are near which."""
    both = ~np.isnan(ref[:, 0]) & ~np.isnan(xy[:, 0])
    a = ref[both] - ref[both].mean(0)
    b0 = xy[both].mean(0)
    b = xy[both] - b0
    u, s, vt = np.linalg.svd(b.T @ a)
    r = u @ vt
    scale = s.sum() / (b ** 2).sum()
    out = xy.copy()
    ok = ~np.isnan(xy[:, 0])
    out[ok] = (xy[ok] - b0) @ r * scale
    return out


def cmd_embed(a, cfg):
    import umap
    from sklearn.decomposition import PCA
    with open(cfg["spec"], encoding="utf-8") as fh:
        spec_all = yaml.safe_load(fh)["backbones"]
    sample = read_sample(a.work, a.subset)
    idx = {r["key"]: i for i, r in enumerate(sample)}
    animal = [r["key"] for r in sample if r["set"] == "crops"]
    bgk = [r["key"] for r in sample if r["set"] == "background"]
    for bb in a.backbones:
        spec = spec_all[bb]
        arrays = readout_arrays(spec["readouts"][:1])   # the WP8 winner
        token = spec["readouts"][0]["token"]
        layer = spec["readouts"][0]["layer"]
        for g in [g for g in GEOMETRIES if g in spec["geometries"]]:
            out = os.path.join(a.work, a.subset, f"umap_{bb}_{g}.npy")
            if os.path.exists(out):
                print(f"  {bb} {g}: exists, skipping")
            else:
                t0 = time.time()
                ksets = {SHARD_SET[g]: animal}
                if SHARD_SET[g] == "crops":
                    ksets["background"] = bgk
                keys, X = load_vectors(cfg, bb, spec["size"], g, arrays,
                                       ksets)
                Xp = PCA(n_components=min(50, X.shape[1]),
                         random_state=a.seed).fit_transform(X)
                xy = umap.UMAP(n_neighbors=30, min_dist=0.1,
                               n_jobs=1, random_state=a.seed).fit_transform(Xp)
                full = np.full((len(sample), 2), np.nan, np.float32)
                full[[idx[k] for k in keys]] = xy
                np.save(out, full)
                print(f"  {bb} {g}: {len(keys):,} points, "
                      f"{time.time() - t0:,.0f}s", flush=True)
            meta = {"backbone": bb, "geometry": g,
                    "readout": f"{token} @ {layer}", "size": spec["size"]}
            with open(out[:-4] + ".json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
    print("embed: done")


# -------------------------------------------------------------------- html
def cmd_html(a, cfg):
    sample = read_sample(a.work, a.subset)
    maps = []
    ref = None
    for bb in a.backbones:
        for g in GEOMETRIES:
            p = os.path.join(a.work, a.subset, f"umap_{bb}_{g}.npy")
            if not os.path.exists(p):
                continue
            xy = np.load(p)
            with open(p[:-4] + ".json", encoding="utf-8") as fh:
                meta = json.load(fh)
            ok = ~np.isnan(xy[:, 0])
            if ref is None:
                xy[ok] = xy[ok] - xy[ok].mean(0)
                ref = xy
            else:
                xy = align(ref, xy)
            coords = [None if np.isnan(v) else round(float(v), 3)
                      for v in xy.reshape(-1)]
            maps.append({**meta, "xy": coords})
    if not maps:
        sys.exit("no maps: run the embed step first")

    def enc(field):
        vals = sorted({r[field] for r in sample})
        ix = {v: i for i, v in enumerate(vals)}
        return vals, [ix[r[field]] for r in sample]

    fields = {}
    for f in ("set", "source", "split", "deployment", "species", "genus",
              "family", "order"):
        fields[f] = dict(zip(("levels", "codes"), enc(f)))
    data = {
        "subset": a.subset, "n": len(sample),
        "keys": [r["key"] for r in sample],
        "short": [int(float(r["short_side"])) if r["short_side"] else None
                  for r in sample],
        "fields": fields, "maps": maps,
        "note": f"cap {a.cap} per species, {a.bg_cap} background per source; "
                f"seed {a.seed}",
    }
    out = os.path.join(a.work, f"umap_{a.subset}.html")
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__TITLE__", f"Embedding maps — {a.subset} set")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"html: {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"{len(maps)} maps, {len(sample):,} points)")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fcfcfb;--ink:#222;--muted:#6b6b6b;--faint:#d4d4d4;--panel:#f2f2f0;--grey:#b9b9b9}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1d;--ink:#eee;--muted:#a3a3a3;--faint:#3a3a3d;--panel:#26262a;--grey:#55555a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px}h1{font-size:17px;margin:0 0 2px}.sub{color:var(--muted);font-size:12px}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;padding:8px 16px;background:var(--panel);align-items:center}
label{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center}
select,input{font:inherit;font-size:13px;padding:3px 6px;background:var(--bg);color:var(--ink);border:1px solid var(--faint);border-radius:4px}
main{display:flex;gap:12px;padding:12px 16px}#wrap{position:relative;flex:1;min-width:0}
canvas{width:100%;height:72vh;display:block;border:1px solid var(--faint);border-radius:6px;cursor:crosshair}
#legend{width:250px;font-size:12px;max-height:72vh;overflow:auto}
#legend .row{display:flex;gap:6px;align-items:center;padding:2px 0;cursor:pointer}#legend .sw{width:11px;height:11px;border-radius:50%;flex:none}
#legend .n{color:var(--muted);margin-left:auto}#legend .off{opacity:.35}
#tip{position:absolute;pointer-events:none;background:var(--bg);border:1px solid var(--faint);border-radius:6px;padding:6px 8px;font-size:12px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.15);max-width:340px}
#tip b{font-weight:600}.warn{color:var(--muted);font-size:12px;padding:0 16px 12px}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub" id="sub"></div></header>
<div class="controls">
 <label>Backbone <select id="bb"></select></label>
 <label>Geometry <select id="geo"></select></label>
 <label>Colour by <select id="col">
  <option value="source">Source</option><option value="set">Animal / background</option>
  <option value="order">Order</option><option value="family">Family</option>
  <option value="genus">Genus</option><option value="species">Species</option>
  <option value="deployment">Deployment</option><option value="short">Crop short side (px)</option></select></label>
 <label>Focus on <input id="focus" list="names" placeholder="genus, family or species" size="22"></label>
 <datalist id="names"></datalist>
 <label>Source <select id="src"><option value="">all</option></select></label>
 <label>Point size <input id="ps" type="range" min="1" max="6" value="2"></label>
</div>
<main><div id="wrap"><canvas id="cv"></canvas><div id="tip"></div></div><div id="legend"></div></main>
<div class="warn">Exploratory only: UMAP keeps local neighbourhoods and distorts the rest. Distances between clusters, cluster sizes and positions are not meaningful, and maps are rotated onto the first for comparability. Colours mark the 8 largest groups among the points in focus; the rest are grey. OzFish background is ~18% contaminated with unboxed fish (WP7).</div>
<script>
const D=__DATA__;
const PAL=["#0072B2","#E69F00","#009E73","#56B4E9","#D55E00","#CC79A7","#6A3D9A","#A35C1E"];
const SEQ=["#deebf7","#9ecae1","#4292c6","#2171b5","#08519c","#08306b"];
const $=id=>document.getElementById(id);
const cv=$("cv"),ctx=cv.getContext("2d"),tip=$("tip");
$("sub").textContent=`${D.n.toLocaleString()} points · ${D.note} · ${D.maps.length} maps`;
const bbs=[...new Set(D.maps.map(m=>m.backbone))];
bbs.forEach(b=>$("bb").add(new Option(b,b)));
function geos(){const s=$("geo"),cur=s.value;s.innerHTML="";D.maps.filter(m=>m.backbone==$("bb").value).forEach(m=>s.add(new Option(m.geometry+" · "+m.readout+" · "+m.size,m.geometry)));if([...s.options].some(o=>o.value==cur))s.value=cur;}
const F=D.fields;F.source.levels.forEach(v=>$("src").add(new Option(v,v)));
const names=new Set();["genus","family","species"].forEach(f=>F[f].levels.forEach(v=>v&&names.add(v)));
[...names].sort().forEach(v=>{const o=document.createElement("option");o.value=v;$("names").appendChild(o)});
let view={k:1,x:0,y:0},hidden=new Set(),cur=null,colours=null,grid=null;
function val(f,i){return f=="short"?D.short[i]:F[f].levels[F[f].codes[i]]}
function inFocus(i){const q=$("focus").value.trim();if(q&&!(val("genus",i)==q||val("family",i)==q||val("species",i)==q))return false;
 const s=$("src").value;return !s||val("source",i)==s}
function build(){
 cur=D.maps.find(m=>m.backbone==$("bb").value&&m.geometry==$("geo").value);
 const f=$("col").value,cnt=new Map();colours=new Array(D.n);
 for(let i=0;i<D.n;i++){if(cur.xy[2*i]==null||!inFocus(i))continue;const v=val(f,i);if(f!="short")cnt.set(v,(cnt.get(v)||0)+1)}
 let leg=[];
 if(f=="short"){const br=[32,64,128,256,512];const lab=["<32","32–63","64–127","128–255","256–511","≥512"];
  for(let i=0;i<D.n;i++){const s=D.short[i];if(s==null)continue;let b=br.findIndex(x=>s<x);if(b<0)b=5;colours[i]=SEQ[b]}
  leg=lab.map((l,j)=>({v:l,c:SEQ[j],n:null}))}
 else{const top=[...cnt.entries()].filter(e=>e[0]!=="").sort((a,b)=>b[1]-a[1]).slice(0,8);
  const cmap=new Map(top.map((e,j)=>[e[0],PAL[j]]));
  for(let i=0;i<D.n;i++){const c=cmap.get(val(f,i));if(c)colours[i]=c}
  leg=top.map((e,j)=>({v:e[0],c:PAL[j],n:e[1]}));const rest=[...cnt.values()].reduce((a,b)=>a+b,0)-top.reduce((a,e)=>a+e[1],0);
  if(rest)leg.push({v:"other / unlabelled",c:"var(--grey)",n:rest})}
 const L=$("legend");L.innerHTML=`<div style="color:var(--muted);margin-bottom:4px">${$("col").selectedOptions[0].text}${$("focus").value?" · focus "+$("focus").value:""}</div>`;
 leg.forEach(e=>{const r=document.createElement("div");r.className="row"+(hidden.has(e.v)?" off":"");
  r.innerHTML=`<span class="sw" style="background:${e.c}"></span><span>${e.v||"(none)"}</span><span class="n">${e.n==null?"":e.n.toLocaleString()}</span>`;
  if(f!="short"&&e.n!=null&&e.c.startsWith("#"))r.onclick=()=>{hidden.has(e.v)?hidden.delete(e.v):hidden.add(e.v);build()};L.appendChild(r)});
 fit();draw()}
let bounds=null;
function fit(){let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;for(let i=0;i<D.n;i++){const x=cur.xy[2*i],y=cur.xy[2*i+1];if(x==null)continue;x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y)}bounds=[x0,x1,y0,y1]}
function toPx(x,y){const w=cv.width,h=cv.height,[x0,x1,y0,y1]=bounds,s=Math.min(w/(x1-x0),h/(y1-y0))*0.94*view.k;
 return[(x-(x0+x1)/2)*s+w/2+view.x,-(y-(y0+y1)/2)*s+h/2+view.y]}
function draw(){const dpr=devicePixelRatio||1;cv.width=cv.clientWidth*dpr;cv.height=cv.clientHeight*dpr;ctx.clearRect(0,0,cv.width,cv.height);
 const r=+$("ps").value*dpr,f=$("col").value,grey=getComputedStyle(document.body).getPropertyValue("--grey").trim();grid=new Map();const G=12*dpr;
 const pass=(coloured)=>{for(let i=0;i<D.n;i++){const x=cur.xy[2*i];if(x==null)continue;const foc=inFocus(i);let c=foc?colours[i]:null;
  if(c&&hidden.has(f=="short"?null:val(f,i)))c=null;if(!!c!==coloured)continue;
  const[p,q]=toPx(x,cur.xy[2*i+1]);ctx.fillStyle=c||grey;ctx.globalAlpha=c?0.85:(foc?0.5:0.18);
  ctx.beginPath();ctx.arc(p,q,c?r:Math.max(1,r*0.7),0,6.283);ctx.fill();
  if(foc){const gk=Math.floor(p/G)+","+Math.floor(q/G);(grid.get(gk)||grid.set(gk,[]).get(gk)).push(i)}}};
 pass(false);pass(true);ctx.globalAlpha=1}
cv.addEventListener("mousemove",ev=>{const dpr=devicePixelRatio||1,rect=cv.getBoundingClientRect(),px=(ev.clientX-rect.left)*dpr,py=(ev.clientY-rect.top)*dpr,G=12*dpr;
 if(drag){view.x+=(ev.clientX-drag[0])*dpr;view.y+=(ev.clientY-drag[1])*dpr;drag=[ev.clientX,ev.clientY];draw();return}
 let best=-1,bd=(10*dpr)**2;const gx=Math.floor(px/G),gy=Math.floor(py/G);
 for(let a=-1;a<=1;a++)for(let b=-1;b<=1;b++)for(const i of grid.get((gx+a)+","+(gy+b))||[]){const[p,q]=toPx(cur.xy[2*i],cur.xy[2*i+1]);const d=(p-px)**2+(q-py)**2;if(d<bd){bd=d;best=i}}
 if(best<0){tip.style.display="none";return}const i=best;
 tip.innerHTML=`<b>${val("species",i)||(val("set",i)=="background"?"background":"(no species)")}</b><br>genus ${val("genus",i)||"–"} · family ${val("family",i)||"–"}<br>order ${val("order",i)||"–"}<br>source ${val("source",i)} · deployment ${val("deployment",i)||"–"}<br>short side ${D.short[i]??"–"} px<br><span style="color:var(--muted)">${D.keys[i]}</span>`;
 tip.style.display="block";tip.style.left=Math.min(ev.clientX-rect.left+14,rect.width-300)+"px";tip.style.top=(ev.clientY-rect.top+14)+"px"});
cv.addEventListener("mouseleave",()=>tip.style.display="none");
let drag=null;cv.addEventListener("mousedown",ev=>drag=[ev.clientX,ev.clientY]);addEventListener("mouseup",()=>drag=null);
cv.addEventListener("wheel",ev=>{ev.preventDefault();const z=ev.deltaY<0?1.15:1/1.15;const dpr=devicePixelRatio||1,rect=cv.getBoundingClientRect();
 const px=(ev.clientX-rect.left)*dpr-cv.width/2,py=(ev.clientY-rect.top)*dpr-cv.height/2;view.x=px-(px-view.x)*z;view.y=py-(py-view.y)*z;view.k*=z;draw()},{passive:false});
cv.addEventListener("dblclick",()=>{view={k:1,x:0,y:0};draw()});
$("bb").onchange=()=>{geos();build()};["geo","col","src"].forEach(id=>$(id).onchange=()=>{hidden.clear();build()});
$("focus").onchange=()=>{hidden.clear();build()};$("ps").oninput=draw;addEventListener("resize",draw);
geos();build();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "sample", "embed", "html", "all"])
    ap.add_argument("--config", default="configs/extract/wp9.yaml")
    ap.add_argument("--subset", choices=["dev", "main"], default="dev")
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    ap.add_argument("--split",
                    default="D:/marineai/classification-experiments/splits/dev_v1")
    ap.add_argument("--coco", nargs="+", default=[
        "D:/marineai/dataset/collated/seavision.json",
        "D:/marineai/dataset/collated/seavision_fathomnet.json"])
    ap.add_argument("--work", default="D:/marineai/scratch/umap_explore")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--bg-cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261013)
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    steps = ["labels", "sample", "embed", "html"] if a.cmd == "all" else [a.cmd]
    for s in steps:
        {"labels": cmd_labels, "sample": cmd_sample, "embed": cmd_embed,
         "html": cmd_html}[s](a, cfg)


if __name__ == "__main__":
    main()
