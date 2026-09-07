import json, subprocess, sys, datetime
from pathlib import Path
import yaml, torch, timm, numpy as np
from PIL import Image

cfg = yaml.safe_load(open(sys.argv[1]))
out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)

paths = sorted(Path(cfg["image_dir"]).glob("*.jpg"))[: cfg["n_images"]]
assert paths, f"no images found in {cfg['image_dir']}"
half = len(paths) // 2
split = {0: paths[:half], 1: paths[half:]}

def embed(device_idx, files):
    dev = f"cuda:{device_idx}"
    model = timm.create_model(cfg["model"], pretrained=True, num_classes=0).eval().to(dev)
    tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model), is_training=False)
    vecs = []
    with torch.inference_mode():
        for i in range(0, len(files), cfg["batch_size"]):
            batch = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i+cfg["batch_size"]]]).to(dev)
            vecs.append(model(batch).float().cpu().numpy())
    return np.concatenate(vecs)

allv = []
for d in cfg["devices"]:
    v = embed(d, split[d])
    print(f"cuda:{d} -> {v.shape}")
    allv.append(v)
emb = np.concatenate(allv)
np.save(out / "embeddings.npy", emb)

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
json.dump({"git_commit": commit, "config": cfg, "built": datetime.datetime.now().isoformat(),
           "n_vectors": int(emb.shape[0]), "dim": int(emb.shape[1])},
          open(out / "build_manifest.json", "w"), indent=2)
print("wrote", out, emb.shape)
