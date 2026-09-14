import json
P = r"D:\marineai\classification-experiments\sw\schema\seavision_schema.json"
s = json.load(open(P, encoding="utf-8"))

def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "crop_provenance" and isinstance(v, dict) and "enum" in v:
                if "cut_from_frame" not in v["enum"]:
                    v["enum"].append("cut_from_frame")
                    print(f"  {path}/{k}: enum -> {v['enum']}")
            if k in ("countries", "ocean_basins") and isinstance(v, dict):
                if v.get("type") == "array":
                    v["type"] = ["array", "null"]
                    print(f"  {path}/{k}: type -> {v['type']}")
            walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}/{i}")

print("changes:")
walk(s)
json.dump(s, open(P, "w", encoding="utf-8"), indent=2)
print("written:", P)
