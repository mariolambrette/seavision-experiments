import json
P = r"D:\marineai\classification-experiments\sw\schema\seavision_schema.json"
s = json.load(open(P, encoding="utf-8"))
ALLOWED = ["1.1", "1.2"]

def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "schema_version" and isinstance(v, dict):
                if "const" in v:
                    print(f"  {path}/{k}: const {v.pop('const')!r} -> enum {ALLOWED}")
                    v["enum"] = list(ALLOWED)
                elif "enum" in v:
                    before = list(v["enum"])
                    v["enum"] = sorted(set(before) | set(ALLOWED))
                    print(f"  {path}/{k}: enum {before} -> {v['enum']}")
            walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}/{i}")

print("changes:")
walk(s)
json.dump(s, open(P, "w", encoding="utf-8"), indent=2)
print("written:", P)
