import json
import sys

nb_path = sys.argv[1]
out_path = sys.argv[2]

nb = json.load(open(nb_path))
parts = []
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    src = src.replace("display(", "print(")
    parts.append(src)

with open(out_path, "w") as f:
    f.write("\n\n".join(parts))
    f.write("\n")

print(f"Wrote {out_path} ({len(parts)} code cells)")
