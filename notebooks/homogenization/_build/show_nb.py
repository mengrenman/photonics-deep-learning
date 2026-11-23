"""Print the text outputs of an executed notebook and save its figures as PNGs for inspection."""
import sys, json, base64, pathlib
nb = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
tag = pathlib.Path(sys.argv[1]).stem[:2]; nfig = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    for o in c.get("outputs", []):
        if o["output_type"] == "stream":
            print(f"--- cell {i} [{o['name']}]"); print("".join(o["text"])[:int(sys.argv[3]) if len(sys.argv) > 3 else 6000])
        elif o["output_type"] == "error":
            print(f"--- cell {i} ERROR", o["ename"], o["evalue"])
        elif "data" in o and "image/png" in o["data"]:
            nfig += 1; p = out / f"{tag}_fig{nfig:02d}_cell{i}.png"; p.write_bytes(base64.b64decode(o["data"]["image/png"])); print(f"--- cell {i} figure -> {p.name}")
