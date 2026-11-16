"""Convert a percent-format .py (cells split by '# %%' / '# %% [markdown]') into an .ipynb, prepending the shared setup cell."""
import sys, pathlib
import nbformat as nbf

SETUP = pathlib.Path(__file__).with_name("setup_cell.py").read_text().rstrip("\n")

def convert(src, dst):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            lines = [ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln) for ln in text.split("\n")]
            cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))
        elif text == "SETUP":
            cells.append(nbf.v4.new_code_cell(SETUP))
        else:
            cells.append(nbf.v4.new_code_cell(text))
    for line in pathlib.Path(src).read_text().split("\n"):
        if line.startswith("# %%"):
            flush(); buf = []
            kind = "md" if "[markdown]" in line else "code"
        else:
            buf.append(line)
    flush()
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    for i, c in enumerate(nb.cells):
        c["id"] = f"cell-{i:02d}"
    nbf.write(nb, dst)

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
