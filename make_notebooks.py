"""Build the notebooks from source cells (so they are reproducible) and execute them.

    python make_notebooks.py 03        # build + run 03_tda_practice.ipynb
    python make_notebooks.py 04 05     # etc.
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent


def build(name: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    path = ROOT / f"{name}.ipynb"
    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    nbf.write(nb, path)
    print("executed ->", path.name)


if __name__ == "__main__":
    import importlib
    for key in sys.argv[1:]:
        for mod in ("notebooks_src", f"notebooks_src_{key}"):
            try:
                S = importlib.import_module(mod)
            except ModuleNotFoundError:
                continue
            if hasattr(S, f"NB{key}"):
                build(*getattr(S, f"NB{key}"))
                break
