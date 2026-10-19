"""Print the study notes and reports (HTML + KaTeX) to PDF with headless Edge.

    python docs/build_docs.py                 # all
    python docs/build_docs.py 01_homology     # one (prefix match)
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
OUTPUTS = {
    "01_homology_for_tda.html": "01_Homology_for_TDA.pdf",
    "02_persistent_homology_for_tda.html": "02_Persistent_Homology_for_TDA.pdf",
    "06_method_walkthrough.html": "06_Method_Walkthrough.pdf",
}


def build(html: Path, pdf: Path):
    edge = next(e for e in EDGE if Path(e).exists())
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()], check=True, capture_output=True)
    print("->", pdf.name)


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    for src, dst in OUTPUTS.items():
        if src.startswith(want) and (DOCS / src).exists():
            build(DOCS / src, ROOT / dst)
