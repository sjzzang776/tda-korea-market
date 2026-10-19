# TDA ― homology, persistent homology, and a study of the Gidea–Katz pipeline

> Study notes on topological data analysis: the part of homology (Hatcher, Ch. 2) that TDA needs, persistent homology, and a step-by-step look at **why** Gidea & Katz (2018) build their "TDA for stock prices" pipeline the way they do, checked with small point-cloud experiments.
> October 2026 · personal study · Python (ripser, persim, numpy, matplotlib). All figures are drawn by the code in this repository (`figures/make_figures.py` and the notebooks).

## Contents

| File | What it covers |
|---|---|
| [`01_Homology_for_TDA.pdf`](01_Homology_for_TDA.pdf) | Note 1: simplices, simplicial complexes, chains, the boundary map, homology groups, Betti numbers, coefficients ($\mathbb{Z}$ vs $\mathbb{Z}/2$), maps induced by inclusions, homotopy invariance. Only the parts of Hatcher Ch. 2 that TDA uses; statements without proofs |
| [`02_Persistent_Homology_for_TDA.pdf`](02_Persistent_Homology_for_TDA.pdf) | Note 2: Čech and Vietoris–Rips complexes and the nerve theorem, filtrations, persistence modules, birth and death, barcodes and diagrams, the structure theorem, bottleneck distance and stability, persistence landscapes and their $L^p$ norms, time series → point clouds |
| [`06_Method_Walkthrough.pdf`](06_Method_Walkthrough.pdf) | Note 3: each step of the TDA-for-prices pipeline asked as "why this?": log returns → sliding window (choosing $w$) → Rips and $H_1$ (compared with $H_0$) → landscapes (they can be averaged) → norms (closed forms, the $c^2$ scale effect, standardization choices) → variance / ACF / spectral power (critical slowing down, an AR(1) experiment) → Kendall $\tau$ (why a null distribution is needed) → Wasserstein change, zigzag persistence |
| [`03_tda_practice.ipynb`](03_tda_practice.ipynb) | Practice: Betti numbers from $\mathbb{Z}/2$ boundary matrices, Rips persistence of a circle, sphere, torus and annulus, stability (bottleneck), a check of the landscape-norm closed forms, synthetic time series |
| [`06_method_walkthrough.ipynb`](06_method_walkthrough.ipynb) | Every experiment and figure of Note 3, with outputs |
| `tda_utils.py` | Shared code: log returns, Rips $H_1$, landscapes, $L^p$ norms, rolling statistics, Kendall $\tau$, random-window null distributions |
| `figures/` | Figures for the notes (`make_figures.py`) and figures saved by the notebooks |
| `docs/` | HTML sources of the notes (KaTeX) and `build_docs.py` (headless Edge → PDF) |
| `notebooks_src*.py`, `make_notebooks.py` | Notebook cell sources and the build-and-run script |
| `data/download.py` | Downloads KOSPI, KOSPI 200 and KOSDAQ closes (FinanceDataReader) for the real-data example at the end of the notebooks |

## What Note 3 checks (summary)

| Step | Why | Watch out for |
|---|---|---|
| Log returns, one point per day | Distances become comparable across time and across axes | Rescale axes that have different units |
| Sliding window, $w\approx50$ | One cloud per day. In the experiment, a larger $w$ means less noise but more lag (20 → 200 days: noise 1.81 → 0.26, lag 36 → 74 days) | Report at least two values of $w$ |
| Vietoris–Rips, $H_1$ | Needs only distances and stays within a factor of 2 of Čech. $H_0$ mostly measures spread (volatility); $H_1$ sees shape | Convention: an edge is present when $d\le\epsilon$ |
| Landscape, $\|\lambda\|_1=\tfrac14\sum\ell^2$, $\|\lambda\|_2=(\tfrac1{12}\sum\ell^3)^{1/2}$ | Diagrams cannot be averaged, functions can. $L^1$ counts every loop, $L^2$ mostly the large ones | Returns $\times c$ → norm $\times c^2$, so a volatility spike is a norm spike |
| Rolling variance / ACF(1) / low-frequency power, Kendall $\tau$ | Critical slowing down; a rank-based trend test | Because windows overlap, even iid returns give $\|\tau\|>0.8$ in 16% of random windows, so always report a random-window null |

## Running it

```bash
pip install ripser persim finance-datareader pandas scipy matplotlib nbformat nbconvert ipykernel
python data/download.py              # real-data example (optional)
python figures/make_figures.py       # figures for the notes
python make_notebooks.py 03 06       # build and run the notebooks
python docs/build_docs.py            # PDFs (needs Microsoft Edge)
```

## References

- A. Hatcher (2002). *Algebraic Topology*, Ch. 2. · H. Edelsbrunner, J. Harer (2010). *Computational Topology*.
- P. Bubenik (2015). Statistical topological data analysis using persistence landscapes. *JMLR* 16.
- M. Gidea, Y. Katz (2018). Topological data analysis of financial time series: Landscapes of crashes. *Physica A* 491.
- M. Scheffer et al. (2009). Early-warning signals for critical transitions. *Nature* 461. · G. Carlsson, V. de Silva (2010). Zigzag persistence. · U. Bauer (2021). Ripser.
