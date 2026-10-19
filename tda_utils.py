"""Shared helpers for the TDA notebooks (Gidea & Katz 2018 pipeline).

    returns  -> sliding-window point clouds -> Vietoris-Rips H1 diagrams
             -> persistence landscapes -> L^p norms -> rolling statistics -> Kendall tau

Everything here is plain numpy / scipy / ripser so the notebooks stay short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ripser import ripser
from scipy import stats


# ----------------------------------------------------------------------------- data
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns r_t = ln(P_t / P_{t-1}) on the common trading days."""
    p = prices.dropna().astype(float)
    return np.log(p / p.shift(1)).dropna()


# ----------------------------------------------------------------------------- homology
def rips_h1(points: np.ndarray, maxdim: int = 1) -> np.ndarray:
    """Finite H1 birth/death pairs of the Vietoris-Rips filtration of a point cloud."""
    dgm = ripser(points, maxdim=maxdim)["dgms"][1]
    return dgm[np.isfinite(dgm[:, 1])] if len(dgm) else dgm


def landscape(dgm: np.ndarray, grid: np.ndarray, k_max: int | None = None) -> np.ndarray:
    """Persistence landscape lambda_1..lambda_k evaluated on `grid` (shape k_max x len(grid)).

    For each (b, d) the tent function f(e) = max(0, min(e - b, d - e)); lambda_k(e) is the
    k-th largest tent value at e (Bubenik 2015). k_max=None keeps every landscape function."""
    if k_max is None:                      # all landscape functions (one per diagram point)
        k_max = max(1, len(dgm))
    if len(dgm) == 0:
        return np.zeros((k_max, len(grid)))
    b, d = dgm[:, 0][:, None], dgm[:, 1][:, None]
    tents = np.maximum(0.0, np.minimum(grid[None, :] - b, d - grid[None, :]))  # n_pairs x grid
    tents = -np.sort(-tents, axis=0)                                           # descending per column
    out = np.zeros((k_max, len(grid)))
    k = min(k_max, tents.shape[0])
    out[:k] = tents[:k]
    return out


def landscape_norm(dgm: np.ndarray, p: int = 1, k_max: int | None = None, n_grid: int = 400) -> float:
    """L^p norm of the landscape: ( sum_k  integral |lambda_k|^p )^(1/p)."""
    if len(dgm) == 0:
        return 0.0
    hi = float(dgm[:, 1].max())
    grid = np.linspace(0.0, hi, n_grid)
    lam = landscape(dgm, grid, k_max)
    return float(np.sum(np.trapezoid(np.abs(lam) ** p, grid, axis=1)) ** (1.0 / p))


def norm_series(returns: pd.DataFrame, w: int = 50, ps=(1, 2), k_max: int | None = None) -> pd.DataFrame:
    """For each trading day t build the cloud {x_{t-w+1}, ..., x_t} and record ||lambda||_p."""
    X = returns.values
    idx = returns.index[w - 1:]
    rows = []
    for t in range(w - 1, len(X)):
        dgm = rips_h1(X[t - w + 1:t + 1])
        rows.append([landscape_norm(dgm, p, k_max) for p in ps] + [len(dgm)])
    cols = [f"L{p}" for p in ps] + ["n_loops"]
    return pd.DataFrame(rows, index=idx, columns=cols)


# ----------------------------------------------------------------------------- early-warning statistics
def _low_freq_density(y: np.ndarray, frac: float = 0.1) -> float:
    """Mean periodogram power over the lowest `frac` of positive frequencies."""
    y = y - y.mean()
    spec = np.abs(np.fft.rfft(y)) ** 2 / len(y)
    spec = spec[1:]                      # drop the zero frequency
    n_low = max(1, int(np.ceil(len(spec) * frac)))
    return float(spec[:n_low].mean())


def rolling_stats(y: pd.Series, L: int = 500) -> pd.DataFrame:
    """Rolling variance, lag-1 autocorrelation and low-frequency spectral density."""
    v = y.rolling(L).var()
    acf = y.rolling(L).apply(lambda a: pd.Series(a).autocorr(1), raw=True)
    lf = y.rolling(L).apply(_low_freq_density, raw=True)
    return pd.DataFrame({"variance": v, "acf1": acf, "lowfreq": lf})


def kendall_trend(z: pd.Series) -> float:
    """Kendall tau between time order and the values of z (NaN-safe)."""
    z = z.dropna()
    if len(z) < 10:
        return np.nan
    return float(stats.kendalltau(np.arange(len(z)), z.values)[0])


def event_taus(stats_df: pd.DataFrame, event_day, n: int = 250) -> pd.Series:
    """Kendall tau of each statistic over the n trading days ending at `event_day`."""
    pos = stats_df.index.get_indexer([pd.Timestamp(event_day)], method="pad")[0]
    win = stats_df.iloc[max(0, pos - n + 1):pos + 1]
    return win.apply(kendall_trend)


def null_taus(stats_df: pd.DataFrame, events, n: int = 250, n_draws: int = 500,
              exclude: int = 250, seed: int = 0) -> pd.DataFrame:
    """Kendall tau over random n-day windows that do not touch the `exclude` days before/after events."""
    rng = np.random.default_rng(seed)
    idx = stats_df.dropna().index
    bad = np.zeros(len(idx), dtype=bool)
    for e in events:
        p = idx.get_indexer([pd.Timestamp(e)], method="pad")[0]
        bad[max(0, p - exclude - n):min(len(idx), p + exclude)] = True
    ok_ends = np.where(~bad)[0]
    ok_ends = ok_ends[ok_ends >= n]
    rows = []
    for end in rng.choice(ok_ends, size=min(n_draws, len(ok_ends)), replace=False):
        win = stats_df.loc[idx[end - n + 1]:idx[end]]
        rows.append(win.apply(kendall_trend))
    return pd.DataFrame(rows)


def empirical_p(obs: float, null: pd.Series) -> float:
    """Share of null draws with tau >= observed tau (one-sided, 'at least this strong a trend')."""
    null = null.dropna()
    return float((null >= obs).mean()) if len(null) else np.nan


def realized_vol(returns: pd.Series, w: int = 20) -> pd.Series:
    return returns.rolling(w).std() * np.sqrt(252)
