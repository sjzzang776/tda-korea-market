"""Self-made figures for the two study notes (01 homology, 02 TDA).

    python figures/make_figures.py      # run from the TDA folder -> figures/*.png

All drawings are generated here with matplotlib (no images copied from books or papers).
Persistence computations in the figures use ripser on small hand-placed point sets.
"""
from pathlib import Path
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch, FancyArrowPatch
from ripser import ripser
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tda_utils import landscape_norm

OUT = Path(__file__).resolve().parent
INK, SUB, HAIR = "#14202B", "#5A6876", "#C9D1DA"
BLUE, BLUE_SOFT = "#2B50E0", "#DCE3FB"
AMBER, AMBER_SOFT = "#E0700B", "#FBE7D2"
GREEN, GREEN_SOFT = "#0F8A5F", "#D9F0E6"
RED = "#D23F37"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": HAIR, "axes.labelcolor": SUB,
    "xtick.color": SUB, "ytick.color": SUB, "axes.titleweight": "bold", "axes.titlesize": 12,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def clean(ax, keep_axes=False):
    if not keep_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax.set_aspect("equal")


def arrow(ax, p, q, color=INK, lw=1.6, shrink=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 shrinkA=shrink, shrinkB=shrink))


def dot(ax, p, label=None, color=INK, dx=0.0, dy=-0.16, size=38):
    ax.scatter([p[0]], [p[1]], s=size, color=color, zorder=5)
    if label:
        ax.text(p[0] + dx, p[1] + dy, label, ha="center", va="center", fontsize=11, color=INK, zorder=6)


# ----------------------------------------------------------------------------- 01 simplices
def fig_simplices():
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    # 0
    ax = axes[0]; dot(ax, (0, 0), "$v_0$"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    # 1
    ax = axes[1]; p, q = (-0.8, 0), (0.8, 0)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=2.2); arrow(ax, (0.1, 0), (0.25, 0), BLUE)
    dot(ax, p, "$v_0$"); dot(ax, q, "$v_1$"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1, 1)
    # 2
    ax = axes[2]; V = np.array([[-0.9, -0.6], [0.9, -0.6], [0, 0.9]])
    ax.add_patch(Polygon(V, closed=True, facecolor=BLUE_SOFT, edgecolor=BLUE, lw=2.2))
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        m = (V[i] + V[j]) / 2; d = (V[j] - V[i]) / np.linalg.norm(V[j] - V[i]) * 0.12
        arrow(ax, m - d, m + d, BLUE)
    for i, (dx, dy) in enumerate([(-0.15, -0.15), (0.15, -0.15), (0, 0.18)]):
        dot(ax, V[i], f"$v_{i}$", dx=dx, dy=dy)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.1, 1.3)
    # 3
    ax = axes[3]; V = np.array([[-0.9, -0.5], [0.95, -0.35], [0.1, 1.0], [0.15, -0.05]])
    for tri in [(0, 1, 3), (1, 2, 3), (0, 2, 3)]:
        ax.add_patch(Polygon(V[list(tri)], closed=True, facecolor=BLUE_SOFT, edgecolor="none", alpha=0.9))
    ax.add_patch(Polygon(V[[0, 1, 2]], closed=True, facecolor="none", edgecolor=BLUE, lw=2.2))
    for i in range(3):
        ax.plot([V[i, 0], V[3, 0]], [V[i, 1], V[3, 1]], color=BLUE, lw=1.6, ls="--" if i == 2 else "-")
    for i, (dx, dy) in enumerate([(-0.15, -0.15), (0.15, -0.15), (0, 0.18), (0.22, 0.08)]):
        dot(ax, V[i], f"$v_{i}$", dx=dx, dy=dy)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.0, 1.35)
    for ax, t in zip(axes, ["0-simplex (vertex)", "1-simplex (edge)", "2-simplex (triangle)", "3-simplex (tetrahedron)"]):
        clean(ax); ax.set_title(t)
    fig.savefig(OUT / "fig_simplices.png"); plt.close(fig)


def fig_faces_boundary():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    V = np.array([[-0.9, -0.6], [0.9, -0.6], [0, 0.9]])
    # left: the simplex with oriented edges
    ax = axes[0]
    ax.add_patch(Polygon(V, closed=True, facecolor=BLUE_SOFT, edgecolor=BLUE, lw=2.2))
    for i, j, lab, off in [(0, 1, "$[v_0,v_1]$", (0, -0.2)), (1, 2, "$[v_1,v_2]$", (0.42, 0.1)), (0, 2, "$[v_0,v_2]$", (-0.42, 0.1))]:
        m = (V[i] + V[j]) / 2; d = (V[j] - V[i]) / np.linalg.norm(V[j] - V[i]) * 0.12
        arrow(ax, m - d, m + d, BLUE); ax.text(m[0] + off[0], m[1] + off[1], lab, ha="center", fontsize=10, color=SUB)
    for i, (dx, dy) in enumerate([(-0.15, -0.15), (0.15, -0.15), (0, 0.18)]):
        dot(ax, V[i], f"$v_{i}$", dx=dx, dy=dy)
    ax.set_title("$\\sigma=[v_0,v_1,v_2]$ and its three faces")
    # middle: boundary = signed sum of faces
    ax = axes[1]
    for i, j, sign, col in [(1, 2, "+", GREEN), (0, 2, "−", RED), (0, 1, "+", GREEN)]:
        ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], color=col, lw=2.6)
        m = (V[i] + V[j]) / 2; d = (V[j] - V[i]) / np.linalg.norm(V[j] - V[i]) * 0.12
        arrow(ax, m - d, m + d, col)
        n = np.array([-(V[j] - V[i])[1], (V[j] - V[i])[0]]); n = n / np.linalg.norm(n) * 0.22
        ax.text(m[0] - n[0], m[1] - n[1], sign, ha="center", va="center", fontsize=14, color=col, weight="bold")
    for i, (dx, dy) in enumerate([(-0.15, -0.15), (0.15, -0.15), (0, 0.18)]):
        dot(ax, V[i], f"$v_{i}$", dx=dx, dy=dy)
    ax.set_title("$\\partial\\sigma=[v_1,v_2]-[v_0,v_2]+[v_0,v_1]$")
    # right: boundary of the boundary cancels at vertices
    ax = axes[2]
    for i, j in [(1, 2), (0, 2), (0, 1)]:
        ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], color=HAIR, lw=2.0)
    signs = {0: ["−", "+"], 1: ["+", "−"], 2: ["+", "−"]}   # (from the two incident oriented edges)
    pos = {0: [(-0.33, -0.05), (-0.33, 0.18)], 1: [(0.33, -0.05), (0.33, 0.18)], 2: [(-0.2, 0.22), (0.2, 0.22)]}
    for i in range(3):
        dot(ax, V[i], f"$v_{i}$", dx=0, dy=-0.2 if i < 2 else 0.4)
        for s, (dx, dy) in zip(signs[i], pos[i]):
            ax.text(V[i, 0] + dx, V[i, 1] + dy, s, ha="center", va="center", fontsize=14,
                    color=GREEN if s == "+" else RED, weight="bold")
    ax.text(0, 0.05, "each vertex gets\n$+1$ and $-1$ → total $0$", ha="center", va="center", fontsize=10, color=SUB)
    ax.set_title("$\\partial(\\partial\\sigma)=0$")
    for ax in axes:
        clean(ax); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.1, 1.35)
    fig.savefig(OUT / "fig_faces_boundary.png"); plt.close(fig)


def fig_complex_chains():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    # complex: hollow triangle (a,b,c) -- edge -- filled triangle (d,e,f); isolated vertex g
    P = {"a": (-2.6, -0.6), "b": (-1.0, -0.6), "c": (-1.8, 0.8), "d": (0.4, -0.6), "e": (2.0, -0.6), "f": (1.2, 0.8), "g": (2.9, 0.9)}
    edges = [("a", "b"), ("b", "c"), ("a", "c"), ("b", "d"), ("d", "e"), ("e", "f"), ("d", "f")]
    for ax in axes:
        ax.add_patch(Polygon([P["d"], P["e"], P["f"]], closed=True, facecolor=BLUE_SOFT, edgecolor="none"))
        for u, v in edges:
            ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=INK, lw=1.6)
        for k, p in P.items():
            dot(ax, p, k, dy=-0.22 if p[1] < 0 else 0.24)
        clean(ax); ax.set_xlim(-3.1, 3.4); ax.set_ylim(-1.9, 1.4)
    ax = axes[0]
    ax.set_title("A simplicial complex: 7 vertices, 7 edges, 1 triangle")
    ax.text(0.15, -1.45, "$b_0=2$ (two pieces), $b_1=1$ (one hole), $b_2=0$", ha="center", fontsize=10.5, color=SUB)
    ax = axes[1]
    for u, v in [("a", "b"), ("b", "c"), ("c", "a")]:
        ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=AMBER, lw=3.2)
    for u, v in [("d", "e"), ("e", "f"), ("f", "d")]:
        ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=GREEN, lw=3.2)
    ax.set_title("Two 1-cycles")
    ax.text(-1.8, -1.5, "cycle, not a boundary\n→ nonzero class in $H_1$", ha="center", fontsize=10, color=AMBER)
    ax.text(1.2, -1.5, "cycle = $\\partial$(triangle)\n→ zero in $H_1$", ha="center", fontsize=10, color=GREEN)
    fig.savefig(OUT / "fig_complex_chains.png"); plt.close(fig)


def fig_betti_shapes():
    fig = plt.figure(figsize=(11, 3.9))
    ax = fig.add_subplot(1, 3, 1)
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=2.4)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); clean(ax); ax.set_title("circle $S^1$")
    ax.text(0, -1.3, "$b_0=1,\\;b_1=1,\\;b_2=0$", ha="center", fontsize=11, color=SUB)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 24))
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=BLUE_SOFT, edgecolor=BLUE, lw=0.3, alpha=0.9)
    ax.set_title("sphere $S^2$"); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    ax.text2D(0.5, -0.02, "$b_0=1,\\;b_1=0,\\;b_2=1$", transform=ax.transAxes, ha="center", fontsize=11, color=SUB)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 48), np.linspace(0, 2 * np.pi, 24))
    R, r = 1.0, 0.4
    ax.plot_surface((R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v),
                    color=BLUE_SOFT, edgecolor=BLUE, lw=0.3, alpha=0.9)
    ax.set_title("torus $T^2$"); ax.set_axis_off(); ax.set_box_aspect((1, 1, 0.4)); ax.view_init(35, 30)
    ax.text2D(0.5, -0.02, "$b_0=1,\\;b_1=2,\\;b_2=1$", transform=ax.transAxes, ha="center", fontsize=11, color=SUB)
    fig.savefig(OUT / "fig_betti_shapes.png"); plt.close(fig)


def fig_induced_map():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    S = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for ax, filled in zip(axes, [False, True]):
        if filled:
            ax.add_patch(Polygon(S[[0, 1, 2]], closed=True, facecolor=BLUE_SOFT, edgecolor="none"))
            ax.add_patch(Polygon(S[[0, 2, 3]], closed=True, facecolor=BLUE_SOFT, edgecolor="none"))
            ax.plot([S[0, 0], S[2, 0]], [S[0, 1], S[2, 1]], color=INK, lw=1.4)
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([S[i, 0], S[j, 0]], [S[i, 1], S[j, 1]], color=AMBER, lw=3)
        for i, p in enumerate(S):
            dot(ax, p, f"$v_{i}$", dy=-0.25 if p[1] < 0 else 0.25)
        clean(ax); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    axes[0].set_title("$K$: the loop $z$ gives $[z]\\neq 0$ in $H_1(K)$")
    axes[1].set_title("$K'\\supset K$: $z=\\partial$(two triangles), so $i_*[z]=0$")
    fig.text(0.5, -0.02, "The inclusion $i:K\\hookrightarrow K'$ induces $i_*:H_1(K)\\to H_1(K')$; this class is born in $K$ and dies in $K'$.",
             ha="center", fontsize=10.5, color=SUB)
    fig.savefig(OUT / "fig_induced_map.png"); plt.close(fig)


# ----------------------------------------------------------------------------- 02 TDA
PTS = np.array([[0.0, 0.0], [1.0, 0.15], [1.9, 0.9], [1.7, 2.0], [0.7, 2.4], [-0.3, 1.6],
                [3.6, 2.3], [4.1, 2.5]])      # a rough ring of 6 points + a close pair


def rips_simplices(P, eps):
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    E = [(i, j) for i, j in itertools.combinations(range(len(P)), 2) if D[i, j] <= eps]
    T = [(i, j, k) for i, j, k in itertools.combinations(range(len(P)), 3) if max(D[i, j], D[j, k], D[i, k]) <= eps]
    return E, T


def draw_rips(ax, P, eps, balls=True, title=None):
    E, T = rips_simplices(P, eps)
    if balls:
        for p in P:
            ax.add_patch(Circle(p, eps / 2, facecolor=BLUE, alpha=0.08, edgecolor=BLUE, lw=0.6))
    for t in T:
        ax.add_patch(Polygon(P[list(t)], closed=True, facecolor=BLUE_SOFT, edgecolor="none", alpha=0.9))
    for i, j in E:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=BLUE, lw=1.5)
    ax.scatter(P[:, 0], P[:, 1], s=26, color=INK, zorder=5)
    clean(ax); ax.set_xlim(-1.6, 5.4); ax.set_ylim(-1.4, 3.8)
    if title:
        ax.set_title(title)


def fig_pointcloud_balls():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, eps in zip(axes, [0.8, 1.4, 2.2]):
        draw_rips(ax, PTS, eps, title=f"$\\epsilon={eps}$")
    fig.text(0.5, -0.02, "An edge appears when the distance is $\\leq\\epsilon$ (the two balls of radius $\\epsilon/2$ touch); "
             "a triangle appears when all three of its edges are present.", ha="center", fontsize=10.5, color=SUB)
    fig.savefig(OUT / "fig_pointcloud_balls.png"); plt.close(fig)


def fig_cech_vs_rips():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    P = np.array([[0, 0], [2, 0], [1, np.sqrt(3)]]); eps = 2.0   # side = 2 = eps, circumradius 1.155 > eps/2
    ax = axes[0]
    for p in P:
        ax.add_patch(Circle(p, eps / 2, facecolor=BLUE, alpha=0.1, edgecolor=BLUE, lw=0.8))
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=BLUE, lw=1.6)
    ax.scatter(P[:, 0], P[:, 1], s=30, color=INK, zorder=5)
    ax.scatter([1], [np.sqrt(3) / 3], s=40, marker="x", color=RED, zorder=6)
    ax.text(1, -0.75, "no point lies in all three balls", ha="center", fontsize=9.5, color=RED)
    ax.set_title("Čech complex: a hole stays")
    ax = axes[1]
    ax.add_patch(Polygon(P, closed=True, facecolor=BLUE_SOFT, edgecolor=BLUE, lw=1.6))
    ax.scatter(P[:, 0], P[:, 1], s=30, color=INK, zorder=5)
    ax.text(1, -0.75, "all three pairs are within $\\epsilon$ → fill the triangle", ha="center", fontsize=9.5, color=SUB)
    ax.set_title("Vietoris–Rips complex: no hole")
    for ax in axes:
        clean(ax); ax.set_xlim(-1.3, 3.3); ax.set_ylim(-1.2, 3.0)
    fig.savefig(OUT / "fig_cech_vs_rips.png"); plt.close(fig)


def fig_filtration_barcode():
    res = ripser(PTS, maxdim=1)["dgms"]
    h0, h1 = res[0], res[1]
    eps_list = [0.6, 1.2, 1.6, 2.4]
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])
    for k, eps in enumerate(eps_list):
        ax = fig.add_subplot(gs[0, k]); draw_rips(ax, PTS, eps, title=f"$\\epsilon={eps}$")
    ax = fig.add_subplot(gs[1, :])
    y = 0
    xmax = 3.2
    for b, d in sorted(h0, key=lambda x: -x[1] if np.isfinite(x[1]) else -1e9):
        d_ = xmax if not np.isfinite(d) else d
        ax.plot([b, d_], [y, y], color=SUB, lw=3); y += 1
    y0 = y + 0.5
    for b, d in sorted(h1, key=lambda x: -(x[1] - x[0])):
        ax.plot([b, d], [y, y], color=AMBER, lw=4); y += 1
    for eps in eps_list:
        ax.axvline(eps, color=HAIR, lw=1, ls="--")
    ax.set_xlim(0, xmax); ax.set_ylim(-0.7, y + 0.2); ax.set_yticks([])
    ax.set_xlabel("$\\epsilon$"); ax.set_title("Barcode of the same point set: gray = $H_0$ (components), orange = $H_1$ (loops)", loc="left")
    ax.text(xmax - 0.05, 0, "→ lives forever", ha="right", va="center", fontsize=9, color=SUB)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_filtration_barcode.png"); plt.close(fig)
    return h0, h1


def fig_diagram(h0, h1):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    m = 3.0
    ax.fill_between([0, m], [0, m], [0, 0], color="#F2F4F7", zorder=0)
    ax.plot([0, m], [0, m], color=SUB, lw=1)
    f0 = h0[np.isfinite(h0[:, 1])]
    ax.scatter(f0[:, 0], f0[:, 1], s=34, color=SUB, label="$H_0$", zorder=4)
    ax.scatter([0], [m - 0.1], s=60, marker="^", color=SUB, zorder=4)
    ax.text(0.12, m - 0.12, "$\\infty$", va="center", fontsize=10, color=SUB)
    ax.scatter(h1[:, 0], h1[:, 1], s=46, color=AMBER, label="$H_1$", zorder=4)
    for b, d in h1:
        ax.annotate("", xy=(b, d), xytext=(b, b), arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.8, ls=":"))
    ax.set_xlim(-0.05, m); ax.set_ylim(-0.05, m); ax.set_aspect("equal")
    ax.set_xlabel("birth $b$"); ax.set_ylabel("death $d$"); ax.legend(frameon=False, loc="lower right")
    ax.set_title("Persistence diagram")
    ax.text(1.9, 0.9, "near the diagonal:\nshort life, noise", fontsize=9, color=SUB, ha="center")
    fig.savefig(OUT / "fig_diagram.png"); plt.close(fig)


def fig_landscape():
    pts = np.array([[0.5, 2.5], [1.0, 1.8], [1.6, 2.2]])
    grid = np.linspace(0, 3, 400)
    tents = np.array([np.maximum(0, np.minimum(grid - b, d - grid)) for b, d in pts])
    lam = -np.sort(-tents, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    ax.fill_between([0, 3], [0, 3], [0, 0], color="#F2F4F7"); ax.plot([0, 3], [0, 3], color=SUB, lw=1)
    cols = [BLUE, AMBER, GREEN]
    for (b, d), c in zip(pts, cols):
        ax.scatter([b], [d], s=50, color=c, zorder=4)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect("equal"); ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title("three points $(b_j,d_j)$")
    ax = axes[1]
    for t, c, (b, d) in zip(tents, cols, pts):
        ax.plot(grid, t, color=c, lw=2)
        ax.text((b + d) / 2, (d - b) / 2 + 0.06, f"$f_{{{cols.index(c)+1}}}$", ha="center", color=c, fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.3); ax.set_xlabel("$\\epsilon$")
    ax.set_title("tents $f_j(\\epsilon)=\\max(0,\\min(\\epsilon-b_j,\\,d_j-\\epsilon))$")
    ax = axes[2]
    for k, (ls, lab) in enumerate([("-", "$\\lambda_1$ (largest)"), ("--", "$\\lambda_2$"), (":", "$\\lambda_3$")]):
        ax.plot(grid, lam[k], color=INK, lw=2, ls=ls, label=lab)
    ax.fill_between(grid, lam[0], color=BLUE, alpha=0.12)
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.3); ax.set_xlabel("$\\epsilon$"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("landscape: $\\lambda_k(\\epsilon)$ = $k$-th largest tent")
    for ax in axes[1:]:
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_landscape.png"); plt.close(fig)


def fig_stability():
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 2 * np.pi, 60)
    base = np.c_[np.cos(t), np.sin(t)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for k, s in enumerate([0.0, 0.08, 0.2]):
        P = base + rng.normal(0, s, base.shape)
        d = ripser(P, maxdim=1)["dgms"]
        ax = axes[0, k]; ax.scatter(P[:, 0], P[:, 1], s=14, color=INK); clean(ax)
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_title(f"noise sd = {s}")
        ax = axes[1, k]; m = 2.2
        ax.fill_between([0, m], [0, m], [0, 0], color="#F2F4F7"); ax.plot([0, m], [0, m], color=SUB, lw=1)
        f0 = d[0][np.isfinite(d[0][:, 1])]
        ax.scatter(f0[:, 0], f0[:, 1], s=16, color=SUB, label="$H_0$")
        ax.scatter(d[1][:, 0], d[1][:, 1], s=30, color=AMBER, label="$H_1$")
        ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_aspect("equal"); ax.set_xlabel("birth"); ax.set_ylabel("death")
        if k == 0:
            ax.legend(frameon=False, loc="upper left")
        long = d[1][np.argmax(d[1][:, 1] - d[1][:, 0])]
        ax.annotate("the circle's loop", xy=long, xytext=(0.9, 0.35), fontsize=9, color=AMBER,
                    arrowprops=dict(arrowstyle="->", color=AMBER))
    fig.suptitle("Stability: small movements of the points move the diagram only a little", y=1.0, fontsize=12, weight="bold")
    fig.savefig(OUT / "fig_stability.png"); plt.close(fig)


def fig_sliding_window():
    rng = np.random.default_rng(7)
    n = 160
    calm = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n) * 0.6
    stress = np.r_[rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n // 2) * 0.6,
                   rng.multivariate_normal([0, 0], [[1, -0.3], [-0.3, 1]], n // 2) * 1.6]
    series = np.r_[calm, stress]
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[0])
    ax.plot(series[:, 0], color=BLUE, lw=0.9, label="index A return")
    ax.plot(series[:, 1] - 6, color=AMBER, lw=0.9, label="index B return (shifted)")
    for a, b, c in [(40, 100, GREEN), (230, 290, RED)]:
        ax.axvspan(a, b, color=c, alpha=0.12)
    ax.text(70, 3.2, "window 1", ha="center", color=GREEN); ax.text(260, 3.2, "window 2", ha="center", color=RED)
    ax.set_yticks([]); ax.set_xlabel("trading day $t$"); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("1. two return series → 2. a window of $w$ days", loc="left")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    for k, (a, b, c, name) in enumerate([(40, 100, GREEN, "window 1: calm"), (230, 290, RED, "window 2: stressed")]):
        ax = fig.add_subplot(gs[k + 1]); P = series[a:b]
        d = ripser(P, maxdim=1)["dgms"][1]; d = d[np.isfinite(d[:, 1])]
        ax.scatter(P[:, 0], P[:, 1], s=12, color=c); ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
        ax.set_xlabel("$r_A$"); ax.set_ylabel("$r_B$")
        ax.set_title(f"3. point cloud $X_t\\subset\\mathbb{{R}}^2$\n{name}: $\\|\\lambda\\|_1$ = {landscape_norm(d, 1):.2f}", fontsize=10)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_sliding_window.png"); plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 2.2))
    steps = ["index prices\n$P_{k,t}$", "log returns\n$r_{k,t}$", "sliding window\n$X_t\\subset\\mathbb{R}^d$", "Vietoris–Rips\nfiltration",
             "$H_1$ persistence\ndiagram $D_t$", "landscape\n$\\|\\lambda\\|_1,\\|\\lambda\\|_2$", "rolling stats\n+ Kendall $\\tau$"]
    n = len(steps); w = 1.5; gap = 0.22
    for i, s in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=BLUE_SOFT if i in (3, 4, 5) else "#F2F4F7", edgecolor=HAIR))
        ax.text(x + w / 2, 0.5, s, ha="center", va="center", fontsize=10)
        if i < n - 1:
            arrow(ax, (x + w + 0.02, 0.5), (x + w + gap - 0.02, 0.5), SUB)
    ax.set_xlim(-0.1, n * (w + gap)); ax.set_ylim(-0.1, 1.1); clean(ax)
    fig.savefig(OUT / "fig_pipeline.png"); plt.close(fig)


if __name__ == "__main__":
    fig_simplices(); fig_faces_boundary(); fig_complex_chains(); fig_betti_shapes(); fig_induced_map()
    fig_pointcloud_balls(); fig_cech_vs_rips()
    h0, h1 = fig_filtration_barcode(); fig_diagram(h0, h1)
    fig_landscape(); fig_stability(); fig_sliding_window(); fig_pipeline()
    print("ok", sorted(p.name for p in OUT.glob("*.png")))
