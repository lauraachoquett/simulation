    
from collections import defaultdict


def r0_by_birth_window(node_children, oldest_alive_birth, window):
    num = defaultdict(int)   # somme des descendants
    den = defaultdict(int)   # taille de cohorte
    for (idx, born), children in node_children.items():
        wk = born // window
        if (wk + 1) * window <= oldest_alive_birth:   
            num[wk] += len(children)
            den[wk] += 1
    return {wk: num[wk] / den[wk] for wk in den}       # {index_fenêtre: R_0}

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot_r0(r0_by_window, window, exp_dir=None, smooth_w=0, fname="r0_evolution.png"):
    if not r0_by_window:
        return None

    ks = sorted(r0_by_window)
    # axe x = pas de temps de simulation (centre de chaque fenetre de naissance)
    x = np.array([(k + 0.5) * window for k in ks], dtype=float)
    y = np.array([r0_by_window[k] for k in ks], dtype=float)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    ymax = max(1.05 * y.max(), 1.1)
    ymin = min(0.95 * y.min(), 0.9)

    # regions sur- / sous-critiques (Galton-Watson)
    ax.axhspan(1.0, ymax, facecolor="#e8f2ec", zorder=0)
    ax.axhspan(ymin, 1.0, facecolor="#fdeceb", zorder=0)
    ax.axhline(1.0, ls="--", lw=1.2, color="#444444", zorder=2)

    # serie R_0
    ax.plot(x, y, color="#1f3b73", lw=1.2, alpha=0.55, zorder=3)
    colors = np.where(y >= 1.0, "#2e7d4f", "#c0392b")
    ax.scatter(x, y, c=colors, s=24, zorder=4,
               edgecolor="white", linewidth=0.5)

    # moyenne glissante optionnelle
    if smooth_w and len(y) >= smooth_w:
        ker = np.ones(smooth_w) / smooth_w
        ys = np.convolve(y, ker, mode="valid")
        off = smooth_w // 2
        ax.plot(x[off:len(x) - off], ys, color="#1f3b73", lw=2.2,
                zorder=5, label=f"rolling mean (w={smooth_w})")
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    # threshold annotation
    ax.text(x.min(), 1.0, " replacement threshold  $R_0 = 1$",
            va="bottom", ha="left", fontsize=8.5, color="#333333")
    ax.text(x.max(), ymax, "supercritical ", va="top", ha="right",
            fontsize=8.5, color="#2e7d4f", alpha=0.9)
    ax.text(x.max(), ymin, "subcritical ", va="bottom", ha="right",
            fontsize=8.5, color="#c0392b", alpha=0.9)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel(r"$R_0$  —  mean offspring per individual")
    ax.set_title(r"Evolution of the net reproduction rate $R_0$",
                 fontsize=12.5, pad=10)

    # steps affiches tels quels (entiers, pas de notation scientifique)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):d}"))
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)
        fig.savefig(os.path.join(exp_dir, fname), dpi=200, bbox_inches="tight")
    return fig