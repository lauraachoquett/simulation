"""Schema des trois conditions sociales, sur la grille scattered.

    python -m simulation.tools.fig_conditions
    python -m simulation.tools.fig_conditions --env patch8x5_s0 -o fig/cond.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from simulation.data_class import color_of
from simulation.tools.fig_lab_envs import charge, panneau

FOCAL, PAIR, INERTE = "#C1121F", "#E36C6C", "#6B7A8F"
POS_FOCAL = (15, 14)
POS_AUTRES = [(11, 18), (19, 11), (17, 21), (12, 11)]

PANNEAUX = [
    ("Alone", "one tested agent", 0, None),
    ("Identical clones", "4 peers, same genome, they eat", 4, PAIR),
    ("Inert peers", "3 peers, random moves, no eating", 3, INERTE),
]


def agents(ax, n, couleur):
    ax.scatter(*POS_FOCAL[::-1], s=210, marker="o", color=FOCAL,
               edgecolor="white", linewidth=1.6, zorder=6)
    for (y, x) in POS_AUTRES[:n]:
        ax.scatter(x, y, s=190, marker="o", color=couleur, edgecolor="white",
                   linewidth=1.4, zorder=5)
    if n and couleur == INERTE:      # politique aleatoire : fleches brouillonnes
        for (y, x) in POS_AUTRES[:n]:
            for dy, dx in ((1.6, 0), (0, 1.6), (-1.6, 0)):
                ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color=INERTE,
                                            lw=1.0, alpha=.65), zorder=4)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env", default="scatter40_s0")
    p.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "lab_envs"))
    p.add_argument("-o", "--out", default="fig/lab_conditions.png")
    a = p.parse_args()

    grille = charge(a.env, a.dir)
    fig, axes = plt.subplots(1, len(PANNEAUX), figsize=(4.5 * len(PANNEAUX), 5.3))
    fig.patch.set_facecolor("white")
    for ax, (titre, sous, n, couleur) in zip(axes, PANNEAUX):
        panneau(ax, grille, color_of(0), titre, sous)
        agents(ax, n, couleur)

    legende = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FOCAL,
               markeredgecolor="white", markersize=12, label="tested agent"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAIR,
               markeredgecolor="white", markersize=12,
               label="clone: same genome, consumes resources"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INERTE,
               markeredgecolor="white", markersize=12,
               label="inert peer: random policy, no consumption, frozen energy"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color_of(0),
               markeredgecolor="white", markersize=11, label="resource"),
    ]
    fig.legend(handles=legende, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(.5, -.01))
    fig.suptitle("Social conditions in the test environment",
                 fontsize=16, fontweight="semibold", y=.995)
    fig.tight_layout(rect=[0, .12, 1, .90])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"Figure saved: {a.out}")


if __name__ == "__main__":
    main()
