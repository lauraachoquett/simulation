"""Ou naissent les descendants : a cote du parent, ou n'importe ou.

    python -m simulation.tools.fig_viscosite
    python -m simulation.tools.fig_viscosite --no-titre -o fig/viscosite.pdf

Schema : rien n'est lu, les positions sont tirees pour l'illustration.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from simulation.data_class import color_of
from simulation.tools.fig_lab_envs import FOND, GRILLE, MUR

PARENT, ENFANT = "#6E0B12", "#E36C6C"


def grille(ax, cote, titre):
    ax.set_facecolor(FOND)
    ax.set_xlim(-.5, cote - .5), ax.set_ylim(cote - .5, -.5)
    ax.set_aspect("equal")
    for k in range(cote + 1):
        ax.axhline(k - .5, color=GRILLE, lw=.35, zorder=1)
        ax.axvline(k - .5, color=GRILLE, lw=.35, zorder=1)
    for x, y, w, h in ((-.5, -.5, cote, 1), (-.5, cote - 1.5, cote, 1),
                       (-.5, .5, 1, cote - 2), (cote - 1.5, .5, 1, cote - 2)):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=MUR, edgecolor="none",
                               zorder=3))
    ax.set_xticks([]), ax.set_yticks([])
    for c in ax.spines.values():
        c.set_visible(False)
    ax.set_title(titre, fontsize=12, pad=10, fontweight="semibold")


def panneau(ax, cote, parent, enfants, ressources, titre):
    grille(ax, cote, titre)
    for (y, x) in ressources:
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, color=color_of(0),
                               zorder=2))
    for (y, x) in enfants:
        ax.add_patch(FancyArrowPatch((parent[1], parent[0]), (x, y),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=PARENT, lw=1.1, alpha=.55,
                                     linestyle=(0, (3, 2)),
                                     shrinkA=7, shrinkB=7, zorder=4))
        ax.scatter(x, y, s=95, color=ENFANT, edgecolor="white", linewidth=1.2,
                   zorder=6)
    ax.scatter(parent[1], parent[0], s=180, color=PARENT, edgecolor="white",
               linewidth=1.4, zorder=7)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cote", type=int, default=22, help="cote de la grille")
    p.add_argument("--n-enfants", dest="n_enfants", type=int, default=4)
    p.add_argument("--n-res", dest="n_res", type=int, default=18)
    p.add_argument("--graine", type=int, default=2)
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--police", type=float, default=12)
    p.add_argument("-o", "--out", default="fig/viscosite.png")
    a = p.parse_args()

    plt.rcParams.update({"font.size": a.police})
    rng = np.random.default_rng(a.graine)
    c = a.cote
    parent = (c // 2, c // 2 - 2)
    res = rng.integers(1, c - 1, size=(a.n_res, 2))

    # a cote du parent : les cases voisines, sans doublon
    autour = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy, dx) != (0, 0)]
    choix = rng.permutation(len(autour))[:a.n_enfants]
    proches = [(parent[0] + autour[k][0], parent[1] + autour[k][1]) for k in choix]
    # n'importe ou : tirage uniforme, loin du parent pour que ce soit lisible
    loin = []
    while len(loin) < a.n_enfants:
        y, x = rng.integers(2, c - 2, 2)
        if abs(y - parent[0]) + abs(x - parent[1]) > c // 2:
            loin.append((int(y), int(x)))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.9))
    fig.patch.set_facecolor("white")
    panneau(axes[0], c, parent, proches, res,
            "Offspring born at the parent's position")
    panneau(axes[1], c, parent, loin, res,
            "Offspring born at a random position")
    for ax in axes:
        ax.set_anchor("N")

    leg = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PARENT,
                      markeredgecolor="white", markersize=11, label="parent"),
           plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ENFANT,
                      markeredgecolor="white", markersize=9, label="offspring"),
           plt.Line2D([0], [0], marker="s", color="none",
                      markerfacecolor=color_of(0), markeredgecolor="white",
                      markersize=9, label="resource")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.01))
    if not a.no_titre:
        fig.suptitle("Where offspring appear", fontsize=a.police + 2, y=.99)
    fig.tight_layout(rect=[0, .07, 1, .94 if not a.no_titre else 1])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Figure saved: {a.out}")


if __name__ == "__main__":
    main()
