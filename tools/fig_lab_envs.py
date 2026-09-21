"""Figure des environnements de lab figes (lab_envs/*.npy).

    python -m simulation.tools.fig_lab_envs
    python -m simulation.tools.fig_lab_envs scatter40_s0 patch8x5_s0 -o fig/envs.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from simulation.data_class import LAB_ENVS, color_of

TITRES = {"scatter40_s0": "Scattered", "patch8x5_s0": "Patchy",
          "blob1x40_s1": "Single blob"}
SOUS_TITRES = {"scatter40_s0": "40 isolated items",
               "patch8x5_s0": "8 patches of 5",
               "blob1x40_s1": "one blob of 40"}
MUR, FOND, GRILLE = "#2B2B2B", "#FAF7F2", "#E2DCD3"


def charge(nom, dossier):
    g = np.load(os.path.join(dossier, nom if nom.endswith(".npy") else nom + ".npy"))
    return g.sum(axis=0) if g.ndim == 3 else g


def panneau(ax, grille, couleur, titre, sous_titre):
    L = grille.shape[0]
    ax.set_facecolor(FOND)
    ax.set_xlim(-.5, L - .5), ax.set_ylim(L - .5, -.5)
    ax.set_aspect("equal")
    for k in range(L + 1):
        ax.axhline(k - .5, color=GRILLE, lw=.35, zorder=1)
        ax.axvline(k - .5, color=GRILLE, lw=.35, zorder=1)

    # bande d'apparition des agents : randint(10, L - 10) dans init_state_lab
    ax.add_patch(Rectangle((9.5, 9.5), L - 20, L - 20, facecolor="#00000008",
                           edgecolor="#9A9A9A", lw=.9, ls=(0, (4, 3)), zorder=2))
    # mur letal : l'anneau exterieur d'une case
    for x, y, w, h in ((-.5, -.5, L, 1), (-.5, L - 1.5, L, 1),
                       (-.5, .5, 1, L - 2), (L - 1.5, .5, 1, L - 2)):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=MUR, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((-.5, -.5), L, L, facecolor="none", edgecolor=MUR,
                           lw=1.6, zorder=5))

    y, x = np.nonzero(grille)
    ax.scatter(x, y, s=46, marker="s", color=couleur, edgecolor="white",
               linewidth=.5, zorder=4)
    ax.set_xticks([]), ax.set_yticks([])
    for c in ax.spines.values():
        c.set_visible(False)
    ax.set_title(titre, fontsize=13, pad=30, fontweight="semibold")
    ax.text(.5, 1.012, sous_titre, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10, color="#5A5A5A")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("noms", nargs="*", default=list(LAB_ENVS))
    p.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "lab_envs"))
    p.add_argument("-o", "--out", default="fig/lab_envs.png")
    a = p.parse_args()

    grilles = [(n, charge(n, a.dir)) for n in a.noms]
    couleur = color_of(0)
    fig, axes = plt.subplots(1, len(grilles), figsize=(4.5 * len(grilles), 5.1))
    fig.patch.set_facecolor("white")
    for ax, (nom, g) in zip(np.atleast_1d(axes), grilles):
        cle = os.path.splitext(os.path.basename(nom))[0]
        panneau(ax, g, couleur, TITRES.get(cle, cle), SOUS_TITRES.get(cle, ""))

    fig.suptitle("Fixed test environments", fontsize=16, fontweight="semibold", y=.99)
    fig.text(.5, .085, "30 × 30 arena, border wall in dark, agents start inside "
             "the dashed area. 40 resource cells in all three.",
             ha="center", fontsize=10, color="#4A4A4A")
    fig.tight_layout(rect=[0, .11, 1, .95])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"Figure saved: {a.out}")


if __name__ == "__main__":
    main()
