"""Schema du concept de ligne de descendance (LOD) : un arbre genealogique.

    python -m simulation.tools.fig_lod_schema
    python -m simulation.tools.fig_lod_schema --graine 7 -o fig/lod_schema.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MORT, CLADE, LOD, VIVANT = "#C9C4BC", "#6B7A8F", "#C1121F", "#1D5C8F"


def arbre(rng, n_gen, p_deux, p_zero):
    """(parent, generation, feuilles) d'un arbre a generations discretes."""
    parent, gen = [-1], [0]
    front = [0]
    for g in range(1, n_gen + 1):
        suivant = []
        for i in front:
            u = rng.random()
            n = 0 if u < p_zero else (2 if u > 1 - p_deux else 1)
            for _ in range(n):
                parent.append(i), gen.append(g)
                suivant.append(len(parent) - 1)
        front = suivant
        if not front:
            break
    return np.array(parent), np.array(gen), front


def ancetres(i, parent):
    chaine = []
    while i != -1:
        chaine.append(i)
        i = parent[i]
    return chaine[::-1]


def mrca(vivants, parent):
    communs = set(ancetres(vivants[0], parent))
    for v in vivants[1:]:
        communs &= set(ancetres(v, parent))
    return max(communs, key=lambda i: len(ancetres(i, parent)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--generations", type=int, default=9)
    p.add_argument("-o", "--out", default="fig/lod_schema.png")
    a = p.parse_args()

    rng = np.random.default_rng(a.graine)
    for _ in range(500):      # arbre lisible : peu de feuilles, MRCA au milieu
        parent, gen, vivants = arbre(rng, a.generations, .30, .28)
        if not 3 <= len(vivants) <= 5 or len(parent) > 30:
            continue
        m = mrca(vivants, parent)
        if .25 * a.generations < gen[m] < .65 * a.generations:
            break

    ordre = []
    def descend(i):
        enfants = np.flatnonzero(parent == i)
        if len(enfants) == 0:
            ordre.append(i)
        for j in enfants:
            descend(int(j))
    descend(0)
    y = np.zeros(len(parent))
    for rang, i in enumerate(ordre):
        y[i] = rang
    for i in range(len(parent) - 1, -1, -1):      # parent centre sur ses enfants
        enfants = np.flatnonzero(parent == i)
        if len(enfants):
            y[i] = y[enfants].mean()

    ligne = set(ancetres(m, parent))
    clade = {i for i in range(len(parent)) if m in ancetres(i, parent)} - ligne

    def style(i):
        if i in ligne:
            return LOD, 2.8, 5
        if i in clade:
            return CLADE, 1.6, 3
        return MORT, 1.2, 1

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor("white")
    for i in range(1, len(parent)):
        couleur, lw, z = style(i)
        pa = parent[i]
        ax.plot([gen[pa], gen[i]], [y[pa], y[i]], color=couleur, lw=lw, zorder=z,
                solid_capstyle="round")
    for i in range(len(parent)):
        couleur, _, z = style(i)
        vif = i in ligne or i in clade
        ax.scatter(gen[i], y[i], s=115 if vif else 70, color=couleur,
                   edgecolor="white", linewidth=1.4, zorder=z + 3)
    for v in vivants:
        ax.scatter(gen[v], y[v], s=150, color=VIVANT, edgecolor="white",
                   linewidth=1.5, zorder=9)

    ax.scatter([gen[m]], [y[m]], s=330, marker="o", facecolor="none",
               edgecolor=LOD, linewidth=2.4, zorder=10)
    ax.annotate("MRCA of the present population", xy=(gen[m], y[m] + .22),
                xytext=(gen[m] - .2, max(y) + 1.2), fontsize=11.5, color=LOD,
                ha="center", arrowprops=dict(arrowstyle="->", color=LOD, lw=1.4))
    mi = min(ligne, key=lambda i: gen[i] if i != 0 else 99)
    ax.annotate("line of descent", xy=(gen[mi], y[mi] - .18),
                xytext=(gen[mi] + .1, -1.5), fontsize=11.5, color=LOD,
                ha="center", arrowprops=dict(arrowstyle="->", color=LOD, lw=1.4))
    ax.text(0, y[0] + .7, "founder", ha="center", fontsize=11, color="#4A4A4A")
    ax.text(max(gen), max(y) + .7, "present population", ha="center",
            fontsize=11, color=VIVANT)

    ax.set_xlim(-.9, max(gen) + .9)
    ax.set_ylim(-2.6, max(y) + 1.9)
    ax.set_xticks([]), ax.set_yticks([])
    for c in ax.spines.values():
        c.set_visible(False)
    ax.annotate("", xy=(max(gen) + .7, -2.25), xytext=(-.6, -2.25),
                arrowprops=dict(arrowstyle="->", color="#8A8A8A", lw=1.2))
    ax.text(max(gen) / 2, -2.45, "generations", ha="center", va="top",
            fontsize=11, color="#4A4A4A")

    ax.legend(handles=[
        Line2D([0], [0], color=LOD, lw=2.8, marker="o", markersize=8,
               label="line of descent"),
        Line2D([0], [0], color=CLADE, lw=1.6, marker="o", markersize=7,
               label="other descendants of the MRCA"),
        Line2D([0], [0], color=MORT, lw=1.2, marker="o", markersize=7,
               label="extinct lineages"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=VIVANT,
               markeredgecolor="white", markersize=9, label="alive today"),
    ], loc="upper center", bbox_to_anchor=(.5, -.02), ncol=4, frameon=False,
        fontsize=10.5)
    ax.set_title("Genealogy of a population: the line of descent",
                 fontsize=14, fontweight="semibold", pad=12)
    fig.tight_layout(rect=[0, .07, 1, 1])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"Figure saved: {a.out}")


if __name__ == "__main__":
    main()
