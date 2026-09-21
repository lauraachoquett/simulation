"""Schema du concept de ligne de descendance (LOD).

    python -m simulation.tools.fig_lod_schema
    python -m simulation.tools.fig_lod_schema --graine 7 -o fig/lod_schema.png

Arbre genealogique fictif : la population du pas t, son MRCA, et la lignee qui
remonte du MRCA jusqu'au premier pas.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MORT, CLADE, LOD, VIVANT = "#C9C4BC", "#6B7A8F", "#C1121F", "#1D5C8F"


def arbre(rng, n_pas, p_naissance, p_mort, n_max):
    """(parents, naissance, fin) d'un arbre de naissances et de morts."""
    parent, naissance, fin = [-1], [0], [None]
    vivants = [0]
    for t in range(1, n_pas + 1):
        for i in list(vivants):
            if len(vivants) > 1 and rng.random() < p_mort:
                vivants.remove(i)
                fin[i] = t
            elif len(vivants) < n_max and rng.random() < p_naissance:
                parent.append(i), naissance.append(t), fin.append(None)
                vivants.append(len(parent) - 1)
    for i in vivants:
        fin[i] = n_pas
    return np.array(parent), np.array(naissance), np.array(fin), vivants


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


def dessine(parent, naissance, fin, vivants, ligne, clade, y, ax, n_pas):
    for i in range(len(parent)):
        couleur, lw, z = (MORT, 1.3, 1)
        if i in clade:
            couleur, lw, z = (CLADE, 1.8, 2)
        if i in ligne:
            couleur, lw, z = (LOD, 3.2, 4)
        ax.plot([naissance[i], fin[i]], [y[i], y[i]], color=couleur, lw=lw, zorder=z,
                solid_capstyle="round")
        p = parent[i]
        if p != -1:
            ax.plot([naissance[i], naissance[i]], [y[p], y[i]], color=couleur,
                    lw=lw * .8, zorder=z)
    for v in vivants:
        ax.scatter(n_pas, y[v], s=60, color=VIVANT, edgecolor="white",
                   linewidth=1.2, zorder=6)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--graine", type=int, default=3)
    p.add_argument("--pas", type=int, default=50, help="duree de l'arbre")
    p.add_argument("--n-max", dest="n_max", type=int, default=6,
                   help="population maximale de l'arbre fictif")
    p.add_argument("--naissance", type=float, default=.10)
    p.add_argument("--mort", type=float, default=.10)
    p.add_argument("-o", "--out", default="fig/lod_schema.png")
    a = p.parse_args()

    rng = np.random.default_rng(a.graine)
    for _ in range(200):     # un arbre dont le MRCA n'est ni la racine ni tout recent
        parent, naissance, fin, vivants = arbre(rng, a.pas, a.naissance,
                                                a.mort, a.n_max)
        if not 3 <= len(vivants) <= 6 or len(parent) > 26:
            continue
        m = mrca(vivants, parent)
        # MRCA mort avant t : sinon il apparait dans la population du pas t
        if 0 < naissance[m] < .7 * a.pas and fin[m] < .92 * a.pas:
            break

    ordre, y = [], np.zeros(len(parent))
    def descend(i):                      # feuilles rangees par parcours prefixe
        ordre.append(i)
        for j in np.flatnonzero(parent == i):
            descend(int(j))
    descend(0)
    for rang, i in enumerate(ordre):
        y[i] = rang

    ligne = set(ancetres(m, parent))
    clade = {i for i in range(len(parent)) if m in ancetres(i, parent)} - ligne

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    fig.patch.set_facecolor("white")
    dessine(parent, naissance, fin, vivants, ligne, clade, y, ax, a.pas)

    ax.axvline(a.pas, color="#2B2B2B", lw=1.2, ls=(0, (4, 3)), zorder=5)
    ax.scatter([naissance[m]], [y[m]], s=180, marker="o", facecolor="white",
               edgecolor=LOD, linewidth=2.6, zorder=7)
    ax.annotate("MRCA\nof the present population",
                xy=(naissance[m], y[m]), xytext=(naissance[m] - .22 * a.pas, y[m] + 2.4),
                fontsize=11, color=LOD, ha="center",
                arrowprops=dict(arrowstyle="->", color=LOD, lw=1.4))
    # la fleche vise un vrai segment de la lignee, pas le vide entre deux branches
    cible = sorted(ligne, key=lambda i: fin[i] - naissance[i])[-1]
    ax.annotate("line of descent",
                xy=((naissance[cible] + fin[cible]) / 2, y[cible]),
                xytext=((naissance[cible] + fin[cible]) / 2, max(y) * .55),
                fontsize=11, color=LOD, ha="center",
                arrowprops=dict(arrowstyle="->", color=LOD, lw=1.4))
    ax.text(a.pas, -.9, "present\nstep t", ha="center", va="top", fontsize=11)
    ax.text(0, y[0] - 1.5, "first step", ha="left", va="top", fontsize=11,
            color="#4A4A4A")

    ax.set_xlim(-.04 * a.pas, a.pas * 1.06)
    ax.set_ylim(-4.2, max(y) + 2.6)
    ax.set_xlabel("simulation step", fontsize=11)
    ax.set_yticks([])
    for c in ax.spines.values():
        c.set_visible(False)
    ax.set_xticks([])
    # fleche du temps : une ligne pleine barrerait les etiquettes du bas
    ax.annotate("", xy=(a.pas * 1.04, -3.6), xytext=(0, -3.6),
                arrowprops=dict(arrowstyle="->", color="#8A8A8A", lw=1.2))

    ax.legend(handles=[
        Line2D([0], [0], color=LOD, lw=3.2, label="line of descent: the ancestors evolution went through"),
        Line2D([0], [0], color=CLADE, lw=1.8, label="descendants of the MRCA"),
        Line2D([0], [0], color=MORT, lw=1.3, label="lineages that died out"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VIVANT,
               markeredgecolor="white", markersize=9, label="individuals alive at step t"),
    ], loc="upper center", bbox_to_anchor=(.5, -.04), ncol=2, frameon=False, fontsize=10)

    ax.set_title("Line of descent: tracing the present population back to its origin",
                 fontsize=14, fontweight="semibold", pad=14)
    fig.tight_layout(rect=[0, .09, 1, 1])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"Figure saved: {a.out}")


if __name__ == "__main__":
    main()
