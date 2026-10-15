"""Mouvement moyen de la population sur une plage de pas, moyenne glissante.

    python -m simulation.tools.plot_mouvement <exp_dir> --jusqu 100000
    python -m simulation.tools.plot_mouvement <exp_dir> --depuis 3000000 \\
        --jusqu 3100000 --lissage 10000

Lit data/chunk_*.npz : rien n'est rejoue. Une seule couleur, pas de colormap --
la couleur par taille de population melange le signal et le bruit d'echantillon
(la moyenne porte sur N agents, son ecart-type vaut ~0.44/sqrt(N)).

Les pas sont ABSOLUS, reconstruits depuis le numero de chunk : une plage prise
au milieu d'un run reste a sa place sur l'axe.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_NUM = re.compile(r"chunk_(\d+)\.npz")
MOV = "#1D5C8F"


def series(source, depuis, jusqu, chunk_size):
    """(pas, mouvement) sur la plage demandee, pas absolus."""
    pas, mov = [], []
    for f in sorted(glob.glob(os.path.join(source, "data", "chunk_*.npz")),
                    key=lambda f: int(_NUM.search(os.path.basename(f)).group(1))):
        c = int(_NUM.search(os.path.basename(f)).group(1))
        debut = (c - 1) * chunk_size
        if jusqu and debut >= jusqu:
            break
        if debut + chunk_size < depuis:
            continue
        with np.load(f) as z:
            if "mean_movement" not in z.files:
                continue
            m = np.asarray(z["mean_movement"], float)
        pas.append(debut + np.arange(len(m))), mov.append(m)
    if not pas:
        raise SystemExit(f"aucun chunk dans [{depuis}, {jusqu or 'fin'}] "
                         f"sous {source}")
    pas, mov = np.concatenate(pas), np.concatenate(mov)
    garde = (pas >= depuis) & ((pas < jusqu) if jusqu else True)
    return pas[garde], mov[garde]


def glissante(y, k):
    """Moyenne glissante sur k points, bords ecartes (pas de fenetre partielle)."""
    if k < 2 or k > len(y):
        return np.arange(len(y)), y
    noyau = np.ones(k) / k
    d = k // 2
    return np.arange(d, len(y) - k + 1 + d), np.convolve(y, noyau, mode="valid")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier d'experience portant data/")
    p.add_argument("--depuis", type=int, default=0, help="premier pas trace")
    p.add_argument("--jusqu", type=int, default=0, help="dernier pas (0 = fin)")
    p.add_argument("--lissage", type=int, default=10000, metavar="K",
                   help="moyenne glissante, en pas (defaut %(default)s)")
    p.add_argument("--brut", action=argparse.BooleanOptionalAction, default=True,
                   help="tracer aussi la serie brute en clair (defaut : oui)")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=1000)
    p.add_argument("--unite", default=None, choices=["pas", "k", "M"],
                   help="unite de l'axe des pas (defaut : selon la plage)")
    p.add_argument("--couleur", default=MOV)
    p.add_argument("--police", type=float, default=13)
    p.add_argument("--taille", type=float, nargs=2, default=(11., 4.2))
    p.add_argument("--ylim", type=float, nargs=2, default=None)
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    plt.rcParams.update({"font.size": a.police, "axes.labelsize": a.police,
                         "axes.titlesize": a.police + 1})
    pas, mov = series(a.source, a.depuis, a.jusqu, a.chunk_size)
    etendue = pas[-1] - pas[0]
    unite = a.unite or ("pas" if etendue <= 2000 else
                        "k" if etendue <= 500_000 else "M")
    div, nom_u = {"pas": (1, "Simulation step"),
                  "k": (1e3, "Simulation step (x10$^3$)"),
                  "M": (1e6, "Simulation step (x10$^6$)")}[unite]

    fig, ax = plt.subplots(figsize=tuple(a.taille))
    if a.brut:
        ax.plot(pas / div, mov, color=a.couleur, lw=.6, alpha=.25)
    idx, lisse = glissante(mov, a.lissage)
    ax.plot(pas[idx] / div, lisse, color=a.couleur, lw=2.2)
    ax.set_xlabel(nom_u)
    ax.set_ylabel("Mean movement (cell / step)")
    ax.set_xlim(pas[0] / div, pas[-1] / div)
    if a.ylim:
        ax.set_ylim(*a.ylim)
    ax.grid(alpha=.3)
    if not a.no_titre:
        ax.set_title(f"Mean population movement, {a.lissage}-step rolling mean")

    nom = f"plot_mouvement_{int(pas[0])}_{int(pas[-1]) + 1}.{a.fig_format}"
    out = a.out or os.path.join(a.source, "fig", nom)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{len(pas)} pas, de {pas[0]} a {pas[-1]} | mouvement lisse : "
          f"{lisse.min():.3f} a {lisse.max():.3f} (brut {mov.min():.3f}-{mov.max():.3f})")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
