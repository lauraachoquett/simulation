"""Profil moyen autour des krachs de population : que font les rescapes ?

    python -m simulation.tools.plot_krach <exp_dir> --jusqu 100000
    python -m simulation.tools.plot_krach <exp_dir> --fenetre 2000 --fig-format pdf

Lit data/chunk_*.npz : rien n'est rejoue. Tous les creux de population sont
alignes sur leur minimum puis moyennes. C'est le seul moyen de lire quoi que ce
soit au creux : la moyenne de population y porte sur quelques agents, donc son
ecart-type vaut ~0.44/sqrt(N) -- a 20 agents, 0.1, soit l'amplitude du signal
cherche. Empiler vingt krachs divise ce bruit par sqrt(20).
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
POP, RES, MOV = "#C1121F", "#2A9131", "#1D5C8F"


def series(source, jusqu, chunk_size):
    """(population, ressources, mouvement) par pas, recolles sur les chunks."""
    fs = sorted(glob.glob(os.path.join(source, "data", "chunk_*.npz")),
                key=lambda f: int(_NUM.search(os.path.basename(f)).group(1)))
    P, R, M = [], [], []
    for f in fs:
        c = int(_NUM.search(os.path.basename(f)).group(1))
        if jusqu and (c - 1) * chunk_size >= jusqu:
            break
        with np.load(f) as z:
            if "mean_movement" not in z.files:
                continue
            p = np.asarray(z["population"], float)
            m = np.asarray(z["mean_movement"], float)
            r = np.asarray(z.get("resources", np.zeros_like(p)), float)
            if r.ndim > 1:
                r = r.sum(axis=1)
            # mean_movement est une quantite de transition : un point de moins
            n = min(len(p), len(m), len(r))
        P.append(p[:n]), R.append(r[:n]), M.append(m[:n])
    if not P:
        raise SystemExit(f"pas de data/chunk_*.npz exploitable sous {source}")
    P, R, M = np.concatenate(P), np.concatenate(R), np.concatenate(M)
    return (P[:jusqu], R[:jusqu], M[:jusqu]) if jusqu else (P, R, M)


def creux(P, largeur):
    """Minimums locaux de population, espaces d'au moins `largeur` pas."""
    out = []
    for t in range(largeur, len(P) - largeur):
        if P[t] == P[t - largeur:t + largeur].min() and (
                not out or t - out[-1] > largeur):
            out.append(t)
    return out


def profil(v, pics, L):
    """(n_krachs, 2L) : la serie decoupee autour de chaque creux."""
    return np.array([v[t - L:t + L] for t in pics])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier d'experience portant data/")
    p.add_argument("--jusqu", type=int, default=0,
                   help="ne garder que les N premiers pas (defaut 0 = tout)")
    p.add_argument("--fenetre", type=int, default=1500, metavar="L",
                   help="demi-fenetre tracee autour du creux (defaut %(default)s)")
    p.add_argument("--largeur-creux", dest="largeur", type=int, default=600,
                   metavar="W", help="deux creux plus proches que W pas n'en "
                                     "font qu'un (defaut %(default)s)")
    p.add_argument("--lissage", type=int, default=51, metavar="K",
                   help="moyenne glissante sur K pas pour la courbe epaisse "
                        "(defaut %(default)s, 1 = brut seulement)")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=1000)
    p.add_argument("--police", type=float, default=13)
    p.add_argument("--taille", type=float, nargs=2, default=(9.5, 6.4))
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    plt.rcParams.update({"font.size": a.police, "axes.labelsize": a.police,
                         "axes.titlesize": a.police + 1,
                         "legend.fontsize": a.police - 1})
    P, R, M = series(a.source, a.jusqu, a.chunk_size)
    L = a.fenetre
    pics = [t for t in creux(P, a.largeur) if L <= t < len(P) - L]
    if len(pics) < 3:
        raise SystemExit(f"{len(pics)} creux seulement : rien a moyenner")
    pm, pp, pr = profil(M, pics, L), profil(P, pics, L), profil(R, pics, L)
    x = np.arange(-L, L)
    err = lambda v: np.nanstd(v, axis=0) / np.sqrt(len(v))

    fig, (h, b) = plt.subplots(2, 1, figsize=tuple(a.taille), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.25],
                                            "hspace": .12})
    for ax in (h, b):
        ax.axvline(0, color="#4A4A4A", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.grid(alpha=.25)

    h.plot(x, np.nanmean(pp, axis=0), color=POP, lw=2, label="population")
    h.set_ylabel("Population size", color=POP)
    h.tick_params(axis="y", labelcolor=POP)
    hr = h.twinx()
    hr.plot(x, np.nanmean(pr, axis=0), color=RES, lw=2, label="resources")
    hr.set_ylabel("Resources amount", color=RES)
    hr.tick_params(axis="y", labelcolor=RES)

    mm, em = np.nanmean(pm, axis=0), err(pm)
    b.plot(x, mm, color=MOV, lw=.8, alpha=.3)
    b.fill_between(x, mm - em, mm + em, color=MOV, alpha=.18, lw=0)
    if a.lissage > 1:
        # 'valid' et non 'same' : le bord d'une convolution 'same' divise par
        # une fenetre pleine un nombre de points partiel, et la courbe plonge
        k = np.ones(a.lissage) / a.lissage
        d = a.lissage // 2
        b.plot(x[d:len(x) - d + (1 - a.lissage % 2)],
               np.convolve(mm, k, mode="valid"), color=MOV, lw=2.2)
    b.set_ylabel("Mean movement (cell / step)", color=MOV)
    b.tick_params(axis="y", labelcolor=MOV)
    b.set_xlabel("Steps relative to the population minimum")
    b.set_xlim(-L, L)

    if not a.no_titre:
        h.set_title(f"Population crashes aligned on their minimum "
                    f"({len(pics)} crashes)")
    out = a.out or os.path.join(a.source, "fig", f"krach.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    avant = slice(L - a.largeur - L // 4, L - a.largeur)
    apres = slice(L + a.largeur, L + a.largeur + L // 4)
    print(f"{len(pics)} krachs, population au creux {np.median(pp[:, L]):.0f} "
          f"(de {pp[:, L].min():.0f} a {pp[:, L].max():.0f})")
    print(f"  mouvement avant {np.nanmean(pm[:, avant]):.3f}, "
          f"apres {np.nanmean(pm[:, apres]):.3f}, "
          f"ecart {np.nanmean(pm[:, apres]) - np.nanmean(pm[:, avant]):+.3f}")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
