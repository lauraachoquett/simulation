"""Cycle moyen de la simulation : que fait le mouvement au fil d'un cycle ?

    python -m simulation.tools.plot_krach <exp_dir> --aligne pop --jusqu 100000
    python -m simulation.tools.plot_krach <exp_dir> --aligne res --fig-format pdf
    python -m simulation.tools.plot_krach <exp_dir> --aligne creux

Lit data/chunk_*.npz : rien n'est rejoue. Tous les cycles sont superposes puis
moyennes. C'est le seul moyen de lire quoi que ce soit au creux : la moyenne de
population y porte sur quelques agents, donc son ecart-type vaut ~0.44/sqrt(N)
-- a 20 agents, 0.1, soit l'amplitude du signal cherche. Empiler vingt cycles
divise ce bruit par sqrt(20).

Trois alignements :
  --aligne pop   : d'un maximum de population au suivant, axe en phase
  --aligne res   : d'un maximum de ressources au suivant, axe en phase
  --aligne creux : fenetre en pas autour du minimum de population
Les deux premiers reechantillonnent chaque cycle sur une phase 0-1, les cycles
n'ayant pas tous la meme duree ; le troisieme garde l'axe en pas, ce qui est ce
qu'il faut pour dater un decrochage par rapport au krach.
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
# ordre du modele : les quatre sorties du reseau, cf. compute_action_fractions_chunk
ACTIONS = ("stay", "rotate left", "rotate right", "move forward")
COUL_ACT = ("#9A9A9A", "#BDD7E7", "#6BAED6", MOV)


def series(source, jusqu, chunk_size):
    """(population, ressources, mouvement, actions) par pas, recolles.

    `actions` vaut None si les chunks ne portent pas action_frac (runs
    anterieurs a cette mesure).
    """
    fs = sorted(glob.glob(os.path.join(source, "data", "chunk_*.npz")),
                key=lambda f: int(_NUM.search(os.path.basename(f)).group(1)))
    P, R, M, A = [], [], [], []
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
            act = (np.asarray(z["action_frac"], float)
                   if "action_frac" in z.files else None)
            # mean_movement est une quantite de transition : un point de moins
            n = min([len(p), len(m), len(r)] + ([len(act)] if act is not None else []))
        P.append(p[:n]), R.append(r[:n]), M.append(m[:n])
        A.append(act[:n] if act is not None else None)
    if not P:
        raise SystemExit(f"pas de data/chunk_*.npz exploitable sous {source}")
    P, R, M = np.concatenate(P), np.concatenate(R), np.concatenate(M)
    A = np.concatenate(A) if all(x is not None for x in A) else None
    if jusqu:
        P, R, M = P[:jusqu], R[:jusqu], M[:jusqu]
        A = A[:jusqu] if A is not None else None
    return P, R, M, A


def extremums(v, largeur, haut=True):
    """Maximums (ou minimums) locaux, espaces d'au moins `largeur` pas.

    Detection sur la serie lissee : brute, une population de 20 agents fait
    des dents de scie qui passent toutes pour des extremums.
    """
    k = max(int(largeur) // 4 * 2 + 1, 3)
    lisse = np.convolve(v, np.ones(k) / k, mode="same")
    lisse[:k], lisse[-k:] = np.nan, np.nan          # bords de la convolution
    out = []
    for t in range(largeur, len(v) - largeur):
        f = lisse[t - largeur:t + largeur]
        if not np.isfinite(lisse[t]) or not np.isfinite(f).all():
            continue
        if lisse[t] == (f.max() if haut else f.min()) and (
                not out or t - out[-1] > largeur):
            out.append(t)
    return out


def cycles(bornes, n_phase):
    """[(debut, fin)] d'un extremum au suivant, les durees aberrantes ecartees."""
    paires = list(zip(bornes[:-1], bornes[1:]))
    if not paires:
        return []
    duree = np.array([b - a for a, b in paires], float)
    med = np.median(duree)
    garde = [p for p, d in zip(paires, duree) if .5 * med <= d <= 2 * med
             and d >= n_phase // 4]
    return garde


def en_phase(v, paires, n_phase):
    """(n_cycles, n_phase) : chaque cycle reechantillonne sur une phase 0-1."""
    phase = np.linspace(0, 1, n_phase)
    return np.array([np.interp(phase, np.linspace(0, 1, f - d), v[d:f])
                     for d, f in paires])


def profil(v, pics, L):
    """(n_krachs, 2L) : la serie decoupee autour de chaque creux."""
    return np.array([v[t - L:t + L] for t in pics])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier d'experience portant data/")
    p.add_argument("--aligne", default="creux", choices=["pop", "res", "creux"],
                   help="pop/res : d'un maximum au suivant, axe en phase. "
                        "creux : fenetre en pas autour du minimum de population "
                        "(defaut %(default)s)")
    p.add_argument("--jusqu", type=int, default=0,
                   help="ne garder que les N premiers pas (defaut 0 = tout)")
    p.add_argument("--fenetre", type=int, default=1500, metavar="L",
                   help="--aligne creux : demi-fenetre en pas (defaut %(default)s)")
    p.add_argument("--largeur-creux", dest="largeur", type=int, default=600,
                   metavar="W", help="deux extremums plus proches que W pas n'en "
                                     "font qu'un (defaut %(default)s)")
    p.add_argument("--n-phase", dest="n_phase", type=int, default=400,
                   help="points de l'axe de phase (defaut %(default)s)")
    p.add_argument("--lissage", type=int, default=51, metavar="K",
                   help="moyenne glissante pour la courbe epaisse "
                        "(defaut %(default)s, 1 = brut seulement)")
    p.add_argument("--no-actions", dest="no_actions", action="store_true",
                   help="ne pas tracer le panneau des fractions d'action")
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
    P, R, M, A = series(a.source, a.jusqu, a.chunk_size)
    if a.no_actions:
        A = None

    if a.aligne == "creux":
        L = a.fenetre
        pics = [t for t in extremums(P, a.largeur, haut=False)
                if L <= t < len(P) - L]
        if len(pics) < 3:
            raise SystemExit(f"{len(pics)} creux seulement : rien a moyenner")
        pp, pr, pm = (profil(v, pics, L) for v in (P, R, M))
        pa = ([profil(A[:, k], pics, L) for k in range(A.shape[1])]
              if A is not None else None)
        x = np.arange(-L, L).astype(float)
        n, xlabel = len(pics), "Steps relative to the population minimum"
        titre = f"Population crashes aligned on their minimum ({n} crashes)"
        reperes = [0.]
    else:
        v = P if a.aligne == "pop" else R
        paires = cycles(extremums(v, a.largeur, haut=True), a.n_phase)
        if not paires:
            raise SystemExit("aucun cycle complet dans cette fenetre : "
                             "elargir --jusqu ou baisser --largeur-creux")
        if len(paires) < 3:
            print(f"ATTENTION : {len(paires)} cycle(s) seulement, pas de "
                  "moyenne possible -- la bande d'erreur n'est pas tracee")
        pp, pr, pm = (en_phase(s, paires, a.n_phase) for s in (P, R, M))
        pa = ([en_phase(A[:, k], paires, a.n_phase) for k in range(A.shape[1])]
              if A is not None else None)
        x = np.linspace(0, 1, a.n_phase)
        quoi = "population" if a.aligne == "pop" else "resource"
        duree = np.median([f - d for d, f in paires])
        n = len(paires)
        xlabel = f"Cycle phase, {quoi} peak to {quoi} peak"
        titre = (f"Mean cycle, aligned on {quoi} maxima "
                 f"({n} cycle{'s' if n > 1 else ''}, "
                 f"median length {duree:.0f} steps)")
        reperes = [0., 1.]

    err = lambda w: np.nanstd(w, axis=0) / np.sqrt(len(w))
    hauteurs = [1, 1.25] + ([1.1] if pa is not None else [])
    taille = (a.taille if pa is None
              else (a.taille[0], a.taille[1] * 1.35))
    fig, axes = plt.subplots(len(hauteurs), 1, figsize=tuple(taille), sharex=True,
                             gridspec_kw={"height_ratios": hauteurs,
                                          "hspace": .12})
    h, b = axes[0], axes[1]
    act = axes[2] if pa is not None else None
    for ax in axes:
        for xr in reperes:
            ax.axvline(xr, color="#4A4A4A", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.grid(alpha=.25)

    h.plot(x, np.nanmean(pp, axis=0), color=POP, lw=2)
    h.set_ylabel("Population size", color=POP)
    h.tick_params(axis="y", labelcolor=POP)
    hr = h.twinx()
    hr.plot(x, np.nanmean(pr, axis=0), color=RES, lw=2)
    hr.set_ylabel("Resources amount", color=RES)
    hr.tick_params(axis="y", labelcolor=RES)

    mm, em = np.nanmean(pm, axis=0), err(pm)
    b.plot(x, mm, color=MOV, lw=.8, alpha=.3)
    if len(pm) > 1:          # a un seul cycle l'erreur-type vaut 0, pas 'petite'
        b.fill_between(x, mm - em, mm + em, color=MOV, alpha=.18, lw=0)
    if a.lissage > 1 and a.lissage < len(x):
        # 'valid' et non 'same' : le bord d'une convolution 'same' divise par
        # une fenetre pleine un nombre de points partiel, et la courbe plonge
        k = np.ones(a.lissage) / a.lissage
        d = a.lissage // 2
        b.plot(x[d:len(x) - d + (1 - a.lissage % 2)],
               np.convolve(mm, k, mode="valid"), color=MOV, lw=2.2)
    b.set_ylabel("Mean movement (cell / step)", color=MOV)
    b.tick_params(axis="y", labelcolor=MOV)
    b.set_xlim(x[0], x[-1])

    if act is not None:
        # empile dans l'ordre des couleurs : avancer en bas, c'est la part qui
        # porte le signal, et une pile se lit depuis sa base
        ordre = [3, 2, 1, 0]
        act.stackplot(x, *[np.nanmean(pa[k], axis=0) for k in ordre],
                      labels=[ACTIONS[k] for k in ordre],
                      colors=[COUL_ACT[k] for k in ordre], edgecolor="none")
        act.set_ylim(0, 1)
        act.set_ylabel("Action fraction")
        act.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=4,
                   frameon=False)
    axes[-1].set_xlabel(xlabel)

    if not a.no_titre:
        h.set_title(titre)
    nom = {"creux": "krach", "pop": "cycle_pop", "res": "cycle_res"}[a.aligne]
    out = a.out or os.path.join(a.source, "fig", f"{nom}.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    q = len(x) // 4
    print(f"{n} cycle(s)/krach(s) | mouvement : "
          f"min {np.nanmin(mm):.3f} a la phase {x[int(np.nanargmin(mm))]:.2f}, "
          f"max {np.nanmax(mm):.3f} a la phase {x[int(np.nanargmax(mm))]:.2f}")
    print(f"  1er quart {np.nanmean(mm[:q]):.3f}, dernier quart "
          f"{np.nanmean(mm[-q:]):.3f}, erreur-type ~{np.nanmean(em):.3f}")
    if pa is not None:
        bas, haut = np.nanargmin(np.nanmean(pr, axis=0)), np.nanargmax(np.nanmean(pr, axis=0))
        print("  actions  (ressources au creux -> au pic) :")
        for k, nom in enumerate(ACTIONS):
            v = np.nanmean(pa[k], axis=0)
            print(f"    {nom:<14} {v[bas]:.3f} -> {v[haut]:.3f}")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
