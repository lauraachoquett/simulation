"""plot_evo et la frise des canaux sur le MEME axe des pas.

    python -m simulation.tools.fig_evo_frise <exp_dir>
    python -m simulation.tools.fig_evo_frise <exp_dir> --depuis 500000 \\
        --fig-format pdf

Lit data/chunk_*.npz et resource_shuffles.jsonl : rien n'est rejoue. Les
ressources sont tracees par EPOQUE, chaque segment portant la couleur de
l'identite que le canal transporte alors -- la frise du bas est donc la lecture
exacte des couleurs du haut.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from simulation.data_class import color_of, label_of
from simulation.tools.plot_corr_mouvement import (_NUM, chaine_de_reprise,
                                                  ordre_initial, permutations)


def series(dossiers, taille, quoi="evo"):
    """(pas, serie_gauche, series_par_canal) sur la chaine de reprise.

    quoi="evo"  : population et quantite de ressource par canal
    quoi="prob" : population et P(manger | en vue) par canal
    """
    par_chunk = {}
    for d in dossiers:
        for f in glob.glob(os.path.join(d, "data", "chunk_*.npz")):
            par_chunk[int(_NUM.search(os.path.basename(f)).group(1))] = f
    pas, pop, res = [], [], []
    for c in sorted(par_chunk):
        with np.load(par_chunk[c]) as z:
            if "population" not in z.files:
                continue
            p = np.asarray(z["population"], float)
            if quoi == "prob":
                if not {"n_seen", "n_eaten_seen"} <= set(z.files):
                    continue
                vu = np.asarray(z["n_seen"], float)
                mg = np.asarray(z["n_eaten_seen"], float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(vu > 0, mg / vu, np.nan)
            else:
                r = np.asarray(z.get("resources", np.zeros((len(p), 1))), float)
        if r.ndim == 1:
            r = r[:, None]
        pas.append((c - 1) * taille + np.arange(len(p)))
        pop.append(p), res.append(r[:len(p)])
    if not pas:
        return None
    return np.concatenate(pas), np.concatenate(pop), np.concatenate(res)


def epoques(ids0, bascules, debut, fin):
    """[(x0, x1, ordre)] : qui porte quoi, bornes aux pas traces."""
    jalons = [(0, list(ids0))] + [(s, list(o)) for s, o in sorted(bascules)]
    out = []
    for i, (x0, ordre) in enumerate(jalons):
        x1 = jalons[i + 1][0] if i + 1 < len(jalons) else fin
        a, b = max(x0, debut), min(x1, fin)
        if b > a:
            out.append((a, b, ordre))
    return out


def reduit(x, y, bloc):
    """Moyennes par blocs : 2 millions de points ne tiennent pas dans un pdf."""
    n = (len(x) // bloc) * bloc
    if bloc <= 1 or n == 0:
        return x, y
    xr = x[:n].reshape(-1, bloc).mean(axis=1)
    with np.errstate(invalid="ignore"):       # P(manger) est NaN si rien n'est vu
        yr = np.nanmean(y[:n].reshape(-1, bloc, *y.shape[1:]), axis=1)
    return xr, yr


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--haut", default="evo", choices=["evo", "prob"],
                   help="panneau du haut : quantite de ressource (evo) ou "
                        "P(manger | en vue) par identite (prob)")
    p.add_argument("--police", type=float, default=15,
                   help="taille de police de base (defaut %(default)s)")
    p.add_argument("--bloc", type=int, default=200,
                   help="moyenner par blocs de N pas pour le trace "
                        "(defaut %(default)s, 1 = tout garder)")
    p.add_argument("--depuis", type=int, default=0)
    p.add_argument("--jusqu", type=int, default=0)
    p.add_argument("--ylim-pop", dest="ylim_pop", type=float, nargs=2, default=None)
    p.add_argument("--ylim-res", dest="ylim_res", type=float, nargs=2, default=None)
    # figure plus petite a police egale : une fois reduite dans une colonne,
    # c'est le RAPPORT police/largeur qui decide de la lisibilite
    p.add_argument("--taille", type=float, nargs=2, default=(10., 5.4),
                   help="largeur et hauteur en pouces")
    p.add_argument("--no-chaine", dest="no_chaine", action="store_true")
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    plt.rcParams.update({
        "font.size": a.police, "axes.titlesize": a.police + 2,
        "axes.labelsize": a.police + 1, "xtick.labelsize": a.police,
        "ytick.labelsize": a.police, "legend.fontsize": a.police})
    ids0, taille = ordre_initial(a.source)
    dossiers = [a.source] if a.no_chaine else chaine_de_reprise(a.source)
    bascules = []
    for d in dossiers:
        bascules += permutations(d)
    s = series(dossiers, taille, a.haut)
    if s is None:
        raise SystemExit(f"pas de data/chunk_*.npz exploitable sous {a.source}")
    pas, pop, res = s
    garde = (pas >= a.depuis) & ((pas <= a.jusqu) if a.jusqu else True)
    pas, pop, res = pas[garde], pop[garde], res[garde]
    if not len(pas):
        raise SystemExit("aucun pas dans l'intervalle demande")

    fig, (h, b) = plt.subplots(
        2, 1, figsize=tuple(a.taille), sharex=True,
        gridspec_kw={"height_ratios": [5, 1], "hspace": .08})

    xr, popr = reduit(pas, pop, a.bloc)
    h.plot(xr, popr, color="tab:red", lw=1.6)
    h.set_ylabel("Population size", color="tab:red")
    h.tick_params(axis="y", labelcolor="tab:red")
    h.grid(alpha=.3)
    if a.ylim_pop:
        h.set_ylim(*a.ylim_pop)

    hr = h.twinx()
    debut, fin = int(pas[0]), int(pas[-1]) + 1
    for x0, x1, ordre in epoques(ids0, bascules, debut, fin):
        m = (pas >= x0) & (pas < x1)
        if not m.any():
            continue
        xe, re = reduit(pas[m], res[m], a.bloc)
        for k, ident in enumerate(ordre[:res.shape[1]]):
            hr.plot(xe, re[:, k], color=color_of(int(ident)), lw=1.5)
    prob = a.haut == "prob"
    hr.set_ylabel("P(eat | in view)" if prob else "Resources amount",
                  color="#4A4A4A" if prob else "tab:green")
    hr.tick_params(axis="y", labelcolor="#4A4A4A" if prob else "tab:green")
    # P(manger) depasse rarement 0.2 : caler sur 1 ecraserait tout
    haut_y = float(np.nanmax(res)) * 1.05
    hr.set_ylim(*(a.ylim_res or (0, haut_y)))
    presents = sorted({int(i) for _, _, o in epoques(ids0, bascules, debut, fin)
                       for i in o})
    hr.legend(handles=[Line2D([0], [0], color=color_of(i), lw=2, label=label_of(i))
                       for i in presents], loc="upper right", frameon=False)

    # frise : memes bornes en x que le haut, une bande par canal
    n_can = len(ids0)
    for x0, x1, ordre in epoques(ids0, bascules, debut, fin):
        for k, ident in enumerate(ordre[:n_can]):
            b.barh(n_can - 1 - k, x1 - x0, left=x0, height=.82,
                   color=color_of(int(ident)), edgecolor="white", linewidth=.6)
    b.set_ylim(-.5, n_can - .5)
    b.set_yticks(range(n_can))
    b.set_yticklabels([f"c{n_can - 1 - k}" for k in range(n_can)],
                      fontsize=a.police - 2)
    b.set_xlim(debut, fin)
    b.set_xlabel("Simulation step")
    b.spines[["right", "top"]].set_visible(False)
    h.set_xlim(debut, fin)

    if not a.no_titre:
        h.set_title("Simulation dynamic")
    nom = "plot_evo_frise" if a.haut == "evo" else "prob_eat_frise"
    out = a.out or os.path.join(a.source, "fig", f"{nom}.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{len(pas)} pas, {len(bascules)} permutation(s)")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
