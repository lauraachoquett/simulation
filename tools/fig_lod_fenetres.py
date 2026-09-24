"""La ligne de descendance en plusieurs simplex, une fenetre de temps par panneau.

    python -m simulation.tools.fig_lod_fenetres <exp_dir>
    python -m simulation.tools.fig_lod_fenetres <exp_dir> --pas 500000 -o fig/lod.png

Lit lod/lab/evaluation.npz, ecrit par tools/lineage_lab : rien n'est reevalue.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from simulation.utils.plots import _bary, _cadre_simplex

LOD, DISPO = "#C1121F", "black"


def charge(exp_dir):
    f = os.path.join(exp_dir, "lod", "lab", "evaluation.npz")
    if not os.path.exists(f):
        raise SystemExit(f"{f} absent : lancer d'abord "
                         "python -m simulation.tools.lineage_lab <exp_dir>")
    with np.load(f) as d:
        regime, born = np.asarray(d["regime"], float), np.asarray(d["born"])
        post = np.asarray(d["post_shuffle"], bool)
        age = np.asarray(d["age"], float) if "age" in d else np.full(len(born), np.nan)
        dispo = np.asarray(d["disponible"], float) if "disponible" in d else None
    total = regime.sum(axis=1)
    ok = np.isfinite(regime).all(axis=1) & (total > 0)
    if dispo is not None and dispo.sum() > 0:
        dispo = dispo / dispo.sum()
    return regime[ok] / total[ok, None], born[ok], post[ok], age[ok], dispo


def coutures(exp_dir):
    """Pas des reprises, inscrites par tools/lineage_global dans le journal."""
    f = os.path.join(exp_dir, "resource_shuffles.jsonl")
    if not os.path.exists(f):
        return []
    return [json.loads(l)["step"] for l in open(f)
            if l.strip() and json.loads(l).get("couture")]


def panneau(ax, p, couleur, post, dispo, bornes, cmap, norm, etiquettes=True,
            n_coutures=0):
    x, y = _bary(p[:, 0], p[:, 1], p[:, 2])
    _cadre_simplex(ax)
    if not etiquettes:      # les sommets debordent : un seul panneau les porte
        for t in list(ax.texts):
            if t.get_text() in ("good", "medium", "poison"):
                t.remove()
    if len(x) > 1:
        seg = np.stack([np.column_stack([x[:-1], y[:-1]]),
                        np.column_stack([x[1:], y[1:]])], axis=1)
        ax.add_collection(LineCollection(seg, colors="0.6", linewidths=1.1, zorder=3))
    ax.scatter(x, y, c=couleur, cmap=cmap, norm=norm, s=46, zorder=4,
               edgecolors="white", linewidths=.4)
    if post.any():          # permutation entre cet ancetre et le precedent
        ax.scatter(x[post], y[post], s=110, marker="D", facecolors="none",
                   edgecolors="0.2", linewidths=1.2, zorder=5)
    if dispo is not None:
        ax.scatter(*_bary(*dispo), marker="o", s=170, facecolor="none",
                   edgecolor=DISPO, linewidth=1.8, zorder=6)
    couture = f"\n{n_coutures} resume(s)" if n_coutures else ""
    ax.set_title(f"{bornes[0] / 1e6:.1f}–{bornes[1] / 1e6:.1f} M steps"
                 f"   ({len(x)} ancestors){couture}", fontsize=11)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dir")
    p.add_argument("--pas", type=int, default=1_000_000,
                   help="largeur d'une fenetre en pas (defaut %(default)s)")
    p.add_argument("--age-max", dest="age_max", type=float, default=None,
                   help="haut de l'echelle de couleur (defaut : le maximum vu)")
    p.add_argument("--colonnes", type=int, default=0,
                   help="panneaux par ligne (defaut : tous sur une ligne)")
    p.add_argument("-o", "--out", default=None,
                   help="defaut <exp_dir>/fig/lod/lod_simplex_fenetres.png")
    a = p.parse_args()

    p_reg, born, post, age, dispo = charge(a.exp_dir)
    bords = np.arange(0, born.max() + a.pas, a.pas)
    fenetres = [(int(lo), int(hi)) for lo, hi in zip(bords[:-1], bords[1:])
                if ((born >= lo) & (born < hi)).sum() >= 2]
    if not fenetres:
        raise SystemExit("moins de deux ancetres par fenetre : baisser --pas")

    cout = coutures(a.exp_dir)
    if cout:
        print(f"{len(cout)} couture(s) de reprise : {cout}")
    # couleur = duree de vie au lab, meme echelle que les autres simplex ;
    # a defaut, le pas de naissance
    par_age = np.isfinite(age).any()
    val = age if par_age else born.astype(float)
    haut = a.age_max or (np.nanmax(val) if par_age else val.max())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0 if par_age else val.min(), haut)
    nc = a.colonnes or len(fenetres)
    nl = -(-len(fenetres) // nc)
    fig, axes = plt.subplots(nl, nc, figsize=(4.6 * nc, 4.6 * nl), squeeze=False)
    # les sommets du triangle debordent de l'axe : sans ecart les etiquettes
    # "medium" et "poison" de deux panneaux voisins se chevauchent
    fig.subplots_adjust(wspace=.18, hspace=.3)
    for ax, (lo, hi) in zip(axes.ravel(), fenetres):
        m = (born >= lo) & (born < hi)
        panneau(ax, p_reg[m], val[m], post[m], dispo, (lo, hi), cmap, norm,
                etiquettes=(ax is axes.ravel()[0]),
                n_coutures=sum(lo <= c < hi for c in cout))
    for ax in axes.ravel()[len(fenetres):]:
        ax.axis("off")

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=axes, shrink=.7, pad=.015)
    cb.set_label("lifespan in the lab (steps)" if par_age else "birth step")
    fig.suptitle("Line of descent through time, diet composition by identity",
                 fontsize=14)
    out = a.out or os.path.join(a.exp_dir, "fig", "lod",
                                "lod_simplex_fenetres.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{len(fenetres)} fenetre(s) de {a.pas} pas")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
