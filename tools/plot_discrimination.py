"""Discrimination le long de la lignee : P(manger X | X en vue) rapportee a l'offre.

    python -m simulation.tools.plot_discrimination <exp_dir>
    python -m simulation.tools.plot_discrimination <exp_dir>/lod_global \\
        --identites poison --fig-format pdf

Lit lod/lab/evaluation.npz, ecrit par tools/lineage_lab (ou tools/lineage_global
pour une chaine de reprises) : rien n'est reevalue.

Haut : les probabilites brutes, une par identite, plus la reference toutes
ressources confondues. Bas : le rapport p_X / p_toutes. A 1 l'ancetre mange X
comme n'importe quelle ressource qu'il croise, sous 1 il l'evite, au-dessus il
le choisit. C'est ce rapport, et non p_X seule, qui separe un vrai tri d'une
simple baisse d'appetit.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.data_class import color_of, label_of
from simulation.utils.plots import _med_glissante

GRIS = "#6E6E6E"


def charge(source):
    f = os.path.join(source, "lod", "lab", "evaluation.npz")
    if not os.path.exists(f):
        raise SystemExit(f"{f} absent : lancer d'abord "
                         f"python -m simulation.tools.lineage_lab {source}")
    with np.load(f) as d:
        return {c: np.asarray(d[c]) for c in d.files}


def ordre_initial(source):
    f = os.path.join(source, "config.json")
    if not os.path.exists(f):
        return None
    return [int(r["id"]) for r in json.load(open(f)).get("resources", [])] or None


def journal(source):
    """(permutations, coutures) en pas, depuis resource_shuffles.jsonl.

    Une permutation compte quand l'affectation canal -> identite CHANGE
    vraiment : la premiere ligne du journal peut deja etre l'ordre initial.
    """
    f = os.path.join(source, "resource_shuffles.jsonl")
    if not os.path.exists(f):
        return [], []
    perm, coutures, ordre = [], [], ordre_initial(source)
    for ligne in sorted((json.loads(l) for l in open(f) if l.strip()),
                        key=lambda e: e["step"]):
        if ligne.get("couture"):
            coutures.append(int(ligne["step"]))
        o = list(ligne["order_ids"])
        if ordre is not None and o != ordre:
            perm.append(int(ligne["step"]))
        ordre = o
    return perm, coutures


def identites(cfg_path, e):
    """Identites presentes, dans l'ordre des colonnes p_<label>."""
    noms = [c[2:] for c in e if c.startswith("p_") and c != "p_toutes"
            and not c.endswith("_par_graine")]
    ids = []
    for i in range(16):
        if label_of(i) in noms:
            ids.append(i)
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier portant lod/lab/evaluation.npz")
    p.add_argument("--identites", nargs="+", default=None,
                   help="identites a tracer (defaut : toutes celles presentes)")
    p.add_argument("--lissage", type=int, default=9,
                   help="mediane glissante, en ancetres (defaut %(default)s)")
    p.add_argument("--depuis", type=int, default=0, help="pas de naissance min")
    p.add_argument("--jusqu", type=int, default=0, help="pas de naissance max")
    p.add_argument("--ymax", type=float, default=None,
                   help="borne haute du rapport (defaut : automatique)")
    p.add_argument("--police", type=float, default=13)
    p.add_argument("--taille", type=float, nargs=2, default=(11.5, 7.))
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    plt.rcParams.update({"font.size": a.police, "axes.labelsize": a.police,
                         "axes.titlesize": a.police + 1,
                         "legend.fontsize": a.police - 2})
    e = charge(a.source)
    x = np.asarray(e["born"], float)
    garde = (x >= a.depuis) & ((x <= a.jusqu) if a.jusqu else True)
    if not garde.any():
        raise SystemExit("aucun ancetre dans l'intervalle demande")

    # denominateur : toutes les ressources si lineage_lab l'a mesure, sinon la
    # greediness, qui ne conditionne que sur les ressources a gain positif
    if "p_toutes" in e:
        denom, nom_denom = e["p_toutes"], "any resource"
    elif "greediness" in e:
        denom, nom_denom = e["greediness"], "edible resource"
        print("p_toutes absent : denominateur = greediness, conditionnee aux "
              "seules ressources a gain positif (relancer lineage_lab pour la "
              "reference sur les trois ressources)")
    else:
        raise SystemExit("ni p_toutes ni greediness dans evaluation.npz")
    denom = np.asarray(denom, float)

    ids = ([i for i in range(16) if label_of(i) in a.identites]
           if a.identites else identites(a.source, e))
    ids = [i for i in ids if f"p_{label_of(i)}" in e]
    if not ids:
        dispo = sorted(c[2:] for c in e if c.startswith("p_"))
        raise SystemExit(f"aucune identite demandee dans le fichier ({dispo})")

    perm, coutures = journal(a.source)
    fig, (h, b) = plt.subplots(2, 1, figsize=tuple(a.taille), sharex=True,
                               gridspec_kw={"hspace": .12})
    xg = x[garde]
    for ax in (h, b):
        for s in perm:
            if xg.min() <= s <= xg.max():
                ax.axvline(s, color="grey", lw=1, ls=":", zorder=1)
        for s in coutures:
            if xg.min() <= s <= xg.max():
                ax.axvline(s, color="black", lw=1.4, zorder=2)

    h.plot(xg, _med_glissante(denom[garde], a.lissage), color=GRIS, lw=2,
           label=f"P(eat | {nom_denom} in view)")
    hauts = []
    for i in ids:
        v = np.asarray(e[f"p_{label_of(i)}"], float)[garde]
        h.plot(xg, _med_glissante(v, a.lissage), color=color_of(i), lw=2,
               label=f"P(eat {label_of(i)} | {label_of(i)} in view)")
    h.set_ylabel("conditional probability")
    h.set_ylim(0, None)
    h.grid(alpha=.25)
    h.legend(loc="upper right", frameon=False, ncol=1)

    b.axhline(1, color="black", lw=1.2, zorder=3)
    with np.errstate(all="ignore"):
        for i in ids:
            v = np.asarray(e[f"p_{label_of(i)}"], float)
            r = np.divide(v, denom, out=np.full(v.shape, np.nan),
                          where=denom > 0)[garde]
            vs, ds = e.get(f"p_{label_of(i)}_par_graine"), e.get("p_toutes_par_graine")
            if vs is not None and ds is not None:
                rs = np.divide(np.asarray(vs, float), np.asarray(ds, float),
                               out=np.full(np.shape(vs), np.nan),
                               where=np.asarray(ds, float) > 0)[garde]
                ok = np.isfinite(rs).any(axis=1)
                b.vlines(xg[ok], np.nanmin(rs[ok], axis=1), np.nanmax(rs[ok], axis=1),
                         color=color_of(i), alpha=.18, lw=1.1)
            b.plot(xg, r, marker=".", ls="none", ms=4, alpha=.35, color=color_of(i))
            lisse = _med_glissante(r, a.lissage)
            b.plot(xg, lisse, color=color_of(i), lw=2, label=label_of(i))
            hauts.append(np.nanpercentile(r, 99) if np.isfinite(r).any() else 1)
    b.set_ylabel(f"p(X) / p({nom_denom})")
    b.set_xlabel("birth step")
    b.set_ylim(0, a.ymax or max(1.25, float(np.nanmax(hauts)) * 1.1))
    b.grid(alpha=.25)
    b.legend(loc="upper right", frameon=False, ncol=len(ids),
             title="selectivity")

    if not a.no_titre:
        h.set_title("Selectivity along the line of descent")
    noms = "_".join(label_of(i) for i in ids)
    out = a.out or os.path.join(a.source, "fig", "lod",
                                f"discrimination_{noms}.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{int(garde.sum())} ancetre(s), {len(perm)} permutation(s), "
          f"{len(coutures)} couture(s)")
    for i in ids:
        v = np.asarray(e[f"p_{label_of(i)}"], float)
        with np.errstate(all="ignore"):
            r = np.divide(v, denom, out=np.full(v.shape, np.nan), where=denom > 0)
        moitie = len(xg) // 2
        rg = r[garde]
        print(f"  {label_of(i)} : rapport median {np.nanmedian(rg):.2f} "
              f"(1re moitie {np.nanmedian(rg[:moitie]):.2f}, "
              f"2e moitie {np.nanmedian(rg[moitie:]):.2f})")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
