"""Variabilite du regime dans la population, au fil de la simulation.

    python -m simulation.tools.plot_diet_variabilite <exp_dir|replay|fusion>
    python -m simulation.tools.plot_diet_variabilite <dir> --pas-chunks 20

Lit les lab_data/simplex_chunk_*.npz : un point par genome evalue, deja ecrits
par le run ou par tools/replay_lab. Rien n'est reevalue.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.data_class import color_of, label_of
from simulation.utils.plots import _bary

_NUM = re.compile(r"simplex_chunk_(\d+)\.npz$")


def data_dir_de(chemin):
    for c in (os.path.join(chemin, "replay", "lab_data"),
              os.path.join(chemin, "lab_data"), chemin):
        if glob.glob(os.path.join(c, "simplex_chunk_*.npz")):
            return c
    return None


def charge(data_dir, pas):
    """[(step, compositions (n,3) par identite, ids)] trie par pas."""
    etapes = []
    for f in sorted(glob.glob(os.path.join(data_dir, "simplex_chunk_*.npz")),
                    key=lambda f: int(_NUM.search(os.path.basename(f)).group(1))):
        with np.load(f) as z:
            eaten, ids, step = np.asarray(z["eaten"], float), z["ids"], int(z["step"])
        par_id = np.zeros_like(eaten)
        for k, i in enumerate(ids):
            par_id[:, int(i)] = eaten[:, k]
        total = par_id.sum(axis=1)
        ok = total > 0
        if ok.sum() >= 3:
            etapes.append((step, par_id[ok] / total[ok, None], [int(i) for i in ids]))
    if pas:              # un point tous les `pas` pas au moins
        garde, dernier = [], None
        for e in etapes:
            if dernier is None or e[0] - dernier >= pas:
                garde.append(e)
                dernier = e[0]
        etapes = garde
    return etapes


def charge_graines(exp_dir):
    """[(pas de naissance, compositions (S,3), ids)] : un ancetre par permutation.

    Chaque point est une GRAINE d'environnement, pas un ancetre : la dispersion
    mesuree est celle des grilles, a genome fixe.
    """
    f = os.path.join(exp_dir, "lod", "lab", "graines_avant.npz")
    if not os.path.exists(f):
        raise SystemExit(f"{f} absent : lancer lineage_lab avec --graines-avant N")
    with np.load(f) as d:
        regime, born = np.asarray(d["regime"], float), np.asarray(d["born"])
    etapes = []
    for k, b in enumerate(born):
        c = regime[k]
        total = c.sum(axis=1)
        ok = np.isfinite(c).all(axis=1) & (total > 0)
        if ok.sum() >= 3:
            etapes.append((int(b), c[ok] / total[ok, None], [0, 1, 2]))
    return sorted(etapes)


def pas_de_reprise(exp_dir):
    """Pas des reprises : coutures inscrites par lineage_global, ou resume_chunk.

    L'arbre genealogique repart de zero a une reprise : la fenetre qui la
    contient melange deux lignees et donne un ecart-type aberrant.
    """
    pas = []
    j = os.path.join(exp_dir, "resource_shuffles.jsonl")
    if os.path.exists(j):
        pas += [json.loads(l)["step"] for l in open(j)
                if l.strip() and json.loads(l).get("couture")]
    c = os.path.join(exp_dir, "config.json")
    if os.path.exists(c):
        cfg = json.load(open(c))
        if cfg.get("resume_from"):
            pas.append(int(cfg.get("resume_chunk", 0)) * int(cfg.get("chunk_size", 1000)))
    return sorted(set(pas))


def charge_lod(exp_dir, fenetre, reprises=()):
    """[(pas milieu, compositions (n,3) par identite, ids)] par fenetre de temps.

    La dispersion mesuree est alors celle des ancetres SUCCESSIFS de la lignee,
    pas celle de la population : elle dit si le regime derive ou se fixe.
    """
    f = os.path.join(exp_dir, "lod", "lab", "evaluation.npz")
    if not os.path.exists(f):
        raise SystemExit(f"{f} absent : lancer d'abord "
                         "python -m simulation.tools.lineage_lab <exp_dir>")
    with np.load(f) as d:
        regime, born = np.asarray(d["regime"], float), np.asarray(d["born"])
    total = regime.sum(axis=1)
    ok = np.isfinite(regime).all(axis=1) & (total > 0)
    comp, born = regime[ok] / total[ok, None], born[ok]
    bords = np.arange(0, born.max() + fenetre, fenetre)
    etapes = []
    saute = 0
    for lo, hi in zip(bords[:-1], bords[1:]):
        if any(lo <= r < hi for r in reprises):
            saute += 1
            continue
        m = (born >= lo) & (born < hi)
        if m.sum() >= 3:
            etapes.append((int((lo + hi) // 2), comp[m], [0, 1, 2]))
    if saute:
        print(f"{saute} fenetre(s) ecartee(s) : elles contiennent une reprise")
    return etapes


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--pas", type=int, default=0, metavar="N",
                   help="un point tous les N pas de simulation au moins")
    p.add_argument("--lod", action="store_true",
                   help="la ligne de descendance au lieu de la population")
    p.add_argument("--graines", action="store_true",
                   help="les ancetres d'avant permutation, un point par graine "
                        "d'environnement (lod/lab/graines_avant.npz)")
    p.add_argument("--fenetre", type=int, default=200_000, metavar="N",
                   help="largeur d'une fenetre avec --lod (defaut %(default)s)")
    p.add_argument("--garder-reprises", dest="garder_reprises",
                   action="store_true",
                   help="garder les fenetres qui contiennent une reprise "
                        "(ecartees par defaut : l'arbre y repart de zero)")
    p.add_argument("--style", choices=["violon", "nuage", "les-deux"],
                   default="violon", help="forme de la distribution (defaut %(default)s)")
    p.add_argument("-o", "--out", default=None,
                   help="defaut <source>/fig/diet_variabilite.png")
    a = p.parse_args()

    data_dir = None if a.lod else data_dir_de(a.source)
    # un dossier de lignee (ou de fusion) n'a pas de simplex de population :
    # bascule automatique plutot qu'un message d'erreur
    if data_dir is None and os.path.exists(
            os.path.join(a.source, "lod", "lab", "evaluation.npz")):
        if not a.lod:
            print("pas de simplex de population : la lignee est utilisee")
        a.lod = True
    if a.graines:
        etapes = charge_graines(a.source)
    elif a.lod:
        rep = () if a.garder_reprises else pas_de_reprise(a.source)
        if rep:
            print(f"reprise(s) au pas {rep}")
        etapes = charge_lod(a.source, a.fenetre, rep)
    else:
        if data_dir is None:
            raise SystemExit(f"ni simplex_chunk_*.npz ni lod/lab/evaluation.npz "
                             f"sous {a.source}")
        etapes = charge(data_dir, a.pas)
    if len(etapes) < 2:
        raise SystemExit("moins de deux chunks exploitables")

    x = np.array([s for s, _, _ in etapes])
    ids = etapes[0][2]
    med = np.array([np.median(c, axis=0) for _, c, _ in etapes])
    p25 = np.array([np.percentile(c, 25, axis=0) for _, c, _ in etapes])
    p75 = np.array([np.percentile(c, 75, axis=0) for _, c, _ in etapes])
    n = np.array([len(c) for _, c, _ in etapes])
    # dispersion dans le simplex : distance moyenne au barycentre, en coordonnees
    # du triangle, donc comparable d'un chunk a l'autre
    etal = []
    for _, c, _ in etapes:
        px, py = _bary(c[:, 0], c[:, 1], c[:, 2])
        etal.append(np.hypot(px - px.mean(), py - py.mean()).mean())
    etal = np.array(etal)

    nom = "diet_variabilite_graines.png" if a.graines else "diet_variabilite.png"
    largeur = max(11, 1.1 * len(etapes) + 3)
    fig, h = plt.subplots(figsize=(largeur, 5.6))
    pos = np.arange(len(etapes), dtype=float)
    ecart = .26
    rng = np.random.default_rng(0)
    for k, i in enumerate(ids):
        dx = (k - (len(ids) - 1) / 2) * ecart
        donnees = [c[:, i] for _, c, _ in etapes]
        if a.style in ("violon", "les-deux"):
            vp = h.violinplot(donnees, positions=pos + dx, widths=ecart * .92,
                              showextrema=False, showmedians=False)
            for corps in vp["bodies"]:
                corps.set_facecolor(color_of(i))
                corps.set_edgecolor("white")
                corps.set_alpha(.75)
        if a.style in ("nuage", "les-deux"):
            for p_, d_ in zip(pos + dx, donnees):
                h.scatter(p_ + rng.uniform(-.06, .06, len(d_)), d_, s=9,
                          color=color_of(i), alpha=.45 if a.style == "les-deux" else .7,
                          edgecolors="none", zorder=3)
        h.plot([], [], color=color_of(i), lw=6, label=label_of(i))

    h.set_ylabel("share of the diet")
    h.set_ylim(0, 1)
    h.grid(alpha=.3, axis="y")
    h.legend(frameon=False, ncol=len(ids))
    if a.graines:
        quoi, par = "of the ancestor born before each permutation", "per lab seed"
    elif a.lod:
        quoi, par = "along the line of descent", "per window"
    else:
        quoi, par = "in the population", "per window"
    h.set_title(f"Diet composition {quoi}: full distribution {par}", fontsize=12)

    h.set_xticks(pos[::max(1, len(pos) // 12)])
    h.set_xticklabels([f"{v / 1e6:.2f}M" for v in x[::max(1, len(pos) // 12)]],
                      rotation=45, ha="right", fontsize=9)
    h.set_xlabel("simulation step")
    fig.tight_layout()
    out = a.out or os.path.join(a.source, "fig",
                                "lod" if (a.lod or a.graines) else "", nom)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{len(etapes)} fenetre(s), pas {x[0]} a {x[-1]}")
    print(f"Figure saved: {out}")

    # dispersion : figure a part, ce n'est pas la meme grandeur que les parts
    fig2, b = plt.subplots(figsize=(min(largeur, 8.5), 3.4))
    b.plot(pos, etal, color="#4C4C4C", lw=2, marker="o", ms=3.5)
    b.set_ylabel("spread in the simplex")
    b.set_xlabel("simulation step")
    pas_etiq = max(1, len(pos) // 12)      # une etiquette sur douze au plus
    b.set_xticks(pos[::pas_etiq])
    b.set_xticklabels([f"{v / 1e6:.2f}M" for v in x[::pas_etiq]], rotation=45,
                      ha="right", fontsize=9)
    b.grid(alpha=.3)
    unite = ("seeds per ancestor" if a.graines else
             "ancestors per window" if a.lod else "genomes per point")
    b.set_title("Dispersion in the simplex: mean distance to the centroid "
                f"({n.min()}–{n.max()} {unite})", fontsize=11)
    fig2.tight_layout()
    racine, ext = os.path.splitext(out)
    out2 = f"{racine}_dispersion{ext}"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"Figure saved: {out2}")


if __name__ == "__main__":
    main()
