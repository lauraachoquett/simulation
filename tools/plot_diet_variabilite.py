"""Variabilite du regime dans la population, au fil de la simulation.

    python -m simulation.tools.plot_diet_variabilite <exp_dir|replay|fusion>
    python -m simulation.tools.plot_diet_variabilite <dir> --pas-chunks 20

Lit les lab_data/simplex_chunk_*.npz : un point par genome evalue, deja ecrits
par le run ou par tools/replay_lab. Rien n'est reevalue.
"""
import argparse
import glob
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--pas", type=int, default=0, metavar="N",
                   help="un point tous les N pas de simulation au moins")
    p.add_argument("-o", "--out", default=None,
                   help="defaut <source>/fig/diet_variabilite.png")
    a = p.parse_args()

    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de simplex_chunk_*.npz sous {a.source}")
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

    fig, (h, b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    for k, i in enumerate(ids):
        h.plot(x, med[:, i], color=color_of(i), lw=2, label=label_of(i))
        h.fill_between(x, p25[:, i], p75[:, i], color=color_of(i), alpha=.18)
    h.set_ylabel("share of the diet")
    h.set_ylim(0, 1)
    h.grid(alpha=.3)
    h.legend(frameon=False, ncol=len(ids))
    h.set_title("Diet composition in the population: median and p25–p75 across genomes",
                fontsize=12)

    b.plot(x, etal, color="#4C4C4C", lw=2)
    b.set_ylabel("spread in the simplex")
    b.set_xlabel("simulation step")
    b.grid(alpha=.3)
    b.set_title("Dispersion: mean distance to the centroid "
                f"({n.min()}–{n.max()} genomes per point)", fontsize=11)

    fig.tight_layout()
    out = a.out or os.path.join(a.source, "fig", "diet_variabilite.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"{len(etapes)} chunk(s), pas {x[0]} a {x[-1]}")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
