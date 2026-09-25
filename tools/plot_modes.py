"""Deux modes dans un env de test : trouver la nourriture, ou ne jamais la trouver.

    python -m simulation.tools.plot_modes <dir> --env alone_blob1x40_s1
    python -m simulation.tools.plot_modes <dir> --env alone_blob1x40_s1 --pas 100000

Lit les lab_data/chunk_N_pheno_<env>.npz : une ligne par genome, rien n'est rejoue.
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

TROUVE, JAMAIS = "#1D5C8F", "#C1121F"
_NUM = re.compile(r"chunk_(\d+)_pheno_")


def data_dir_de(chemin):
    for c in (os.path.join(chemin, "replay", "lab_data"),
              os.path.join(chemin, "lab_data"), chemin):
        if glob.glob(os.path.join(c, "chunk_*_pheno_*.npz")):
            return c
    return None


def charge(data_dir, env, taille, pas):
    """[(step, age, a_mange, t_explore)] par chunk, trie."""
    etapes = []
    for f in sorted(glob.glob(os.path.join(data_dir, f"chunk_*_pheno_{env}.npz")),
                    key=lambda f: int(_NUM.search(os.path.basename(f)).group(1))):
        chunk = int(_NUM.search(os.path.basename(f)).group(1))
        with np.load(f) as z:
            if "ever_ate" not in z.files:
                continue
            age = np.asarray(z["age"], float)
            mange = np.nan_to_num(np.asarray(z["ever_ate"], float)) > .5
            t_exp = (np.asarray(z["t_explore"], float) if "t_explore" in z.files
                     else np.full(len(age), np.nan))
        etapes.append((chunk * taille, age, mange, t_exp))
    if pas:
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
    p.add_argument("--env", default=None,
                   help="environnement des phenotypes (defaut : le premier 'alone')")
    p.add_argument("--pas", type=int, default=0, metavar="N",
                   help="un point tous les N pas de simulation au moins")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("-o", "--out", default=None,
                   help="defaut <source>/fig/modes_<env>.png")
    a = p.parse_args()

    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {a.source}")
    envs = sorted({re.fullmatch(r"chunk_\d+_pheno_(.+)\.npz", os.path.basename(f)).group(1)
                   for f in glob.glob(os.path.join(data_dir, "chunk_*_pheno_*.npz"))})
    env = a.env or next((e for e in envs if e.startswith("alone")), envs[0])
    taille = a.chunk_size
    if taille is None:
        f = os.path.join(a.source, "config.json")
        parent = os.path.join(os.path.dirname(os.path.abspath(a.source)), "config.json")
        taille = int(json.load(open(f if os.path.exists(f) else parent)).get(
            "chunk_size", 1000)) if (os.path.exists(f) or os.path.exists(parent)) else 1000

    etapes = charge(data_dir, env, taille, a.pas)
    if not etapes:
        raise SystemExit(f"aucun phenotype pour {env} (dispo : {', '.join(envs)})")
    x = np.array([s for s, _, _, _ in etapes])
    part = np.array([m.mean() for _, _, m, _ in etapes])
    print(f"{env} : {len(etapes)} chunk(s), part qui mange "
          f"{part.min():.2f} a {part.max():.2f}")

    fig, (h, b) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.6]})
    h.plot(x, part, color=TROUVE, lw=2, marker="o", ms=3.5)
    h.set_ylabel("share that ate at least once")
    h.set_ylim(0, 1)
    h.grid(alpha=.3)
    h.set_title(f"Two modes in {env}: eating at least once, or never eating",
                fontsize=12)

    rng = np.random.default_rng(0)
    largeur = (x[1] - x[0]) * .22 if len(x) > 1 else 1
    for (s, age, mange, _) in etapes:
        for m, coul in ((mange, TROUVE), (~mange, JAMAIS)):
            if m.any():
                b.scatter(s + rng.uniform(-largeur, largeur, m.sum()), age[m],
                          s=9, color=coul, alpha=.45, edgecolors="none")
    b.scatter([], [], color=TROUVE, s=28, label="ate at least once")
    b.scatter([], [], color=JAMAIS, s=28, label="never ate")
    b.set_ylabel("lifespan in the lab (steps)")
    b.set_xlabel("simulation step")
    b.grid(alpha=.3)
    b.legend(frameon=False, ncol=2)

    fig.tight_layout()
    out = a.out or os.path.join(a.source, "fig", f"modes_{env}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
