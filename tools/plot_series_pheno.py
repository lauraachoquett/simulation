"""Evolution de mesures par genome, un panneau par mesure, empiles.

    python -m simulation.tools.plot_series_pheno <dir>/replay
    python -m simulation.tools.plot_series_pheno <dir>/replay \\
        --mesures mean_speed age --env alone --fig-format pdf

Par instant : mediane sur les genomes, bande interquartile. Lit les
lab_data/chunk_N_pheno_<env>.npz, rien n'est rejoue.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.tools.box_chunks import (MESURES, a_mange, charge, data_dir_de,
                                         envs_dispo, nom_court, taille_de_chunk)

_NUM = re.compile(r"chunk_(\d+)_pheno_")
COULEURS = ("#1D5C8F", "#C1121F", "#2A9131", "#9C27B0")


def titre_de(col):
    for c, t, _ in MESURES.values():
        if c == col:
            return t
    return col


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--env", default=None,
                   help="environnement de test (defaut : le premier 'alone')")
    p.add_argument("--mesures", nargs="+", default=["mean_speed", "age"],
                   help="colonnes des fichiers pheno, de haut en bas")
    p.add_argument("--pas-chunks", dest="pas_chunks", type=int, default=1)
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("--mange", action="store_true",
                   help="ne garder que les genomes ayant mange au moins une fois")
    p.add_argument("--police", type=float, default=13)
    p.add_argument("--taille", type=float, nargs=2, default=(11., 6.2))
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    plt.rcParams.update({"font.size": a.police, "axes.labelsize": a.police,
                         "axes.titlesize": a.police + 1,
                         "legend.fontsize": a.police - 1})
    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {a.source}")
    envs = envs_dispo(data_dir)
    env = a.env or next((e for e in envs if e.startswith("alone")), envs[0])
    taille = a.chunk_size or taille_de_chunk(a.source)
    chunks = sorted(int(_NUM.search(os.path.basename(f)).group(1))
                    for f in glob.glob(os.path.join(
                        data_dir, f"chunk_*_pheno_{env}.npz")))
    chunks = chunks[::max(a.pas_chunks, 1)]
    if not chunks:
        raise SystemExit(f"aucun chunk pour {env} (dispo : {', '.join(envs)})")

    colonnes = list(a.mesures) + (["ever_ate", "greediness"] if a.mange else [])
    x, stats, effectifs = [], {m: [] for m in a.mesures}, []
    for c in chunks:
        d = charge(data_dir, c, env, colonnes)
        if d is None:
            continue
        garde = (a_mange(d["ever_ate"], d.get("greediness"))
                 if (a.mange and "ever_ate" in d) else None)
        n = 0
        for m in a.mesures:
            v = np.asarray(d.get(m, []), float)
            if garde is not None and len(v):
                v = v[garde]
            v = v[np.isfinite(v)]
            stats[m].append((np.median(v), np.percentile(v, 25),
                             np.percentile(v, 75)) if v.size
                            else (np.nan, np.nan, np.nan))
            n = max(n, v.size)
        x.append(c * taille), effectifs.append(n)
    if not x:
        raise SystemExit("aucun genome exploitable")
    x = np.array(x, float) / 1e6

    fig, axes = plt.subplots(len(a.mesures), 1, figsize=tuple(a.taille),
                             sharex=True, squeeze=False,
                             gridspec_kw={"hspace": .12})
    for ax, m, coul in zip(axes[:, 0], a.mesures, COULEURS * 3):
        s = np.array(stats[m], float)
        ax.plot(x, s[:, 0], color=coul, lw=1.8)
        ax.fill_between(x, s[:, 1], s[:, 2], color=coul, alpha=.2, lw=0)
        ax.set_ylabel(titre_de(m))
        ax.grid(alpha=.3)
    axes[-1, 0].set_xlabel("simulation step (M)")
    if not a.no_titre:
        fig.suptitle(f"{nom_court(env)} — median and IQR over "
                     f"{min(effectifs)}–{max(effectifs)} genomes",
                     fontsize=a.police + 1)
    fig.tight_layout(rect=[0, 0, 1, .95 if not a.no_titre else 1])
    quoi = "_".join(a.mesures)
    out = a.out or os.path.join(a.source, "fig",
                                f"serie_{quoi}_{env}.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{env} : {len(x)} instant(s), {min(effectifs)}–{max(effectifs)} genomes")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
