"""Correlation entre deux mesures par genome, sur les donnees de lab.

    python -m simulation.tools.plot_corr_pheno <dir> --x mean_speed --y age
    python -m simulation.tools.plot_corr_pheno <dir> --env alone --chunks 100 2000

Lit les lab_data/chunk_N_pheno_<env>.npz : une ligne par genome, rien n'est
rejoue. Panneau de gauche : un point par genome, couleur = instant du run.
Panneau de droite : la correlation calculee instant par instant, qui dit si la
pente globale est un vrai lien ou un melange d'epoques.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.tools.box_chunks import (MESURES, a_mange, charge, data_dir_de,
                                         envs_dispo, nom_court, taille_de_chunk)

_NUM = re.compile(r"chunk_(\d+)_pheno_")


def titre_de(col):
    for _, (c, t, _) in MESURES.items():
        if c == col:
            return t
    return col


def chunks_dispo(data_dir, env):
    import glob
    return sorted(int(_NUM.search(os.path.basename(f)).group(1))
                  for f in glob.glob(os.path.join(data_dir,
                                                  f"chunk_*_pheno_{env}.npz")))


def correlations(x, y):
    if len(x) < 3:
        return np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    rg = lambda v: np.argsort(np.argsort(v))
    return r, float(np.corrcoef(rg(x), rg(y))[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--env", default=None,
                   help="environnement de test (defaut : le premier 'alone')")
    p.add_argument("--x", default="mean_speed", help="colonne en abscisse")
    p.add_argument("--y", default="age", help="colonne en ordonnee")
    p.add_argument("--chunks", type=int, nargs="+", default=None,
                   help="instants a garder (defaut : tous)")
    p.add_argument("--pas-chunks", dest="pas_chunks", type=int, default=1,
                   help="un chunk sur N (defaut %(default)s)")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("--mange", action="store_true",
                   help="ne garder que les genomes ayant mange au moins une fois")
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--police", type=float, default=12)
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    plt.rcParams.update({"font.size": a.police, "axes.labelsize": a.police + 1,
                         "axes.titlesize": a.police + 1})
    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {a.source}")
    envs = envs_dispo(data_dir)
    env = a.env or next((e for e in envs if e.startswith("alone")), envs[0])
    taille = a.chunk_size or taille_de_chunk(a.source)
    chunks = a.chunks or chunks_dispo(data_dir, env)[::max(a.pas_chunks, 1)]
    if not chunks:
        raise SystemExit(f"aucun chunk pour {env} (dispo : {', '.join(envs)})")

    colonnes = [a.x, a.y] + (["ever_ate", "greediness"] if a.mange else [])
    xs, ys, ts, par_chunk = [], [], [], []
    for c in chunks:
        d = charge(data_dir, c, env, colonnes)
        if d is None or a.x not in d or a.y not in d:
            continue
        vx, vy = np.asarray(d[a.x], float), np.asarray(d[a.y], float)
        m = np.isfinite(vx) & np.isfinite(vy)
        if a.mange and "ever_ate" in d:
            m &= a_mange(d["ever_ate"], d.get("greediness"))
        if m.sum() < 3:
            continue
        xs.append(vx[m]), ys.append(vy[m])
        ts.append(np.full(int(m.sum()), c * taille))
        par_chunk.append((c * taille, *correlations(vx[m], vy[m]), int(m.sum())))
    if not xs:
        raise SystemExit("aucun genome exploitable")
    X, Y, T = np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)
    r, rho = correlations(X, Y)
    print(f"{env} : {len(X)} genomes sur {len(par_chunk)} instant(s) | "
          f"Pearson r = {r:+.3f} | Spearman rho = {rho:+.3f}")

    fig, (g, d) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                               gridspec_kw={"width_ratios": [1, 1.1]})
    sc = g.scatter(X, Y, c=T / 1e6, cmap="viridis", s=12, alpha=.65,
                   edgecolors="none")
    co = np.polyfit(X, Y, 1)
    xx = np.linspace(X.min(), X.max(), 50)
    g.plot(xx, np.polyval(co, xx), color="#C1121F", lw=1.8)
    g.set_xlabel(titre_de(a.x)), g.set_ylabel(titre_de(a.y))
    g.set_title(f"r = {r:+.2f}   ρ = {rho:+.2f}")
    g.grid(alpha=.3)
    cb = fig.colorbar(sc, ax=g)
    cb.set_label("simulation step (M)")

    pc = np.array(par_chunk, float)
    d.axhline(0, color="#8A8A8A", lw=.9)
    d.plot(pc[:, 0] / 1e6, pc[:, 1], color="#1D5C8F", lw=1.4, marker="o", ms=3,
           label="Pearson r")
    d.plot(pc[:, 0] / 1e6, pc[:, 2], color="#C1121F", lw=1.4, marker="o", ms=3,
           label="Spearman ρ")
    d.set_xlabel("simulation step (M)")
    d.set_ylabel("correlation within one instant")
    d.set_ylim(-1, 1)
    d.grid(alpha=.3)
    d.legend(frameon=False)
    d.set_title(f"{int(pc[:, 3].min())}–{int(pc[:, 3].max())} genomes per point")

    if not a.no_titre:
        fig.suptitle(f"{nom_court(env)} — {titre_de(a.y)} vs {titre_de(a.x)}",
                     fontsize=a.police + 2)
    fig.tight_layout(rect=[0, 0, 1, .92 if not a.no_titre else 1])
    out = a.out or os.path.join(a.source, "fig",
                                f"corr_{a.y}_{a.x}_{env}.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
