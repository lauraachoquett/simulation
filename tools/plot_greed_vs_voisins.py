"""Greediness en fonction du nombre de voisins en vue, un point par genome.

    python -m simulation.tools.plot_greed_vs_voisins <exp_dir|fusion>
    python -m simulation.tools.plot_greed_vs_voisins <dir> --envs 'figurants*' --chunks 3000 3200

Un point = un genome dans un env social ; le voisinage et la greediness y sont
moyennes sur son rollout. Les env `alone` n'ont pas de voisin, ils sont ignores.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.tools.pca_phenotypes import charge, familles, filtre, lab_data_de


def correlation(x, y):
    """Pearson et Spearman, sans scipy."""
    if len(x) < 3:
        return np.nan, np.nan
    r = np.corrcoef(x, y)[0, 1]
    rg = lambda v: np.argsort(np.argsort(v)).astype(float)
    return r, np.corrcoef(rg(x), rg(y))[0, 1]


def medianes(x, y, n=12):
    """Mediane de y par casier de x d'effectif egal : la tendance, sans lisser."""
    q = np.quantile(x, np.linspace(0, 1, n + 1))
    q = np.unique(q)
    cx, cy = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (x >= lo) & (x <= hi if hi == q[-1] else x < hi)
        if m.sum() >= 5:
            cx.append(np.median(x[m])), cy.append(np.median(y[m]))
    return np.array(cx), np.array(cy)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier d'experience, de replay ou de fusion")
    p.add_argument("--envs", nargs="+", default=["clones*", "figurants*"],
                   help="environnements a tracer (defaut %(default)s)")
    p.add_argument("--chunks", type=int, nargs=2, default=None,
                   metavar=("DEBUT", "FIN"), help="ne garder que cette plage")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=1000)
    p.add_argument("-o", "--out", default=None,
                   help="defaut <source>/fig/greediness_vs_voisins.png")
    a = p.parse_args()

    data_dir = lab_data_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de lab_data sous {a.source}")
    fichiers = filtre(familles(data_dir), a.envs)
    if a.chunks:
        fichiers = [t for t in fichiers if a.chunks[0] <= t[1] <= a.chunks[1]]
    if not fichiers:
        raise SystemExit(f"aucun phenotype {a.envs} dans {data_dir}")

    envs = sorted({e for e, _, _ in fichiers})
    fig, axes = plt.subplots(1, len(envs), figsize=(6.4 * len(envs), 5.4),
                             squeeze=False)
    for ax, env in zip(axes[0], envs):
        X, chunk, _, _, colonnes, _ = charge(
            [t for t in fichiers if t[0] == env], ["voisinage", "greediness"])
        if colonnes != ["voisinage", "greediness"]:
            raise SystemExit(f"{env} : colonnes manquantes ({colonnes})")
        v, g = X[:, 0], X[:, 1]
        ok = np.isfinite(v) & np.isfinite(g)
        v, g, pas = v[ok], g[ok], chunk[ok] * a.chunk_size

        sc = ax.scatter(v, g, c=pas, cmap="viridis", s=12, alpha=.55,
                        edgecolor="none")
        cx, cy = medianes(v, g)
        if len(cx) > 1:
            ax.plot(cx, cy, color="black", lw=2, marker="o", ms=4,
                    label="median per bin")
            ax.legend(frameon=False, fontsize=9)
        r, rho = correlation(v, g)
        ax.set_xlabel("Neighbours in view / step")
        ax.set_ylabel("Greediness  G = Cr/Tr")
        ax.set_title(f"{env}  —  {len(v)} genomes\n"
                     f"Pearson r = {r:.2f}, Spearman ρ = {rho:.2f}", fontsize=11)
        ax.grid(alpha=.3)
        fig.colorbar(sc, ax=ax, pad=.02).set_label("simulation step")

    fig.suptitle("Greediness against social density, one point per genome",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = a.out or os.path.join(a.source, "fig", "greediness_vs_voisins.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
