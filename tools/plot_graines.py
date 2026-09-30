"""Une mesure agregee sur plusieurs graines, par environnement et par instant.

    python -m simulation.tools.plot_graines A/fusion B/fusion C/fusion \\
        --chunks 100 1000 2000 3000 4000 4500 \\
        --env alone_patch8x5_s0 figurants_patch8x5_s0 \\
        --mesures age greediness mean_speed

    python -m simulation.tools.plot_graines A/fusion B/fusion C/fusion \\
        --chunks 100 1000 2000 3000 4000 4500 \\
        --env alone_blob1x40_s1 figurants_blob1x40_s1 --mesures jamais

Chaque graine donne une mediane par instant ; la courbe est la mediane de ces
medianes, la bande leur etendue. Une graine sans ce chunk est simplement
absente du calcul. La mesure `jamais` est la PART des genomes n'ayant jamais
mange, une proportion et non une mediane.

Lit les lab_data/chunk_N_pheno_<env>.npz : rien n'est rejoue.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.tools.box_chunks import (MESURES, a_mange, charge, condition,
                                         data_dir_de, geometrie, nom_court,
                                         nom_run, taille_de_chunk)

JAMAIS = ("jamais", "Share that never ate", True)
# les mesures se donnent par leur colonne brute (age) ou leur nom court (lifespan)
COURT = {v[0]: k for k, v in MESURES.items() if isinstance(v[0], str)}


def titre_de(nom):
    if nom == "jamais":
        return JAMAIS[1]
    cle = nom if nom in MESURES else COURT.get(nom)
    return MESURES[cle][1] if cle in MESURES else nom


def par_graine(data_dir, chunk, env, nom, mange):
    """Valeur d'une graine pour un instant : mediane, ou part pour `jamais`."""
    part = nom == "jamais"
    col = MESURES.get(nom, (nom,))[0]
    voulues = (["ever_ate", "greediness"] if (part or mange) else []) + \
              ([] if part else [col])
    d = charge(data_dir, chunk, env, voulues)
    if d is None:
        return np.nan, 0
    if part:
        if "ever_ate" not in d:
            return np.nan, 0
        m = a_mange(d["ever_ate"], d.get("greediness"))
        return float(1 - m.mean()), int(m.size)
    if col not in d:
        return np.nan, 0
    v = np.asarray(d[col], float)
    if mange and "ever_ate" in d:
        v = v[a_mange(d["ever_ate"], d.get("greediness"))]
    v = v[np.isfinite(v)]
    return (float(np.median(v)), int(v.size)) if v.size else (np.nan, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sources", nargs="+", metavar="DIR", help="une graine par dossier")
    p.add_argument("--chunks", type=int, nargs="+", required=True)
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("--env", nargs="+", required=True)
    p.add_argument("--mesures", nargs="+", default=["age", "greediness"],
                   help="colonnes des fichiers pheno, ou `jamais` pour la part "
                        "des genomes n'ayant jamais mange")
    p.add_argument("--mange", action="store_true",
                   help="ne garder que les genomes ayant mange au moins une fois")
    p.add_argument("--points", action="store_true", default=True,
                   help="montrer chaque graine (defaut)")
    p.add_argument("--no-points", dest="points", action="store_false")
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    dirs = []
    for s in a.sources:
        d = data_dir_de(s)
        if d is None:
            raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {s}")
        dirs.append((nom_run(s), d))
    taille = a.chunk_size or taille_de_chunk(a.sources[0])
    x = np.array([c * taille / 1e6 for c in a.chunks])

    # geometrie commune : la couleur suit la condition sociale, comme box_chunks
    par_condition = len({geometrie(e) for e in a.env}) == 1 and len(a.env) > 1
    cmap = plt.get_cmap("magma" if par_condition else "viridis")
    bornes = (.30, .72) if par_condition else (.12, .82)
    teintes = {e: cmap(t) for e, t in
               zip(a.env, np.linspace(*bornes, max(len(a.env), 2)))}
    etiq = {e: (condition(e) if par_condition else nom_court(e)) for e in a.env}

    fig, axes = plt.subplots(1, len(a.mesures),
                             figsize=(4.4 * len(a.mesures) + 1.8, 4.6),
                             squeeze=False)
    for ax, nom in zip(axes[0], a.mesures):
        titre = titre_de(nom)
        for e in a.env:
            vals = np.full((len(dirs), len(a.chunks)), np.nan)
            for i, (_, dd) in enumerate(dirs):
                for j, c in enumerate(a.chunks):
                    vals[i, j] = par_graine(dd, c, e, nom, a.mange)[0]
            n_gr = np.sum(np.isfinite(vals), axis=0)
            med = np.nanmedian(vals, axis=0)
            lo = np.nanmin(vals, axis=0)
            hi = np.nanmax(vals, axis=0)
            ok = n_gr > 0
            ax.plot(x[ok], med[ok], color=teintes[e], lw=2, marker="o", ms=5,
                    label=etiq[e], zorder=3)
            ax.fill_between(x[ok], lo[ok], hi[ok], color=teintes[e], alpha=.18,
                            lw=0, zorder=1)
            if a.points:
                for i in range(len(dirs)):
                    ax.scatter(x, vals[i], s=13, color=teintes[e], alpha=.55,
                               edgecolors="none", zorder=2)
        ax.set_title(titre, fontsize=11)
        ax.set_xlabel("simulation step (M)")
        ax.grid(alpha=.3)
        if nom == "jamais":
            ax.set_ylim(0, 1)
    axes[0][0].legend(frameon=False, fontsize=9,
                      title="social condition" if par_condition else "environment")

    if not a.no_titre:
        fig.suptitle(f"Median over {len(dirs)} seeds, band = seed range",
                     fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93 if not a.no_titre else 1])
    quoi = "_".join(a.mesures)
    geo = (next(iter({geometrie(e) for e in a.env})) if par_condition
           else "envs")
    defaut = f"graines_{quoi}_{geo}.{a.fig_format}"
    out = a.out or os.path.join(os.path.dirname(a.sources[0].rstrip("/")) or ".",
                                defaut)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{len(dirs)} graine(s) : {', '.join(n for n, _ in dirs)}")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
