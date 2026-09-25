"""Comparer des instants du run : une boite par chunk, pour les mesures choisies.

    python -m simulation.tools.box_chunks <dir> --chunks 50 3000 --lifespan --motion
    python -m simulation.tools.box_chunks <dir> --chunks 50 1500 3000 --mesures age greediness

Lit les lab_data/chunk_N_pheno_<env>.npz : une ligne par genome, rien n'est rejoue.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# nom court -> colonne des fichiers pheno, titre, et si la mesure est binaire
MESURES = {
    "lifespan":   ("age", "Lifespan (steps)", False),
    "motion":     ("mean_speed", "Movement / step", False),
    "intake":     ("mean_rew", "Consumption / step", False),
    "greediness": ("greediness", "Greediness  G = Cr/Tr", False),
    "energy":     ("energy_end", "Final energy", False),
    "neighbours": ("voisinage", "Neighbours in view / step", False),
    "repro":      ("repro_rate", "Reproduction rate", False),
    "explore":    ("t_explore", "Steps before first resource", False),
    "death":      ("died", "Fraction dead", True),
    "wall":       ("wall_death", "Fraction wall deaths", True),
    # part des morts dues au mur PARMI les morts : deux colonnes, cas a part
    "wallpart":   (("wall_death", "died"), "Wall deaths among deaths", True),
}


def data_dir_de(chemin):
    for c in (os.path.join(chemin, "replay", "lab_data"),
              os.path.join(chemin, "lab_data"), chemin):
        if glob.glob(os.path.join(c, "chunk_*_pheno_*.npz")):
            return c
    return None


def charge(data_dir, chunk, env, colonnes):
    """{colonne: valeurs} pour un chunk et un environnement."""
    f = os.path.join(data_dir, f"chunk_{chunk}_pheno_{env}.npz")
    if not os.path.exists(f):
        return None
    with np.load(f) as z:
        return {c: np.asarray(z[c], float) for c in colonnes if c in z.files}


def envs_dispo(data_dir):
    out = set()
    for f in glob.glob(os.path.join(data_dir, "chunk_*_pheno_*.npz")):
        m = re.fullmatch(r"chunk_(\d+)_pheno_(.+)\.npz", os.path.basename(f))
        if m:
            out.add(m.group(2))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--chunks", type=int, nargs="+", required=True,
                   help="instants a comparer")
    p.add_argument("--env", default=None,
                   help="environnement des phenotypes (defaut : alone s'il existe)")
    p.add_argument("--mesures", nargs="+", default=None,
                   help="colonnes brutes des fichiers pheno, par exemple "
                        "age greediness mean_speed")
    p.add_argument("--points", action="store_true", help="superposer les genomes")
    p.add_argument("--cmap", default="Blues",
                   help="palette des instants, du clair au fonce (defaut %(default)s)")
    p.add_argument("--no-erreur", dest="no_erreur", action="store_true",
                   help="barres de proportion sans erreur d'echantillonnage")
    p.add_argument("-o", "--out", default=None,
                   help="defaut <source>/fig/box_chunks.png")
    for nom in MESURES:
        p.add_argument(f"--{nom}", action="store_true", help=MESURES[nom][1])
    a = p.parse_args()

    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {a.source}")
    envs = envs_dispo(data_dir)
    env = a.env or next((e for e in envs if e.startswith("alone")), envs[0])
    print(f"environnement : {env}   (dispo : {', '.join(envs)})")

    demandees = [n for n in MESURES if getattr(a, n)]
    if a.mesures:
        inverse = {v[0]: k for k, v in MESURES.items()}
        demandees += [inverse.get(m, m) for m in a.mesures]
    if not demandees:
        demandees = ["lifespan", "motion", "greediness"]
    demandees = list(dict.fromkeys(demandees))

    colonnes = []
    for n in demandees:
        c = MESURES.get(n, (n,))[0]
        colonnes += list(c) if isinstance(c, tuple) else [c]
    par_chunk = {}
    for c in a.chunks:
        d = charge(data_dir, c, env, colonnes)
        if d is None:
            print(f"  [info] chunk {c} absent pour {env}, ignore")
            continue
        par_chunk[c] = d
    if len(par_chunk) < 1:
        raise SystemExit("aucun chunk exploitable")

    chunks = sorted(par_chunk)
    # un seul ton, du clair au fonce : l'ordre des chunks est un ordre de temps,
    # une palette arc-en-ciel ne le dit pas
    couleurs = plt.get_cmap(a.cmap)(np.linspace(.35, .85, len(chunks)))
    fig, axes = plt.subplots(1, len(demandees),
                             figsize=(3.9 * len(demandees) + 1, 4.8), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, nom in zip(axes[0], demandees):
        col, titre, binaire = MESURES.get(nom, (nom, nom, False))
        if isinstance(col, tuple):      # rapport de deux colonnes, par chunk
            mur, morts = col
            vals = []
            for c in chunks:
                m = par_chunk[c].get(mur, np.array([]))
                d_ = par_chunk[c].get(morts, np.array([]))
                n_morts = float(np.nansum(d_))
                vals.append(np.array([]) if n_morts == 0 else
                            np.repeat([1., 0.], [int(round(np.nansum(m))),
                                                 int(round(n_morts - np.nansum(m)))]))
        else:
            vals = [par_chunk[c].get(col, np.array([])) for c in chunks]
        vals = [v[np.isfinite(v)] for v in vals]
        if binaire:      # une part n'a pas de quartiles : barre de la moyenne
            moy = [float(v.mean()) if len(v) else np.nan for v in vals]
            err = [float(v.std() / max(np.sqrt(len(v)), 1)) if len(v) else np.nan
                   for v in vals]
            ax.bar(range(len(chunks)), moy, yerr=None if a.no_erreur else err,
                   color=couleurs, width=.6, capsize=4, edgecolor="white")
            ax.set_ylim(0, 1)
        else:
            bp = ax.boxplot(vals, positions=range(len(chunks)), widths=.55,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="#2B2B2B", lw=2),
                            whiskerprops=dict(color="#6B6B6B"),
                            capprops=dict(color="#6B6B6B"),
                            boxprops=dict(edgecolor="#6B6B6B"))
            for corps, coul in zip(bp["boxes"], couleurs):
                corps.set_facecolor(coul), corps.set_alpha(.85)
            if a.points:
                for k, v in enumerate(vals):
                    ax.scatter(k + rng.uniform(-.14, .14, len(v)), v, s=8,
                               color="#2B2B2B", alpha=.3, edgecolors="none", zorder=3)
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels([f"chunk {c}" for c in chunks], rotation=20, ha="right")
        ax.set_title(titre, fontsize=11)
        ax.grid(alpha=.3, axis="y")
        n = [len(v) for v in vals]
        quoi = "deaths" if isinstance(col, tuple) else "genomes"
        ax.set_xlabel(f"{min(n)}–{max(n)} {quoi}" if n else "")

    fig.suptitle(f"Comparison across chunks — {env}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = a.out or os.path.join(a.source, "fig", "box_chunks.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
