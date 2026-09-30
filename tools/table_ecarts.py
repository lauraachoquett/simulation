"""Tableau d'ecart d'une mesure entre deux experiences, par environnement.

    python -m simulation.tools.table_ecarts A/fusion B/fusion \\
        --labels classic random_offspring --chunks 100 1000 2000 4000 \\
        --env alone_scatter40_s0 alone_patch8x5_s0 alone_blob1x40_s1

Lit les lab_data/chunk_N_pheno_<env>.npz : une ligne par genome, rien n'est
rejoue. Ecart = B - A, avec son erreur type sqrt(sA^2/nA + sB^2/nB).
"""
import argparse
import os

import numpy as np

from simulation.tools.box_chunks import (MESURES, a_mange, charge, data_dir_de,
                                         nom_court, nom_run, taille_de_chunk)


def valeurs(data_dir, chunk, env, col, mange):
    """Valeurs finies d'une colonne, eventuellement restreintes a ceux qui ont mange."""
    voulues = [col] + (["ever_ate", "greediness"] if mange else [])
    d = charge(data_dir, chunk, env, voulues)
    if d is None or col not in d:
        return np.array([])
    v = np.asarray(d[col], float)
    if mange and "ever_ate" in d:
        v = v[a_mange(d["ever_ate"], d.get("greediness"))]
    return v[np.isfinite(v)]


def ligne(n1, m1, s1, n2, m2, s2):
    delta = m2 - m1
    err = np.sqrt(s1 ** 2 / max(n1, 1) + s2 ** 2 / max(n2, 1))
    return delta, err


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sources", nargs=2, metavar="DIR",
                   help="les deux experiences, dans l'ordre reference puis test")
    p.add_argument("--labels", nargs=2, default=None)
    p.add_argument("--env", nargs="+", required=True)
    p.add_argument("--chunks", type=int, nargs="+", required=True)
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("--mesure", default="greediness",
                   help="colonne des fichiers pheno (defaut %(default)s)")
    p.add_argument("--mange", action="store_true",
                   help="ne garder que les genomes ayant mange au moins une fois")
    p.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    p.add_argument("--decimales", type=int, default=3)
    a = p.parse_args()

    col = MESURES.get(a.mesure, (a.mesure,))[0]
    if isinstance(col, tuple):
        raise SystemExit(f"{a.mesure} est une proportion, pas une mesure par genome")
    dirs = [data_dir_de(s) for s in a.sources]
    for s, d in zip(a.sources, dirs):
        if d is None:
            raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {s}")
    noms = a.labels or [nom_run(s) for s in a.sources]
    taille = a.chunk_size or taille_de_chunk(a.sources[0])
    d = a.decimales

    lignes = []
    for env in a.env:
        for c in a.chunks:
            v = [valeurs(dd, c, env, col, a.mange) for dd in dirs]
            if not all(len(x) for x in v):
                manquants = [n for n, x in zip(noms, v) if not len(x)]
                print(f"  [info] {env} chunk {c} : rien pour {', '.join(manquants)}")
                continue
            n = [len(x) for x in v]
            m = [float(np.mean(x)) for x in v]
            s = [float(np.std(x, ddof=1)) if len(x) > 1 else 0. for x in v]
            delta, err = ligne(n[0], m[0], s[0], n[1], m[1], s[1])
            rel = 100 * delta / m[0] if m[0] else float("nan")
            lignes.append((nom_court(env), c * taille / 1e6,
                           m[0], s[0], n[0], m[1], s[1], n[1], delta, err, rel))

    if not lignes:
        raise SystemExit("aucune cellule exploitable")

    titre = MESURES.get(a.mesure, (None, a.mesure))[1]
    entete = ["Environment", "M steps", f"{noms[0]}", f"{noms[1]}",
              f"$\\Delta$ ({noms[1]} $-$ {noms[0]})", "%"]
    def cellule(l):
        return [l[0], f"{l[1]:.2f}",
                f"{l[2]:.{d}f} ± {l[3]:.{d}f} (n={l[4]})",
                f"{l[5]:.{d}f} ± {l[6]:.{d}f} (n={l[7]})",
                f"{l[8]:+.{d}f} ± {l[9]:.{d}f}",
                f"{l[10]:+.0f}"]

    if a.format == "markdown":
        entete = [e.replace("$\\Delta$", "Δ").replace(" $-$ ", " − ") for e in entete]
        print(f"\n{titre}" + (", genomes having eaten at least once"
                              if a.mange else ""))
        print("| " + " | ".join(entete) + " |")
        print("|" + "|".join(["---"] * len(entete)) + "|")
        for l in lignes:
            print("| " + " | ".join(cellule(l)) + " |")
    else:
        print("\\begin{tabular}{llrrrr}")
        print("\\toprule")
        # sans echappement, le % du dernier en-tete commenterait la ligne
        print(" & ".join(e.replace("%", "\\%") for e in entete) + " \\\\")
        print("\\midrule")
        for l in lignes:
            print(" & ".join(c.replace("±", "$\\pm$") for c in cellule(l)) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
