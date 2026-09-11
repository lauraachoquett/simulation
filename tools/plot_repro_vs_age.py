"""Descendance potentielle en fonction de la duree de vie, un point par genome.

    python -m simulation.tools.plot_repro_vs_age ~/fusion
    python -m simulation.tools.plot_repro_vs_age ~/fusion/lab_data -o fig/repro.png

`repro_ready` compte les fois ou le seuil de reproduction est atteint, AVANT le
plafond de places libres : c'est donc le nombre de descendants qu'un genome
aurait eus. Le tracer contre l'age separe les deux facons de l'obtenir -- mieux
se nourrir, ou simplement vivre plus longtemps -- que le total seul confond.

La couleur donne le pas de simulation : si les points recents montent au-dessus
de la tendance des anciens A AGE EGAL, c'est un gain de strategie et non de
longevite.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def lab_data_de(chemin):
    """Le dossier lab_data, qu'on donne la fusion, l'experience ou son replay."""
    for candidat in (chemin,
                     os.path.join(chemin, "lab_data"),
                     os.path.join(chemin, "replay", "lab_data")):
        if glob.glob(os.path.join(candidat, "chunk_*.npz")):
            return candidat
    return None


def charge(data_dir, chunk_size):
    """(age, repro, step) concatenes sur tous les chunks.

    Les fichiers `_lowres` sont ecartes : l'env low_res ne porte pas repro_ready
    et mesure l'exploration, pas la fecondite.
    """
    age, repro, step = [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz"))):
        base = os.path.basename(f)
        m = re.fullmatch(r"chunk_(\d+)\.npz", base)       # exclut _lowres, _adapt_*
        if not m:
            continue
        d = np.load(f)
        if "repro_ready" not in d.files:
            continue
        n = len(d["age"])
        age.append(np.asarray(d["age"], float))
        repro.append(np.asarray(d["repro_ready"], float))
        step.append(np.full(n, int(m.group(1)) * chunk_size, float))
        d.close()
    if not age:
        return None
    return (np.concatenate(age), np.concatenate(repro), np.concatenate(step))


def casiers(age, n_x=26):
    """Bords des casiers : entiers en y, reguliers en x.

    repro_ready est un COMPTE : des casiers a cheval sur les entiers
    melangeraient 1 et 2 descendants dans la meme case et liseraient la densite
    de travers.
    """
    bx = np.linspace(0, max(age.max(), 1), n_x + 1)
    return bx


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier de fusion, d'experience, ou lab_data")
    p.add_argument("-o", "--out", default=None,
                   help="fichier de sortie (defaut <source>/fig/repro_vs_age.png)")
    p.add_argument("--tranches", type=int, default=6,
                   help="nombre de vignettes, decoupees en tranches de PAS "
                        "d'effectif egal (defaut %(default)s)")
    p.add_argument("--repro-max", type=int, default=None,
                   help="dernier casier de descendance. Defaut : le quantile 99, "
                        "pas le maximum -- quelques valeurs rares etiraient l'axe "
                        "et ecrasaient la masse. Le compte du titre dit combien "
                        "sont au-dela")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="pas par chunk (defaut : lu dans config.json, sinon 1000)")
    a = p.parse_args()

    data_dir = lab_data_de(a.source)
    if data_dir is None:
        print(f"Aucun chunk_*.npz sous {a.source}")
        return

    chunk_size = a.chunk_size
    if chunk_size is None:
        for c in (os.path.join(a.source, "config.json"),
                  os.path.join(os.path.dirname(data_dir), "..", "config.json")):
            if os.path.isfile(c):
                chunk_size = json.load(open(c)).get("chunk_size")
                break
    if chunk_size is None:
        chunk_size = 1000
        print("[info] chunk_size non trouve, 1000 suppose (--chunk-size pour forcer)")

    d = charge(data_dir, chunk_size)
    if d is None:
        print(f"{data_dir} : aucun fichier ne porte repro_ready. Ces donnees "
              "sont anterieures a son ajout, il faut rejouer le lab.")
        return
    age, repro, step = d
    nuls = int((repro == 0).sum())
    print(f"{len(age)} genomes, {len(np.unique(step))} instants, "
          f"{nuls} a zero descendant ({100*nuls/len(age):.0f} %)")

    # Tranches a EFFECTIF egal et non a duree egale : les checkpoints ne sont pas
    # forcement reguliers, et une tranche vide ferait une vignette blanche.
    pas_uniques = np.unique(step)
    k = min(a.tranches, len(pas_uniques))
    bornes = np.quantile(step, np.linspace(0, 1, k + 1))
    bornes[0] -= 1                               # inclure le premier instant

    r_max = (a.repro_max if a.repro_max is not None
             else int(max(np.ceil(np.quantile(repro, 0.99)), 1)))
    hors = int((repro > r_max).sum())
    by = np.arange(-0.5, r_max + 1.5)            # un casier par entier
    bx = casiers(age)

    # Toutes les vignettes partagent l'echelle de couleur, sinon deux images
    # d'allure identique porteraient des effectifs differents. On normalise en
    # FRACTION de la tranche : les tranches n'ont pas le meme nombre de genomes.
    grilles = []
    for i in range(k):
        m = (step > bornes[i]) & (step <= bornes[i + 1])
        h, _, _ = np.histogram2d(age[m], repro[m], bins=[bx, by])
        grilles.append((h / max(m.sum(), 1), int(m.sum()),
                        step[m].min() if m.any() else 0,
                        step[m].max() if m.any() else 0))
    vmax = max(g[0].max() for g in grilles) or 1.0

    cols = min(k, 3)
    lignes = -(-k // cols)
    fig, axes = plt.subplots(lignes, cols, figsize=(4.5 * cols, 3.7 * lignes),
                             squeeze=False, sharex=True, sharey=True)
    for ax in axes.ravel()[k:]:
        ax.axis("off")

    for i, (h, n, s0, s1) in enumerate(grilles):
        ax = axes[i // cols][i % cols]
        im = ax.pcolormesh(bx, by, h.T, cmap="magma_r", vmin=0, vmax=vmax,
                           shading="flat")
        # mediane par casier d'age : la tendance, sur les valeurs exactes
        m = (step > bornes[i]) & (step <= bornes[i + 1])
        centres, med = [], []
        for j in range(len(bx) - 1):
            sel = m & (age >= bx[j]) & (age < bx[j + 1])
            if sel.sum() >= 4:
                centres.append((bx[j] + bx[j + 1]) / 2)
                med.append(np.median(repro[sel]))
        if centres:
            ax.plot(centres, med, color="#1D3557", lw=1.8, zorder=3)
        ax.set_title(f"steps {s0:,.0f}–{s1:,.0f}   ({n} genomes)", fontsize=10)
        ax.grid(alpha=.18, zorder=0)

    for ax in axes[-1]:
        ax.set_xlabel("Lifespan in the lab (steps)")
    for ligne in axes:
        ligne[0].set_ylabel("Potential offspring")

    barre = fig.colorbar(im, ax=axes, pad=.015, fraction=.025)
    barre.set_label("Fraction of the genomes in that panel", fontsize=10)

    fig.suptitle("Potential offspring against lifespan, over time\n"
                 f"{len(age)} genomes — {nuls} with none "
                 f"({100*nuls/len(age):.0f} %)"
                 + (f" — {hors} above {r_max}, off scale" if hors else "")
                 + " — shared colour scale",
                 fontsize=12.5)

    sortie = a.out or os.path.join(a.source, "fig", "repro_vs_age.png")
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)
    fig.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {sortie}")


if __name__ == "__main__":
    main()
