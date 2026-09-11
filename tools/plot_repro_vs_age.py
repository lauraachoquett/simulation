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


def tendance(x, y, n_bins=12):
    """Mediane par casier d'age : la tendance d'ensemble, robuste aux zeros."""
    if x.size < n_bins * 3:
        return None
    bords = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bords = np.unique(bords)
    if len(bords) < 3:
        return None
    idx = np.clip(np.digitize(x, bords) - 1, 0, len(bords) - 2)
    cx, cy = [], []
    for b in range(len(bords) - 1):
        m = idx == b
        if m.sum() >= 5:
            cx.append(x[m].mean())
            cy.append(np.median(y[m]))
    return (np.array(cx), np.array(cy)) if cx else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier de fusion, d'experience, ou lab_data")
    p.add_argument("-o", "--out", default=None,
                   help="fichier de sortie (defaut <source>/fig/repro_vs_age.png)")
    p.add_argument("--jitter", type=float, default=0.16,
                   help="decalage vertical aleatoire (defaut %(default)s). "
                        "repro_ready est ENTIER : sans lui les points s'empilent "
                        "en lignes et les densites sont invisibles. 0 pour couper")
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

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    norm = mcolors.Normalize(vmin=step.min(), vmax=step.max())
    # le decalage ne touche QUE l'affichage : la tendance plus bas est calculee
    # sur les valeurs exactes
    y = repro + (np.random.default_rng(0).uniform(-a.jitter, a.jitter, len(repro))
                 if a.jitter else 0.0)
    sc = ax.scatter(age, y, c=step, cmap="viridis", norm=norm, s=26,
                    alpha=.72, edgecolor="none", zorder=3)

    t = tendance(age, repro)
    if t is not None:
        ax.plot(t[0], t[1], color="#C1121F", lw=2.0, marker="o", ms=4,
                zorder=4, label="median by age bin")
        ax.legend(loc="upper left", fontsize=9, frameon=False)

    barre = fig.colorbar(sc, ax=ax, pad=.02)
    barre.set_label("Simulation step", fontsize=10)

    ax.set_xlabel("Lifespan in the lab (steps)")
    ax.set_ylabel("Potential offspring"
                  + (f"  (jittered \u00b1{a.jitter:g} for readability)"
                     if a.jitter else ""))
    ax.grid(alpha=.3, zorder=0)
    ax.set_title("Potential offspring against lifespan\n"
                 f"{len(age)} genomes — {nuls} with none "
                 f"({100*nuls/len(age):.0f} %)", fontsize=12)

    fig.tight_layout()
    sortie = a.out or os.path.join(a.source, "fig", "repro_vs_age.png")
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {sortie}")


if __name__ == "__main__":
    main()
