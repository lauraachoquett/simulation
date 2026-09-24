"""TMRCA en pas de simulation, reconstruit sans relancer le run.

    python -m simulation.tools.replot_tmrca <exp_dir>

`lod/lignee.jsonl` porte, a chaque fixation, le pas courant et le MRCA avec sa
date de naissance : l'age du MRCA se recalcule donc apres coup. Les generations
viennent de data/tmrca.npz quand il existe.
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

from simulation.genealogy.mrca import plot_tmrca_gen
from simulation.utils.utils_sim import load_shuffle_log

_NUM = re.compile(r"chunk_(\d+)\.npz$")


def evenements(exp_dir):
    """[(step, born du MRCA)] par ordre chronologique."""
    f = os.path.join(exp_dir, "lod", "lignee.jsonl")
    if not os.path.exists(f):
        return []
    out = []
    for ligne in open(f):
        if ligne.strip():
            rec = json.loads(ligne)
            out.append((int(rec["step"]), int(rec["mrca"][1])))
    return sorted(out)


def generations(exp_dir):
    for f in (os.path.join(exp_dir, "data", "tmrca.npz"),
              os.path.join(exp_dir, "tmrca.npz")):
        if os.path.exists(f):
            # les runs d'avant portent des None : numpy les a ecrits en tableau
            # d'objets, illisible sans allow_pickle
            with np.load(f, allow_pickle=True) as z:
                brut = z["tmrca"]
            return np.array([np.nan if v is None else float(v) for v in brut],
                            dtype=float)
    return None


def serie(exp_dir, taille):
    """(x en pas, TMRCA en pas, TMRCA en generations) au rythme des chunks."""
    ev = evenements(exp_dir)
    if not ev:
        raise SystemExit(f"{exp_dir} : pas de lod/lignee.jsonl. Ce fichier est "
                         "ecrit par un run lance avec --lod.")
    gen = generations(exp_dir)
    chunks = sorted(int(m.group(1)) for f in
                    glob.glob(os.path.join(exp_dir, "data", "chunk_*.npz"))
                    for m in [_NUM.search(os.path.basename(f))] if m)
    n = (len(gen) if gen is not None
         else (max(chunks) + 1 if chunks else ev[-1][0] // taille + 1))
    x = (np.arange(n) + 1) * taille
    # entre deux fixations le MRCA ne change pas : son age croit d'un pas par pas
    pas_ev = np.array([s for s, _ in ev])
    born_ev = np.array([b for _, b in ev])
    i = np.searchsorted(pas_ev, x, side="right") - 1
    tmrca_pas = np.where(i >= 0, x - born_ev[np.clip(i, 0, None)], np.nan)
    if gen is None:
        gen = np.full(n, np.nan)
    elif len(gen) != n:
        gen = np.resize(gen.astype(float), n)
    return x, tmrca_pas, gen, len(ev)


def boite(series, noms, out, titre, unite="steps"):
    """Un box plot par experience : distribution du TMRCA au fil des chunks."""
    fig, ax = plt.subplots(figsize=(1.9 * len(series) + 3.2, 5.2))
    donnees = [v[np.isfinite(v)] for v in series]
    if not any(len(v) for v in donnees):
        print(f"  [info] aucune valeur en {unite}, figure sautee")
        plt.close(fig)
        return
    bp = ax.boxplot(donnees, labels=noms, showfliers=False, widths=.55,
                    medianprops=dict(color="#C1121F", lw=2),
                    boxprops=dict(color="#4C4C4C"),
                    whiskerprops=dict(color="#4C4C4C"),
                    capprops=dict(color="#4C4C4C"))
    rng = np.random.default_rng(0)
    for k, v in enumerate(donnees, start=1):     # nuage : la boite seule cache n
        ax.scatter(k + rng.uniform(-.13, .13, len(v)), v, s=7, alpha=.25,
                   color="#1D5C8F", zorder=1)
    ax.set_ylabel(f"TMRCA ({unite})")
    ax.set_title(titre, fontsize=12)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dir", nargs="+", metavar="EXP_DIR")
    p.add_argument("--box", action="store_true",
                   help="box plot du TMRCA en pas, un par experience")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("-o", "--out", default=None,
                   help="defaut <exp_dir>/fig/tmrca_gen.png")
    a = p.parse_args()

    series, gens, noms = [], [], a.labels or [
        os.path.basename(os.path.normpath(d)) for d in a.exp_dir]
    for d in a.exp_dir:
        cfg = {}
        f = os.path.join(d, "config.json")
        if os.path.exists(f):
            cfg = json.load(open(f))
        taille = a.chunk_size or int(cfg.get("chunk_size", 1000))
        x, tmrca_pas, gen, n_ev = serie(d, taille)
        fini = np.isfinite(tmrca_pas)
        print(f"{os.path.basename(os.path.normpath(d))} : {n_ev} fixation(s), "
              f"{len(x)} chunk(s), TMRCA median "
              f"{np.nanmedian(tmrca_pas[fini]) if fini.any() else float('nan'):.0f} pas")
        series.append(tmrca_pas)
        gens.append(gen)
        if not a.box:
            out = a.out or os.path.join(d, "fig", "tmrca_gen.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            # seulement les permutations qui changent VRAIMENT l'ordre
            journal, perms, prec = load_shuffle_log(d), [], None
            for e in journal:
                if prec is None or e["order_ids"] != prec:
                    perms.append(e["step"])
                prec = e["order_ids"]
            rep = ([int(cfg.get("resume_chunk", 0)) * taille]
                   if cfg.get("resume_from") else None)
            plot_tmrca_gen(np.zeros(int(x[-1])), gen, d, t_points=x,
                           filename=os.path.basename(out), tmrca_pas=tmrca_pas,
                           permutations=perms, reprises=rep)
            print(f"Figure saved: {out}")

    if a.box:
        out = a.out or os.path.join(a.exp_dir[0], "fig", "tmrca_box.png")
        boite(series, noms, out,
              "Age of the MRCA over the run (one value per chunk)")
        racine, ext = os.path.splitext(out)
        boite(gens, noms, f"{racine}_generations{ext}",
              "Depth of the MRCA over the run (one value per chunk)",
              unite="generations")


if __name__ == "__main__":
    main()
