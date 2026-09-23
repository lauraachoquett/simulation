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

import numpy as np

from simulation.genealogy.mrca import plot_tmrca_gen

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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dir")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("-o", "--out", default=None,
                   help="defaut <exp_dir>/fig/tmrca_gen.png")
    a = p.parse_args()

    cfg = {}
    f = os.path.join(a.exp_dir, "config.json")
    if os.path.exists(f):
        cfg = json.load(open(f))
    taille = a.chunk_size or int(cfg.get("chunk_size", 1000))

    ev = evenements(a.exp_dir)
    if not ev:
        raise SystemExit(f"{a.exp_dir} : pas de lod/lignee.jsonl. Ce fichier est "
                         "ecrit par un run lance avec --lod.")
    gen = generations(a.exp_dir)

    chunks = sorted(int(m.group(1)) for f in
                    glob.glob(os.path.join(a.exp_dir, "data", "chunk_*.npz"))
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

    print(f"{len(ev)} fixation(s), {n} chunk(s), premier MRCA au pas {ev[0][1]}")
    out = a.out or os.path.join(a.exp_dir, "fig", "tmrca_gen.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    chemin = plot_tmrca_gen(np.zeros(int(x[-1])), gen, os.path.dirname(
        os.path.dirname(out)) if os.path.basename(os.path.dirname(out)) == "fig"
        else a.exp_dir, t_points=x, filename=os.path.basename(out),
        tmrca_pas=tmrca_pas)
    print(f"Figure saved: {chemin}")


if __name__ == "__main__":
    main()
