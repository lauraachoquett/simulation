"""Régénère les figures d'historique à PLEINE RÉSOLUTION, hors du run.

Pendant la simulation, `simulation_data.core.plot()` trace des séries agrégées
par blocs pour que son coût reste constant. Les données brutes, elles, sont
intégralement sauvées chunk par chunk dans `<exp>/data/chunk_*.npz`. Ce script
les relit et rejoue les mêmes fonctions de tracé sans aucune réduction.

À lancer sur un frontend ou un nœud CPU, la simulation n'a pas à attendre :

    python -m simulation.tools.replot exp/2026-08-13_10-00-00
    python -m simulation.tools.replot exp/... --n-target 5000   # réduction douce
    python -m simulation.tools.replot exp/... --out fig_full    # ailleurs que fig/

Hors périmètre : les figures du lab (il faudrait rejouer les rollouts), les
métriques de poids et le TMRCA, qui dépendent de l'arbre généalogique construit
en mémoire pendant le run.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

from simulation.data_class import BASE_RESOURCES
from simulation.utils.utils_sim import load_config, load_shuffle_log
from simulation.utils.plots import (
    block_edges, block_apply, block_steps,
    plot_evolution, plot_consumption, plot_prob_eat_given_seen,
    plot_phase_portrait_png, plot_mean_movement,
    plot_lifetime_vs_step, plot_life_expectancy,
)

_NUM = re.compile(r"chunk_(\d+)\.npz$")


def load_history(data_dir):
    """Concatène les .npz de chunks dans l'ordre NUMÉRIQUE.

    Le tri lexicographique placerait chunk_100 avant chunk_9 ; les noms sont
    zero-paddés à 5 chiffres, mais on ne s'y fie pas — un run repris ou un
    renommage suffirait à casser l'ordre, et une série d'historique dans le
    désordre ne se voit pas sur la figure.
    """
    fichiers = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")),
                      key=lambda p: int(_NUM.search(os.path.basename(p)).group(1)))
    if not fichiers:
        raise SystemExit(f"aucun chunk_*.npz dans {data_dir}")

    morceaux = {}
    for f in fichiers:
        with np.load(f) as z:
            for cle in z.files:
                morceaux.setdefault(cle, []).append(z[cle])

    hist = {}
    for cle, vals in morceaux.items():
        # mean_life est (2, n_morts) : les morts s'empilent sur l'axe 1
        axe = 1 if cle == "mean_life" else 0
        hist[cle] = np.concatenate(vals, axis=axe)
    hist["_n_chunks"] = len(fichiers)
    hist["_premier"] = int(_NUM.search(os.path.basename(fichiers[0])).group(1))
    return hist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="dossier d'expérience (contenant data/ et config.json)")
    ap.add_argument("--out", default=None,
                    help="sous-dossier de sortie (défaut : fig/, écrase les figures du run)")
    ap.add_argument("--n-target", type=int, default=0,
                    help="réduire à ~N points ; 0 = pleine résolution (défaut)")
    args = ap.parse_args(argv)

    exp_dir = args.exp_dir
    data_dir = os.path.join(exp_dir, "data")
    hist = load_history(data_dir)

    cfg, _ = load_config(exp_dir)
    shuffle_log = load_shuffle_log(exp_dir)
    initial_order_ids = [r.id for r in BASE_RESOURCES]

    pop = hist["population"]
    # le premier chunk du dossier donne l'origine de l'axe (run repris ou non)
    chunk_size = len(pop) // hist["_n_chunks"]
    start_step = hist["_premier"] * chunk_size

    print(f"{hist['_n_chunks']} chunks, {len(pop)} steps, début à {start_step}")

    # Sortie : par défaut fig/, comme le run. `--out` permet de comparer côte à
    # côte les figures agrégées du run et celles à pleine résolution.
    cible = exp_dir if args.out is None else os.path.join(exp_dir, "_replot")
    if args.out is not None:
        os.makedirs(os.path.join(cible, "fig"), exist_ok=True)

    series = {k: hist[k] for k in
              ("population", "resources", "consumed", "n_seen", "n_eaten_seen")}
    steps = None
    if args.n_target:
        e = block_edges(len(pop), start_step,
                        n_target=args.n_target,
                        cut_steps=[x["step"] for x in shuffle_log])
        steps = block_steps(e, start_step)
        for k in ("population", "resources", "consumed"):
            series[k] = block_apply(series[k], e, "mean")
        for k in ("n_seen", "n_eaten_seen"):          # comptes -> somme
            series[k] = block_apply(series[k], e, "sum")
        print(f"réduit à {len(steps)} points")

    pop_s = series["population"]
    plot_evolution(pop_s, series["resources"], cible, shuffle_log,
                   initial_order_ids, start_step, steps=steps)
    plot_consumption(pop_s, series["consumed"], cible, shuffle_log,
                     initial_order_ids, start_step,
                     window=1 if args.n_target else 100,
                     name_fig="plot_conso_window_mean", steps=steps)
    plot_prob_eat_given_seen(pop_s, series["n_seen"], series["n_eaten_seen"], cible,
                             shuffle_log, initial_order_ids, start_step,
                             window=1 if args.n_target else 100, steps=steps)
    plot_phase_portrait_png(pop_s, series["resources"], cible, cfg, start_step,
                            steps=steps)

    mov = hist["mean_movement"][1500:]
    plot_mean_movement(mov, cible, start_step + 1500)

    vie = hist["mean_life"]
    plot_lifetime_vs_step(vie[1], vie[0], cible, cfg)
    plot_life_expectancy(vie[1], vie[0], cible, bin_width=1000)

    print(f"figures écrites dans {os.path.join(cible, 'fig')}")


if __name__ == "__main__":
    main()
