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


def load_history(*data_dirs, bornes=None):
    """Concatène les .npz de chunks dans l'ordre NUMÉRIQUE, sur un ou plusieurs
    dossiers.

    Le tri lexicographique placerait chunk_100 avant chunk_9 ; les noms sont
    zero-paddés à 5 chiffres, mais on ne s'y fie pas — un run repris ou un
    renommage suffirait à casser l'ordre, et une série d'historique dans le
    désordre ne se voit pas sur la figure.

    Plusieurs dossiers : un run repris vit dans un dossier neuf, et sa suite doit
    se recoller à l'originale. Un chunk présent des deux côtés n'est gardé
    QU'UNE fois, et c'est le DERNIER dossier donné qui l'emporte — d'où l'ordre
    chronologique. Sans ce dédoublonnage la période commune serait tracée deux
    fois et l'axe des pas ne correspondrait plus à rien.

    Une série absente d'un des dossiers fait échouer le chargement plutôt que de
    produire des séries de longueurs différentes, qui se décaleraient entre elles
    sans que la figure le montre.
    """
    par_chunk = {}
    for d in data_dirs:
        for f in glob.glob(os.path.join(d, "chunk_*.npz")):
            m = _NUM.search(os.path.basename(f))
            if m:
                par_chunk[int(m.group(1))] = f       # le dernier dossier gagne
    if not par_chunk:
        raise SystemExit(f"aucun chunk_*.npz dans {', '.join(data_dirs)}")

    if bornes:
        a, b = bornes
        par_chunk = {c: f for c, f in par_chunk.items() if a <= c <= b}
        if not par_chunk:
            raise SystemExit(f"aucun chunk entre {a} et {b}")
    chunks = sorted(par_chunk)
    fichiers = [par_chunk[c] for c in chunks]

    morceaux, cles_vues = {}, None
    for f in fichiers:
        with np.load(f) as z:
            if cles_vues is None:
                cles_vues = set(z.files)
            elif set(z.files) != cles_vues:
                manque = cles_vues.symmetric_difference(z.files)
                raise SystemExit(
                    f"{os.path.basename(f)} n'a pas les mêmes séries que les "
                    f"précédents ({sorted(manque)}). Les dossiers ne viennent pas "
                    "de la même version du code ; les tracer ensemble décalerait "
                    "les séries entre elles.")
            for cle in z.files:
                morceaux.setdefault(cle, []).append(z[cle])

    hist = {}
    for cle, vals in morceaux.items():
        # mean_life est (2, n_morts) : les morts s'empilent sur l'axe 1
        axe = 1 if cle == "mean_life" else 0
        hist[cle] = np.concatenate(vals, axis=axe)
    hist["_n_chunks"] = len(fichiers)
    hist["_premier"] = chunks[0]
    trous = [c for c in range(chunks[0], chunks[-1] + 1) if c not in par_chunk]
    if trous:
        print(f"[replot] {len(trous)} chunk(s) manquant(s) entre {chunks[0]} et "
              f"{chunks[-1]} : l'axe des pas les ignore, la serie est donc "
              f"comprimee a cet endroit (ex. {trous[:5]})")
    hist["_chunks"] = chunks
    return hist


def chaine_de_reprise(exp_dir):
    """[ancetres..., exp_dir] en remontant `resume_from` de config en config.

    Un run repris vit dans un dossier neuf ; sans cette remontee il faudrait
    enumerer la chaine a la main, et un maillon oublie tronque les figures sans
    que rien ne le signale. Un dossier parent disparu arrete la remontee avec un
    message plutot que de produire une serie amputee en silence.
    """
    chaine, vus = [], set()
    courant = os.path.abspath(exp_dir)
    while courant and courant not in vus:
        vus.add(courant)
        chaine.append(courant)
        try:
            cfg, _ = load_config(courant)
        except Exception:
            break
        parent = getattr(cfg, "resume_from", "")
        if not parent:
            break
        if not os.path.isdir(parent):
            print(f"[replot] parent introuvable : {parent} — la serie commence "
                  f"au chunk {getattr(cfg, 'resume_chunk', 0) + 1}")
            break
        courant = parent
    return list(reversed(chaine))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dirs", nargs="+", metavar="EXP_DIR",
                    help="dossier(s) d'expérience contenant data/ et config.json. "
                         "Plusieurs : les séries sont recollées, dans l'ORDRE "
                         "CHRONOLOGIQUE — un chunk présent des deux côtés est "
                         "pris dans le dernier dossier donné")
    ap.add_argument("--out", default=None,
                    help="dossier de sortie ; ses figures vont dans <out>/fig. "
                         "Défaut : le dossier du run, donc ses figures sont "
                         "écrasées. À donner pour une fusion, qui n'appartient "
                         "à aucun des runs")
    ap.add_argument("--no-resume", action="store_true",
                    help="ne PAS remonter la chaine des reprises ; ne tracer que "
                         "le dossier donne")
    ap.add_argument("--chunks", type=int, nargs=2, default=None,
                    metavar=("DEBUT", "FIN"),
                    help="ne tracer que cette plage de chunks (bornes incluses) ; "
                         "les figures vont dans fig_chunks_<DEBUT>_<FIN>/ sauf --out")
    ap.add_argument("--n-target", type=int, default=0,
                    help="réduire à ~N points ; 0 = pleine résolution (défaut)")
    args = ap.parse_args(argv)

    # Un seul dossier : on remonte la chaine des reprises tout seul. Plusieurs :
    # on prend exactement ce qui est donne, sans deviner.
    dossiers = (chaine_de_reprise(args.exp_dirs[0]) if len(args.exp_dirs) == 1
                and not args.no_resume else list(args.exp_dirs))
    if len(dossiers) > 1 and len(args.exp_dirs) == 1:
        print("[replot] chaine de reprise suivie : "
              + " -> ".join(os.path.basename(d.rstrip("/")) for d in dossiers))
    exp_dir = dossiers[0]
    hist = load_history(*[os.path.join(d, "data") for d in dossiers],
                        bornes=args.chunks)

    # config et journal de permutations viennent du PREMIER dossier : c'est lui
    # qui porte l'ordre initial des canaux, dont depend la lecture de tout le
    # reste. Le journal d'une reprise ne contient que ses propres permutations.
    cfg, _ = load_config(exp_dir)
    shuffle_log = load_shuffle_log(exp_dir)
    for d in dossiers[1:]:
        for e in load_shuffle_log(d):
            if not any(abs(e["step"] - v["step"]) < 1 for v in shuffle_log):
                shuffle_log.append(e)
    shuffle_log.sort(key=lambda e: e["step"])
    initial_order_ids = [r.id for r in BASE_RESOURCES]

    pop = hist["population"]
    # le premier chunk du dossier donne l'origine de l'axe (run repris ou non)
    chunk_size = len(pop) // hist["_n_chunks"]
    start_step = hist["_premier"] * chunk_size

    print(f"{hist['_n_chunks']} chunks ({hist['_chunks'][0]} a "
          f"{hist['_chunks'][-1]}), {len(pop)} steps, debut a {start_step}"
          + (f", {len(args.exp_dirs)} dossiers" if len(args.exp_dirs) > 1 else ""))

    # Sortie : par défaut le dossier du run, donc ses figures sont remplacées.
    # `--out` prend un CHEMIN et l'utilise tel quel -- il etait auparavant lu
    # puis ignore au profit d'un "_replot" en dur, ce qui rendait impossible de
    # ranger ailleurs les figures d'une fusion.
    cible = args.out
    if cible is None:
        cible = (exp_dir if not args.chunks else
                 os.path.join(os.path.abspath(args.exp_dirs[0]),
                              f"zoom_chunks_{args.chunks[0]}_{args.chunks[1]}"))
    if cible != exp_dir:
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

    # 1500 pas de chauffe : a ignorer seulement si la plage part du debut du run
    saute = 1500 if hist["_premier"] <= 1 else 0
    plot_mean_movement(hist["mean_movement"][saute:], cible, start_step + saute)

    vie = hist["mean_life"]
    plot_lifetime_vs_step(vie[1], vie[0], cible, cfg)
    plot_life_expectancy(vie[1], vie[0], cible, bin_width=1000)

    print(f"figures écrites dans {os.path.join(cible, 'fig')}")


if __name__ == "__main__":
    main()
