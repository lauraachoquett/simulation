"""Écarte la greediness fausse des env adapt déjà sur disque.

`_per_agent_metrics` lisait `delta_energy` dans l'ordre des canaux de BASE.
L'env adapt tourne avec les canaux permutés, donc `good_channels` désignait des
canaux qui ne portaient plus good et medium : la greediness y comptait « a vu
une bonne ressource » sur les mauvais canaux. Corrigé, mais les chunks déjà
calculés gardent la mauvaise valeur.

Elle n'est PAS recalculable hors ligne : il faudrait `saw`, dérivé de `obs`, qui
n'existe que dans les sorties du rollout. `_save_lab_data` ne garde que les
métriques déjà agrégées. Seul un replay des rollouts depuis les checkpoints la
reconstruirait.

D'ici là le problème n'est pas la valeur fausse, c'est le RACCORD : sans rien
faire, la courbe colle des chunks faux à des chunks corrects et la marche qui en
résulte se lit comme un effet réel. On renomme donc la clé en `greediness_bugue`
— `_get` retourne NaN sur une clé absente, donc le début de courbe devient un
trou, et la donnée n'est pas perdue pour autant.

Seuls les env adapt sont touchés. high_res, clones et low_res tournent sur les
canaux de base : leur greediness est juste. Et dans les figures adapt, seule la
greediness l'était — durée de vie, consommation, mouvement et causes de mort ne
lisent pas delta_energy.

Prend une expérience ou une racine : tous les `lab_data/` situés dessous sont
traités, quelle que soit la profondeur.

    python -m simulation_meta.tools.fix_adapt_greediness exp/           # simulation
    python -m simulation_meta.tools.fix_adapt_greediness exp/ --apply   # tous les runs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

PREFIXE = "greediness"
NOUVEAU = "greediness_bugue"
# chunk_12_adapt_medium_to_poison_summary.json, chunk_12_adapt_rot1.npz, ...
_ADAPT = re.compile(r"chunk_\d+_adapt_.+")


def _summaries(data_dir):
    return [f for f in glob.glob(os.path.join(data_dir, "chunk_*_adapt_*_summary.json"))
            if _ADAPT.match(os.path.basename(f))]


def _archives(data_dir):
    return [f for f in glob.glob(os.path.join(data_dir, "chunk_*_adapt_*.npz"))
            if _ADAPT.match(os.path.basename(f))]


def corrige_summary(chemin, apply):
    with open(chemin) as fh:
        d = json.load(fh)
    touchees = [k for k in d if k.startswith(PREFIXE + "_")
                and not k.startswith(NOUVEAU)]
    if not touchees:
        return 0
    if apply:
        for k in touchees:
            d[NOUVEAU + k[len(PREFIXE):]] = d.pop(k)
        with open(chemin, "w") as fh:
            json.dump(d, fh, indent=2)
    return len(touchees)


def corrige_npz(chemin, apply):
    with np.load(chemin) as z:
        contenu = {k: z[k] for k in z.files}
    if PREFIXE not in contenu:
        return 0
    if apply:
        contenu[NOUVEAU] = contenu.pop(PREFIXE)
        np.savez_compressed(chemin, **contenu)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", nargs="+",
                    help="dossier(s) : une expérience, ou une racine à explorer")
    ap.add_argument("--apply", action="store_true",
                    help="écrire (sans ce drapeau : simulation seule)")
    args = ap.parse_args(argv)

    if not args.apply:
        print("SIMULATION — rien n'est écrit. Ajouter --apply pour appliquer.\n")

    # On descend chercher les lab_data/ : la profondeur varie (exp/<date>/<run>/),
    # et passer une racine doit marcher comme passer un run. Set : donner à la
    # fois un parent et son enfant ne doit pas traiter deux fois le même dossier.
    runs = sorted({os.path.dirname(os.path.realpath(os.path.join(r, d)))
                   for arg in args.exp_dir
                   for r, ds, _ in os.walk(arg)
                   for d in ds if d == "lab_data"})
    if not runs:
        print(f"Aucun lab_data/ trouvé sous {', '.join(args.exp_dir)}")
        return

    total_s = total_n = 0
    for exp in runs:
        data_dir = os.path.join(exp, "lab_data")
        sums, arch = _summaries(data_dir), _archives(data_dir)
        if not sums and not arch:
            continue                       # lab_data sans env adapt : rien à faire
        ns = sum(corrige_summary(f, args.apply) for f in sums)
        na = sum(corrige_npz(f, args.apply) for f in arch)
        total_s += ns
        total_n += na
        print(f"{exp}\n  {len(sums):4d} summary  -> {ns:4d} clés renommées"
              f"\n  {len(arch):4d} npz      -> {na:4d} tableaux renommés")

    verbe = "renommé" if args.apply else "à renommer"
    print(f"\nTotal {verbe} : {total_s} clés de summary, {total_n} tableaux npz")
    if args.apply:
        print("Relancer plot_lab_metrics (ou tools/replot.py) pour redessiner "
              "les figures : le début de la courbe de greediness sera vide.")


if __name__ == "__main__":
    main()
