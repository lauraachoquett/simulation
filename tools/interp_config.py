"""Interpole entre deux configurations, sur les parametres qu'on choisit.

    python -m simulation.tools.interp_config <cfgA> <cfgB> --steps 5
    python -m simulation.tools.interp_config <cfgA> <cfgB> --steps 7 \\
           --params prob_factor pop_res_prob

Les arguments sont des dossiers d'experience ou des config.json.

Interpolation GEOMETRIQUE et non lineaire. Ces parametres sont des echelles :
`pop_res_prob` va de 1e-6 a 5e-5, soit un facteur 50. Une interpolation lineaire
passerait 90 % de sa course au-dessus de 2.5e-5, donc pres de la borne haute,
et n'echantillonnerait presque pas le regime rare. En log, chaque pas multiplie
par un facteur constant.

Les valeurs sont rendues en surcharges de ligne de commande, a poser sur un
`--from` : rien n'est duplique, et la provenance reste lisible dans le
config.json du run.
"""
import argparse
import json
import math
import os

# parametre -> drapeau de run.py. Ceux du bloc `resources` passent par les
# options par-ressource (--pf, --prp), qui acceptent une valeur par canal.
DRAPEAUX = {
    "prob_factor":              "--pf",
    "pop_res_prob":             "--prp",
    "delta_energy":             "--de",
    "init_number_of_resources": "--init-res",
    "energy_max":               "-e",
    "energy_decay":             "-d",
    "min_energy_repr":          "--min-energy-repr",
    "starting_energy":          "--start-energy",
    "param_mutate":             "-p",
    "mutation_var":             "--mvar",
    "temperature":              "-t",
}
PAR_RESSOURCE = {"prob_factor", "pop_res_prob", "delta_energy",
                 "init_number_of_resources"}


def charge(chemin):
    """Le dict de config, qu'on donne le dossier ou le fichier."""
    if os.path.isdir(chemin):
        chemin = os.path.join(chemin, "config.json")
    with open(chemin) as f:
        return json.load(f)


def valeur(cfg, nom):
    """La valeur d'un parametre, y compris dans le bloc `resources`."""
    if nom in PAR_RESSOURCE:
        res = cfg.get("resources")
        if res:
            return [float(r[nom]) for r in res]
        return [float(cfg[nom])] if nom in cfg else None      # config a plat
    return float(cfg[nom]) if nom in cfg else None


def interpole(a, b, t):
    """Geometrique si les deux bornes sont > 0, lineaire sinon.

    Une borne nulle ou negative rend le log impossible ; on retombe alors sur
    du lineaire plutot que d'echouer, en le signalant.
    """
    if a > 0 and b > 0:
        return math.exp((1 - t) * math.log(a) + t * math.log(b))
    return (1 - t) * a + t * b


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cfg_a"); p.add_argument("cfg_b")
    p.add_argument("--steps", type=int, default=5,
                   help="nombre de configurations, bornes comprises (defaut "
                        "%(default)s)")
    p.add_argument("--params", nargs="+",
                   default=["prob_factor", "pop_res_prob", "energy_max"],
                   help="parametres a interpoler (defaut %(default)s)")
    p.add_argument("--from-dir", default=None,
                   help="dossier a passer a --from dans les commandes "
                        "(defaut : cfg_a)")
    p.add_argument("--borne", nargs=3, action="append", default=[],
                   metavar=("PARAM", "A", "B"),
                   help="remplacer les deux bornes d'un parametre. Utile quand "
                        "une valeur de config n'est pas une vraie borne : "
                        "energy_max=1e9 veut dire 'pas de plafond', et "
                        "interpoler jusque-la passerait presque toute la course "
                        "dans un regime identique. Mesurer l'energie atteignable "
                        "et borner la (ex. --borne energy_max 8 850)")
    p.add_argument("--lineaire", action="store_true",
                   help="interpoler lineairement au lieu de geometriquement")
    a = p.parse_args()

    A, B = charge(a.cfg_a), charge(a.cfg_b)
    base = a.from_dir or a.cfg_a

    plans = []
    for nom in a.params:
        va, vb = valeur(A, nom), valeur(B, nom)
        if va is None or vb is None:
            print(f"[saute] {nom} absent de "
                  f"{'A' if va is None else 'B'}")
            continue
        plans.append((nom, va, vb))

    forcees = {n: (float(x), float(y)) for n, x, y in a.borne}
    for i, (nom, va, vb) in enumerate(plans):
        if nom in forcees:
            x, y = forcees[nom]
            plans[i] = (nom, [x] * len(va) if isinstance(va, list) else x,
                        [y] * len(vb) if isinstance(vb, list) else y)
            print(f"[borne] {nom} force a {x:g} -> {y:g}")

    if not plans:
        print("Aucun parametre commun a interpoler.")
        return

    print(f"{a.steps} configurations, "
          f"{'lineaire' if a.lineaire else 'geometrique'}\n")
    print(f"  {'parametre':<26}{'A':>13}{'B':>13}   rapport")
    for nom, va, vb in plans:
        fa = va[0] if isinstance(va, list) else va
        fb = vb[0] if isinstance(vb, list) else vb
        r = fb / fa if fa else float("inf")
        print(f"  {nom:<26}{fa:>13.6g}{fb:>13.6g}   x{r:.4g}")
    print()

    for i in range(a.steps):
        t = i / (a.steps - 1) if a.steps > 1 else 0.0
        options = []
        for nom, va, vb in plans:
            if isinstance(va, list):
                vals = [interpole(x, y, t) if not a.lineaire
                        else (1 - t) * x + t * y for x, y in zip(va, vb)]
                options.append(f"{DRAPEAUX[nom]} "
                               + " ".join(f"{v:.6g}" for v in vals))
            else:
                v = (interpole(va, vb, t) if not a.lineaire
                     else (1 - t) * va + t * vb)
                fmt = f"{v:.6g}" if nom not in ("init_number_of_resources",) \
                      else f"{int(round(v))}"
                options.append(f"{DRAPEAUX[nom]} {fmt}")
        print(f"# t = {t:.3f}")
        print(f"python -m simulation.run --from {base} "
              + " ".join(options))


if __name__ == "__main__":
    main()
