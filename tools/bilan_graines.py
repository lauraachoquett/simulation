"""Bilan chiffre d'un groupe d'experiences, une ligne par graine.

    python -m simulation.tools.bilan_graines \\
        --groupe classic exp/A exp/B exp/C \\
        --groupe model_v2 exp/D exp/E exp/F -o bilan.json

Lit data/chunk_*.npz et data/tmrca.npz : rien n'est rejoue.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ACTIONS = ["stay", "rotate left", "rotate right", "move forward"]
_NUM = re.compile(r"chunk_(\d+)\.npz$")


def series(dossiers, depuis, taille):
    """Concatene les series de la chaine, et le premier indice au-dela de `depuis`.

    Un chunk present dans deux dossiers n'est lu qu'une fois, le dernier donne
    l'emportant : une reprise remplace la fin du parent.
    """
    par_chunk = {}
    for d in dossiers:
        for f in glob.glob(os.path.join(d, "data", "chunk_*.npz")):
            par_chunk[int(_NUM.search(os.path.basename(f)).group(1))] = f
    fichiers = [par_chunk[c] for c in sorted(par_chunk)]
    if not fichiers:
        return None
    premier = min(par_chunk)
    parts = {}
    vies = []
    for f in fichiers:
        with np.load(f) as z:
            for c in ("population", "resources", "action_frac"):
                if c in z.files:
                    parts.setdefault(c, []).append(np.asarray(z[c]))
            if "mean_life" in z.files:
                vies.append(np.asarray(z["mean_life"]))
    out = {c: np.concatenate(v) for c, v in parts.items()}
    out["_debut"] = premier * taille
    out["_vies"] = (np.concatenate(vies, axis=1) if vies else np.zeros((2, 0)))
    out["_i0"] = max(0, depuis - out["_debut"])
    return out


def bilan_run(exp_dir, depuis, chaine=True):
    cfg = {}
    f = os.path.join(exp_dir, "config.json")
    if os.path.exists(f):
        cfg = json.load(open(f))
    taille = int(cfg.get("chunk_size", 1000))
    dossiers = [exp_dir]
    if chaine and cfg.get("resume_from"):
        from simulation.tools.replot import chaine_de_reprise
        dossiers = chaine_de_reprise(exp_dir)
    s = series(dossiers, depuis, taille)
    if s is None:
        return {"nom": os.path.basename(os.path.normpath(exp_dir)),
                "erreur": "pas de data/chunk_*.npz"}

    pop = np.asarray(s["population"], float)
    i0 = min(s["_i0"], len(pop) - 1)
    res = np.asarray(s.get("resources", np.zeros((len(pop), 1))), float)
    res_tot = res.sum(axis=1) if res.ndim > 1 else res
    stable = pop[i0:]

    # fin du run : population nulle, ou grille videe alors qu'il reste des agents
    fin_pop, fin_res = float(pop[-1]), float(res_tot[-1])
    seuil = .05 * float(res_tot.max()) if res_tot.max() > 0 else 0
    fin = ("extinction" if fin_pop <= 0 else
           "depletion" if fin_res <= seuil else "aucune")

    d = {
        "nom": os.path.basename(os.path.normpath(exp_dir)),
        "chemin": os.path.abspath(exp_dir),
        "pas_total": int(s["_debut"] + len(pop)),
        "duree_millions_de_pas": round((s["_debut"] + len(pop)) / 1e6, 3),
        "dossiers": [os.path.basename(os.path.normpath(x)) for x in dossiers],
        "population_stabilisee": float(np.mean(stable)),
        "population_stabilisee_ecart_type": float(np.std(stable)),
        "population_finale": fin_pop,
        "ressources_finales": fin_res,
        "fin": fin,
    }

    if "action_frac" in s:
        af = np.asarray(s["action_frac"], float)[i0:]
        moy = np.nanmean(af, axis=0)
        d["actions"] = {ACTIONS[k] if k < len(ACTIONS) else f"action_{k}": float(v)
                        for k, v in enumerate(moy)}
        d["part_stay"] = float(moy[0])

    vies = s["_vies"]
    if vies.size:
        d["age_max"] = float(np.max(vies[0]))
        d["age_moyen"] = float(np.mean(vies[0]))

    for c in (os.path.join(exp_dir, "data", "tmrca.npz"),
              os.path.join(exp_dir, "tmrca.npz")):
        if os.path.exists(c):
            try:    # les vieux tmrca.npz sont des tableaux d'objets
                with np.load(c, allow_pickle=True) as z:
                    brut = z["tmrca"]
                g = np.array([np.nan if v is None else float(v) for v in brut])
            except Exception as e:
                print(f"    [info] {os.path.basename(c)} illisible : {e}")
                break
            if np.isfinite(g).any():
                d["generations_jusqu_au_mrca_moyen"] = float(np.nanmean(g))
                d["generations_jusqu_au_mrca_median"] = float(np.nanmedian(g))
            break
    return d


def agrege(runs):
    """Moyenne et ecart-type entre graines, et part des fins de chaque type."""
    ok = [r for r in runs if "erreur" not in r]
    out = {"n_graines": len(ok)}
    for cle in ("duree_millions_de_pas", "population_stabilisee", "part_stay",
                "age_max", "generations_jusqu_au_mrca_moyen"):
        v = [r[cle] for r in ok if cle in r]
        if v:
            out[cle] = {"moyenne": float(np.mean(v)), "ecart_type": float(np.std(v)),
                        "valeurs": [float(x) for x in v]}
    fins = [r.get("fin") for r in ok]
    out["fins"] = {t: {"n": fins.count(t), "part": fins.count(t) / len(fins)}
                   for t in ("extinction", "depletion", "aucune") if fins.count(t)}
    act = [r["actions"] for r in ok if "actions" in r]
    if act:
        out["actions_moyennes"] = {k: float(np.mean([a[k] for a in act]))
                                   for k in act[0]}
    return out


def figure(bilan, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groupes = list(bilan["groupes"])
    mesures = [("duree_millions_de_pas", "Run length (M steps)"),
               ("population_stabilisee", "Stabilised population"),
               ("part_stay", "Share of 'stay still'"),
               ("age_max", "Oldest agent (steps)")]
    fig, axes = plt.subplots(1, len(mesures), figsize=(4.4 * len(mesures), 4.4))
    couleurs = plt.get_cmap("viridis")(np.linspace(.15, .8, len(groupes)))
    for ax, (cle, titre) in zip(axes, mesures):
        for k, g in enumerate(groupes):
            vals = [r[cle] for r in bilan["groupes"][g]["runs"] if cle in r]
            if not vals:
                continue
            ax.scatter([k] * len(vals), vals, s=70, color=couleurs[k], zorder=3,
                       edgecolor="white")
            ax.hlines(np.mean(vals), k - .25, k + .25, color=couleurs[k], lw=2.4)
        ax.set_xticks(range(len(groupes)))
        ax.set_xticklabels(groupes, rotation=30, ha="right", fontsize=9)
        ax.set_title(titre, fontsize=11)
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("One value per seed, bar = group mean", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(chemin, dpi=150)
    print(f"Figure saved: {chemin}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--groupe", action="append", nargs="+", required=True,
                   metavar=("NOM", "DIR"),
                   help="nom du groupe puis ses dossiers ; repetable")
    p.add_argument("--depuis", type=int, default=500_000,
                   help="pas a partir duquel la population est dite stabilisee "
                        "(defaut %(default)s)")
    p.add_argument("--no-chaine", dest="no_chaine", action="store_true",
                   help="ne pas remonter la chaine des reprises")
    p.add_argument("--figure", action="store_true", help="tracer le bilan")
    p.add_argument("-o", "--out", default="bilan_graines.json")
    a = p.parse_args()

    bilan = {"depuis": a.depuis, "groupes": {}}
    for entree in a.groupe:
        nom, dossiers = entree[0], entree[1:]
        runs = []
        for d in dossiers:
            r = bilan_run(d, a.depuis, chaine=not a.no_chaine)
            runs.append(r)
            if "erreur" in r:
                print(f"  {r['nom']} : {r['erreur']}")
            else:
                print(f"  {r['nom']:<24} {r['duree_millions_de_pas']:5.2f} M pas "
                      f"| pop {r['population_stabilisee']:7.1f} "
                      f"| stay {r.get('part_stay', float('nan')):.3f} "
                      f"| age max {r.get('age_max', float('nan')):.0f} "
                      f"| fin {r['fin']}")
        bilan["groupes"][nom] = {"runs": runs, "resume": agrege(runs)}
        print(f"{nom} : {bilan['groupes'][nom]['resume']['fins']}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(bilan, f, indent=2)
    print(f"Bilan ecrit dans {a.out}")
    if a.figure:
        figure(bilan, os.path.splitext(a.out)[0] + ".png")


if __name__ == "__main__":
    main()
