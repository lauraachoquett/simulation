"""Comparer des instants du run : une boite par chunk, pour les mesures choisies.

    python -m simulation.tools.box_chunks <dir> --chunks 50 3000 --lifespan --motion
    python -m simulation.tools.box_chunks <dir> --chunks 50 1500 3000 --mesures age greediness

Lit les lab_data/chunk_N_pheno_<env>.npz : une ligne par genome, rien n'est rejoue.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# nom court -> colonne des fichiers pheno, titre, et si la mesure est binaire
MESURES = {
    "lifespan":   ("age", "Lifespan (steps)", False),
    "motion":     ("mean_speed", "Movement / step", False),
    "intake":     ("mean_rew", "Consumption / step", False),
    "greediness": ("greediness", "Greediness  G = Cr/Tr", False),
    "energy":     ("energy_end", "Final energy", False),
    "neighbours": ("voisinage", "Neighbours in view / step", False),
    "repro":      ("repro_rate", "Reproduction rate", False),
    "explore":    ("t_explore", "Steps before first resource", False),
    "death":      ("died", "Fraction dead", True),
    "wall":       ("wall_death", "Fraction wall deaths", True),
    # part des morts dues au mur PARMI les morts : deux colonnes, cas a part
    "wallpart":   (("wall_death", "died"), "Wall deaths among deaths", True),
}

# meme vocabulaire que plot_geometries pour les legendes
NOMS = {"scatter40_s0": "scattered", "patch8x5_s0": "patchy",
        "blob1x40_s1": "single blob", "low_res": "low resources"}


# chaque env garde sa teinte ; seuls ceux qui n'ont jamais mange passent en gris
GRIS = "#9A9A9A"


def condition(env):
    """alone / clones / figurants."""
    return env.split("_", 1)[0]


def geometrie(env):
    return env.split("_", 1)[1] if "_" in env else env


def taille_de_chunk(source):
    """Pas par chunk, lu dans le config.json du run ou de son parent."""
    ici = os.path.join(source, "config.json")
    parent = os.path.join(os.path.dirname(os.path.abspath(source)), "config.json")
    f_cfg = ici if os.path.exists(ici) else parent
    return (int(json.load(open(f_cfg)).get("chunk_size", 1000))
            if os.path.exists(f_cfg) else 1000)


def nom_run(chemin):
    """Nom lisible d'une experience : <run> plutot que fusion/ ou replay/."""
    p = os.path.normpath(os.path.abspath(chemin))
    b = os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if b in (
        "fusion", "replay", "lab_data", "replay_merge") else b


def nom_court(env):
    """clones_patch8x5_s0 -> patchy : la legende ne redit pas la condition."""
    return NOMS.get(geometrie(env), geometrie(env))


def a_mange(ever_ate, greediness=None):
    """Qui a mange au moins une fois, par genome.

    En clones ever_ate est une part : "jamais" veut dire aucun clone. Et les
    pheno ecrits avant le correctif du decalage d'un pas ratent un repas pris
    au dernier pas vecu ; G > 0 le prouve, on rattrape a la lecture.
    """
    m = np.nan_to_num(np.asarray(ever_ate, float)) > 0
    if greediness is not None:
        m |= np.nan_to_num(np.asarray(greediness, float)) > 0
    return m


def data_dir_de(chemin):
    for c in (os.path.join(chemin, "replay", "lab_data"),
              os.path.join(chemin, "lab_data"), chemin):
        if glob.glob(os.path.join(c, "chunk_*_pheno_*.npz")):
            return c
    return None


def charge(data_dir, chunk, env, colonnes):
    """{colonne: valeurs} pour un chunk et un environnement."""
    f = os.path.join(data_dir, f"chunk_{chunk}_pheno_{env}.npz")
    if not os.path.exists(f):
        return None
    with np.load(f) as z:
        return {c: np.asarray(z[c], float) for c in colonnes if c in z.files}


def envs_dispo(data_dir):
    out = set()
    for f in glob.glob(os.path.join(data_dir, "chunk_*_pheno_*.npz")):
        m = re.fullmatch(r"chunk_(\d+)_pheno_(.+)\.npz", os.path.basename(f))
        if m:
            out.add(m.group(2))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--chunks", type=int, nargs="+", required=True,
                   help="instants a comparer, en chunks")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None,
                   help="pas par chunk (defaut : lu dans config.json)")
    p.add_argument("--env", nargs="+", default=None,
                   help="un ou plusieurs environnements a comparer "
                        "(defaut : alone s'il existe)")
    p.add_argument("--mesures", nargs="+", default=None,
                   help="colonnes brutes des fichiers pheno, par exemple "
                        "age greediness mean_speed")
    p.add_argument("--vs", nargs="+", default=None, metavar="DIR",
                   help="autres experiences a comparer au meme instant : la "
                        "couleur distingue alors les runs, pas les environnements")
    p.add_argument("--labels", nargs="+", default=None,
                   help="noms des runs dans la legende, avec --vs")
    p.add_argument("--titre-legende", dest="titre_legende", default=None,
                   help="en-tete de la legende (defaut : run, social condition "
                        "ou test environment selon ce qui varie)")
    p.add_argument("--points", action="store_true", help="superposer les genomes")
    p.add_argument("--modes", action="store_true",
                   help="dans l'amas unique, separer ceux qui ont mange au moins "
                        "une fois de ceux qui n'ont jamais mange")
    p.add_argument("--mange", action="store_true",
                   help="ne garder que les genomes ayant mange au moins une fois "
                        "(le groupe jamais est retire au lieu d'etre trace)")
    p.add_argument("--separer", action="store_true",
                   help="separer ceux qui ont mange au moins une fois (ever_ate) "
                        "de ceux qui n'ont jamais mange")
    p.add_argument("--cmap", default="Blues",
                   help="palette des instants, du clair au fonce (defaut %(default)s)")
    p.add_argument("--no-titre", dest="no_titre", action="store_true",
                   help="figure sans titre general, pour un article")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"],
                   help="format de la figure (defaut %(default)s)")
    p.add_argument("--no-erreur", dest="no_erreur", action="store_true",
                   help="barres de proportion sans erreur d'echantillonnage")
    p.add_argument("-o", "--out", default=None,
                   help="defaut <source>/fig/box_chunks.png")
    for nom in MESURES:
        p.add_argument(f"--{nom}", action="store_true", help=MESURES[nom][1])
    a = p.parse_args()

    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {a.source}")
    envs = envs_dispo(data_dir)
    env = a.env or [next((e for e in envs if e.startswith("alone")), envs[0])]
    print(f"environnement(s) : {', '.join(env)}   (dispo : {', '.join(envs)})")
    taille = a.chunk_size or taille_de_chunk(a.source)

    demandees = [n for n in MESURES if getattr(a, n)]
    if a.mesures:
        inverse = {v[0]: k for k, v in MESURES.items()}
        demandees += [inverse.get(m, m) for m in a.mesures]
    if not demandees:
        demandees = ["lifespan", "motion", "wallpart"]
    demandees = list(dict.fromkeys(demandees))

    colonnes = (["ever_ate", "greediness"]
                if (a.separer or a.modes or a.mange) else [])
    for n in demandees:
        c = MESURES.get(n, (n,))[0]
        colonnes += list(c) if isinstance(c, tuple) else [c]

    # comparer des runs : meme environnement, un dossier par experience
    compare_runs = bool(a.vs)
    if compare_runs:
        if len(env) > 1:
            raise SystemExit("--vs compare des runs : donner un seul --env")
        sources = [a.source] + list(a.vs)
        noms = a.labels or [nom_run(s) for s in sources]
        if len(noms) != len(sources):
            raise SystemExit("--labels doit donner un nom par experience")
        series = []
        for nom, src in zip(noms, sources):
            dd = data_dir_de(src)
            if dd is None:
                raise SystemExit(f"pas de chunk_*_pheno_*.npz sous {src}")
            series.append((nom, dd, env[0]))
    else:
        series = [(e, data_dir, e) for e in env]

    par_env, env_de = {}, {}
    for cle, dd, e in series:
        d_env = {}
        for c in a.chunks:
            d = charge(dd, c, e, colonnes)
            if d is None:
                print(f"  [info] chunk {c} absent pour {cle}, ignore")
                continue
            d_env[c] = d
        if d_env:
            par_env[cle] = d_env
            env_de[cle] = e
    if not par_env:
        raise SystemExit("aucun chunk exploitable")
    chunks = sorted({c for d in par_env.values() for c in d})

    # plusieurs environnements : la couleur les distingue, et les boites se
    # decalent autour du pas. Un seul : la couleur redit l'ordre du temps.
    multi = len(par_env) > 1
    # si la geometrie est commune, c'est la condition sociale qui varie : la
    # couleur et la legende suivent alors alone / figurants / clones
    conds = {condition(e) for e in par_env}
    geos  = {geometrie(e) for e in par_env}
    par_condition = (multi and not compare_runs
                     and len(geos) == 1 and len(conds) > 1)
    def etiq_env(e):
        return (e if compare_runs else
                condition(e) if par_condition else nom_court(e))
    if multi:
        # deux palettes distinctes : on ne confond pas une figure ou varie la
        # geometrie avec une figure ou varie la condition sociale
        cmap = plt.get_cmap("cividis" if compare_runs else
                            "magma" if par_condition else "viridis")
        bornes = ((.15, .80) if compare_runs else
                  (.30, .72) if par_condition else (.12, .82))
        teintes = {e: cmap(t) for e, t in
                   zip(par_env, np.linspace(*bornes, len(par_env)))}
    else:
        couleurs = plt.get_cmap(a.cmap)(np.linspace(.35, .85, len(chunks)))

    fig, axes = plt.subplots(1, len(demandees),
                             figsize=(4.3 * len(demandees) + 1.6, 4.8), squeeze=False)
    rng = np.random.default_rng(0)
    # deux sous-populations : celle qui a mange et l'autre. Dans un env a un seul
    # amas la distribution est bimodale, et une boite unique melange deux
    # comportements sans rapport.
    if a.separer and multi:
        raise SystemExit("--separer ne se combine pas avec plusieurs environnements")
    if a.separer:
        groupes = [(next(iter(par_env)), 1., "ate at least once", -.17, .78, None, None),
                   (next(iter(par_env)), 0., "never ate", .17, .38, None, None)]
    else:
        # --modes : seul l'amas unique porte deux comportements, les autres
        # geometries n'en ont qu'un et gardent une boite
        entrees = []
        for e in par_env:
            if a.mange:          # un seul groupe, restreint a ceux qui ont mange
                entrees.append((e, 1., etiq_env(e) if (multi or a.modes) else None,
                                None, None))
            elif a.modes and "blob" in env_de[e]:
                # a geometrie commune la couleur dit la condition : le mode se
                # lit alors a la hachure, pas au gris
                # gris tant qu'une seule serie varie ; sinon la couleur sert
                # deja a la distinguer et le mode se lit a la hachure
                distinct = par_condition or compare_runs
                entrees += [(e, 1., f"{etiq_env(e)}, ate", None, None),
                            (e, 0., f"{etiq_env(e)}, never ate",
                             None if distinct else GRIS,
                             "///" if distinct else None)]
            else:
                entrees.append((e, None, etiq_env(e) if (multi or a.modes)
                                else None, None, None))
        n_g = len(entrees)
        ecart = .8 / n_g
        groupes = [(e, t, lab, (k - (n_g - 1) / 2) * ecart if n_g > 1 else 0., .85,
                    coul if coul is not None else (teintes[e] if multi else None),
                    hach)
                   for k, (e, t, lab, coul, hach) in enumerate(entrees)]

    def valeurs(e, c, col, garde):
        """Valeurs d'un chunk pour une colonne, ou proportion si col est un couple."""
        d = par_env[e].get(c)
        if d is None:
            return np.array([])
        m = np.ones(len(next(iter(d.values()))), bool) if garde is None else garde
        if isinstance(col, tuple):
            mur, morts = col
            nm = float(np.nansum(d.get(morts, np.zeros(1))[m]))
            nw = float(np.nansum(d.get(mur, np.zeros(1))[m]))
            if nm == 0:
                return np.array([])
            return np.repeat([1., 0.], [int(round(nw)), int(round(nm - nw))])
        v = d.get(col, np.array([]))
        return v[m][np.isfinite(v[m])] if len(v) else v

    proxies, etiquettes = [], []       # legende : un aplat par groupe
    for ax, nom in zip(axes[0], demandees):
        col, titre, binaire = MESURES.get(nom, (nom, nom, False))
        effectifs = []
        for gi, (env_g, trouve, etiquette, dx, alpha, coul, hach) in enumerate(groupes):
            vals = []
            for c in chunks:
                garde = None
                if trouve is not None and c in par_env[env_g]:
                    ea = par_env[env_g][c].get("ever_ate")
                    m = a_mange(ea, par_env[env_g][c].get("greediness"))
                    garde = m if trouve else ~m
                vals.append(valeurs(env_g, c, col, garde))
            effectifs += [len(v) for v in vals if len(v)]
            if etiquette and nom == demandees[0] and any(len(v) for v in vals):
                proxies.append(Patch(facecolor=coul if coul is not None else "#6B6B6B",
                                     alpha=alpha, hatch=hach,
                                     edgecolor="#4A4A4A" if hach else "none"))
                etiquettes.append(etiquette)
            xs = np.arange(len(chunks)) + dx
            teinte = [coul] * len(chunks) if coul is not None else couleurs
            largeur = (.3 if a.separer else
                       .8 / len(groupes) * .8 if len(groupes) > 1 else .55)
            if binaire:      # une part n'a pas de quartiles : barre de la moyenne
                moy = [float(v.mean()) if len(v) else np.nan for v in vals]
                err = [float(v.std() / max(np.sqrt(len(v)), 1)) if len(v) else np.nan
                       for v in vals]
                ax.bar(xs, moy, yerr=None if a.no_erreur else err, color=teinte,
                       width=largeur, capsize=4, edgecolor="white", alpha=alpha)
                ax.set_ylim(0, 1)
            else:
                bp = ax.boxplot([v if len(v) else [np.nan] for v in vals],
                                positions=xs, widths=largeur,
                                showfliers=False, patch_artist=True,
                                medianprops=dict(color="#2B2B2B", lw=2),
                                whiskerprops=dict(color="#6B6B6B"),
                                capprops=dict(color="#6B6B6B"),
                                boxprops=dict(edgecolor="#6B6B6B"))
                for corps, c_b in zip(bp["boxes"], teinte):
                    corps.set_facecolor(c_b), corps.set_alpha(alpha)
                    if hach:
                        corps.set_hatch(hach), corps.set_edgecolor("#4A4A4A")

                if a.points:
                    for k, v in enumerate(vals):
                        ax.scatter(xs[k] + rng.uniform(-.07, .07, len(v)), v, s=8,
                                   color="#2B2B2B", alpha=.3, edgecolors="none",
                                   zorder=3)
        for k in range(len(chunks) - 1):   # separer les instants
            ax.axvline(k + .5, color="#BBBBBB", lw=.8, ls=":", zorder=0)
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels([f"{c * taille / 1e6:.2f} M steps" for c in chunks],
                           rotation=20, ha="right")
        ax.set_title(titre, fontsize=11)
        ax.grid(alpha=.3, axis="y")
        quoi = "deaths" if isinstance(col, tuple) else "genomes"
        ax.set_xlabel(f"{min(effectifs)}–{max(effectifs)} {quoi}" if effectifs else "")

    # legende hors des axes : elle ne recouvre aucune boite
    if proxies:
        fig.legend(proxies, etiquettes, frameon=False, fontsize=9,
                   loc="center left", bbox_to_anchor=(1., .5),
                   title=(a.titre_legende or
                          ("run" if compare_runs else
                           "social condition" if par_condition else
                           "test environment" if multi else None)))
    if not a.no_titre:
        quoi = " · ".join(par_env) if multi else next(iter(par_env))
        fig.suptitle(f"Comparison across simulation steps — {quoi}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94 if not a.no_titre else 1])
    # le nom porte les mesures et les pas : plusieurs figures cohabitent
    etiq = "_".join(demandees)
    pas_txt = "_".join(f"{c * taille // 1000}k" for c in chunks)
    nom_env = (f"{env[0]}_runs" if compare_runs else
               f"{next(iter(geos))}_conditions" if par_condition else
               f"{next(iter(conds))}_envs" if multi and len(conds) == 1 else
               "envs" if multi else next(iter(par_env)))
    if a.modes:
        nom_env += "_modes"
    if a.mange:
        nom_env += "_ate"
    out = a.out or os.path.join(a.source, "fig",
                                f"box_{etiq}_{pas_txt}_{nom_env}.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
