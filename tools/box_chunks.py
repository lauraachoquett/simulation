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
    p.add_argument("--points", action="store_true", help="superposer les genomes")
    p.add_argument("--separer", action="store_true",
                   help="separer ceux qui ont mange au moins une fois (ever_ate) "
                        "de ceux qui n'ont jamais mange")
    p.add_argument("--cmap", default="Blues",
                   help="palette des instants, du clair au fonce (defaut %(default)s)")
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
    taille = a.chunk_size
    if taille is None:      # l'axe est en pas de simulation, pas en chunks
        ici = os.path.join(a.source, "config.json")
        parent = os.path.join(os.path.dirname(os.path.abspath(a.source)),
                              "config.json")
        f_cfg = ici if os.path.exists(ici) else parent
        taille = (int(json.load(open(f_cfg)).get("chunk_size", 1000))
                  if os.path.exists(f_cfg) else 1000)

    demandees = [n for n in MESURES if getattr(a, n)]
    if a.mesures:
        inverse = {v[0]: k for k, v in MESURES.items()}
        demandees += [inverse.get(m, m) for m in a.mesures]
    if not demandees:
        demandees = ["lifespan", "motion", "wallpart"]
    demandees = list(dict.fromkeys(demandees))

    colonnes = ["ever_ate"] if a.separer else []
    for n in demandees:
        c = MESURES.get(n, (n,))[0]
        colonnes += list(c) if isinstance(c, tuple) else [c]

    par_env = {}
    for e in env:
        d_env = {}
        for c in a.chunks:
            d = charge(data_dir, c, e, colonnes)
            if d is None:
                print(f"  [info] chunk {c} absent pour {e}, ignore")
                continue
            d_env[c] = d
        if d_env:
            par_env[e] = d_env
    if not par_env:
        raise SystemExit("aucun chunk exploitable")
    chunks = sorted({c for d in par_env.values() for c in d})

    # plusieurs environnements : la couleur les distingue, et les boites se
    # decalent autour du pas. Un seul : la couleur redit l'ordre du temps.
    multi = len(par_env) > 1
    if multi:
        cmap = plt.get_cmap("viridis")
        teintes = {e: cmap(t) for e, t in
                   zip(par_env, np.linspace(.12, .82, len(par_env)))}
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
        groupes = [(next(iter(par_env)), 1., "ate at least once", -.17, .78),
                   (next(iter(par_env)), 0., "never ate", .17, .38)]
    elif multi:
        n_e = len(par_env)
        ecart = .8 / n_e
        groupes = [(e, None, e, (k - (n_e - 1) / 2) * ecart, .85)
                   for k, e in enumerate(par_env)]
    else:
        groupes = [(next(iter(par_env)), None, None, 0., .85)]

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

    for ax, nom in zip(axes[0], demandees):
        col, titre, binaire = MESURES.get(nom, (nom, nom, False))
        effectifs = []
        for gi, (env_g, trouve, etiquette, dx, alpha) in enumerate(groupes):
            vals = []
            for c in chunks:
                garde = None
                if trouve is not None and c in par_env[env_g]:
                    ea = par_env[env_g][c].get("ever_ate")
                    garde = ((np.nan_to_num(ea) > .5) if trouve
                             else (np.nan_to_num(ea) <= .5))
                vals.append(valeurs(env_g, c, col, garde))
            effectifs += [len(v) for v in vals]
            xs = np.arange(len(chunks)) + dx
            teinte = ([teintes[env_g]] * len(chunks) if multi else couleurs)
            largeur = (.8 / len(par_env) * .8 if multi
                       else .3 if a.separer else .55)
            if binaire:      # une part n'a pas de quartiles : barre de la moyenne
                moy = [float(v.mean()) if len(v) else np.nan for v in vals]
                err = [float(v.std() / max(np.sqrt(len(v)), 1)) if len(v) else np.nan
                       for v in vals]
                ax.bar(xs, moy, yerr=None if a.no_erreur else err, color=teinte,
                       width=largeur, capsize=4, edgecolor="white",
                       alpha=alpha, label=etiquette if nom == demandees[0] else None)
                ax.set_ylim(0, 1)
            else:
                bp = ax.boxplot([v if len(v) else [np.nan] for v in vals],
                                positions=xs, widths=largeur,
                                showfliers=False, patch_artist=True,
                                medianprops=dict(color="#2B2B2B", lw=2),
                                whiskerprops=dict(color="#6B6B6B"),
                                capprops=dict(color="#6B6B6B"),
                                boxprops=dict(edgecolor="#6B6B6B"))
                for corps, coul in zip(bp["boxes"], teinte):
                    corps.set_facecolor(coul), corps.set_alpha(alpha)
                if etiquette and nom == demandees[0]:
                    ax.plot([], [], lw=8, alpha=alpha, label=etiquette,
                            color=teintes[env_g] if multi else "#6B6B6B")
                if a.points:
                    for k, v in enumerate(vals):
                        ax.scatter(xs[k] + rng.uniform(-.07, .07, len(v)), v, s=8,
                                   color="#2B2B2B", alpha=.3, edgecolors="none",
                                   zorder=3)
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels([f"{c * taille / 1e6:.2f} M steps" for c in chunks],
                           rotation=20, ha="right")
        ax.set_title(titre, fontsize=11)
        ax.grid(alpha=.3, axis="y")
        quoi = "deaths" if isinstance(col, tuple) else "genomes"
        ax.set_xlabel(f"{min(effectifs)}–{max(effectifs)} {quoi}" if effectifs else "")
        if (a.separer or multi) and nom == demandees[0]:
            ax.legend(frameon=False, fontsize=9, loc="best",
                      title="test environment" if multi else None)

    quoi = " · ".join(par_env) if multi else next(iter(par_env))
    fig.suptitle(f"Comparison across simulation steps — {quoi}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    # le nom porte les mesures et les pas : plusieurs figures cohabitent
    etiq = "_".join(demandees)
    pas_txt = "_".join(f"{c * taille // 1000}k" for c in chunks)
    nom_env = "envs" if multi else next(iter(par_env))
    out = a.out or os.path.join(a.source, "fig",
                                f"box_{etiq}_{pas_txt}_{nom_env}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
