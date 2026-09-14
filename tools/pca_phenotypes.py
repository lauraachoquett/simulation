"""PCA des phenotypes de lab : ou va la population dans l'espace des mesures.

    python -m simulation.tools.pca_phenotypes <exp_dir>
    python -m simulation.tools.pca_phenotypes <exp_dir> --vars age greediness voisinage

Les axes sont ajustes UNE SEULE FOIS sur TOUS les environnements et tous les
chunks empiles, puis chaque sous-ensemble y est projete. C'est ce qui rend la trajectoire lisible : avec une PCA
par chunk, les axes tourneraient d'une image a l'autre et un deplacement
apparent ne voudrait rien dire.

Un seul plan pour tous les environnements, et c'est le point : les fichiers
`chunk_N_pheno_<env>.npz` portent UNE LIGNE PAR GENOME, et la ligne b designe le
meme genome partout puisque tous les env partagent agent_params dans le meme
ordre. On peut donc demander non seulement ou se situe chaque environnement,
mais de combien un MEME genome s'y deplace -- ce qu'une PCA par environnement,
avec ses axes propres, rendrait impossible a lire.

La question visee n'est pas "quels axes de variation existent" mais "la
population revient-elle quelque part". D'ou la trajectoire du barycentre et les
permutations marquees dessus : un attracteur, c'est un retour APRES
perturbation, donc une affirmation sur la dynamique et pas sur un nuage fige.

Deux choix a connaitre.

1. Variables en TAUX, jamais les totaux. total_rew = mean_rew x (age+1),
   total_move = mean_speed x age, repro_ready = repro_rate x age : les garder
   ferait de PC1 un synonyme de la duree de vie, presente dans quatre colonnes.

2. PCA sur la matrice de CORRELATION (z-score prealable) et non de covariance.
   age est en centaines de pas, greediness dans [0,1] : sans standardisation PC1
   serait age et rien d'autre.

La PCA est faite par SVD numpy et non par sklearn : cinq lignes, un resultat
identique, et une dependance de moins pour un outil qui doit tourner partout.
"""
import argparse
import fnmatch
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Jeu par defaut : des taux, deux a deux non redondants.
#
# adapt_score en est absent a dessein -- c'est "Delta e moyen mange - Delta e
# moyen vu", identiquement NUL a une seule ressource puisque les deux termes
# sont le meme delta_energy. Colonne constante, que la standardisation ferait
# exploser. adapt_gain aussi : sans ressource a Delta e negatif il se confond
# avec mean_rew.
VARS_DEFAUT = ("age", "mean_rew", "mean_speed", "energy_end",
               "greediness", "repro_rate", "voisinage", "t_explore")

# Regroupements par defaut de --groupes. La coupure est SOLITAIRE / SOCIAL et
# non high_res / low_res : low_res n'a qu'un agent vivant (n_agents_max=2, cf.
# launch_env_low_res), il appartient donc au meme monde que les alone_*. Les
# melanger aux env sociaux ferait de PC1 un axe "y a-t-il quelqu'un", ce qui est
# vrai mais tautologique -- voisinage vaut 0 par construction d'un cote.
GROUPES = {
    "solo":      ("alone_*", "lowres"),
    "clones":    ("clones_*",),
    "figurants": ("figurants_*",),
}

# Une forme par environnement. La couleur etant prise par une variable continue
# (--color), la forme est ce qui reste pour distinguer les contextes : sans elle
# on voit un degrade sans savoir d'ou vient chaque point.
MARQUEURS = ("o", "s", "^", "D", "v", "P", "X", "*")


def lab_data_de(chemin):
    """replay/ EN PREMIER : le lab_data d'un run ne porte que les 50 premiers
    survivants, la ou le rejeu porte toute la population."""
    for candidat in (os.path.join(chemin, "replay", "lab_data"),
                     os.path.join(chemin, "lab_data"),
                     chemin):
        if glob.glob(os.path.join(candidat, "chunk_*.npz")):
            return candidat
    return None


def familles(data_dir):
    """[(env, chunk, fichier)] : tous les phenotypes, tous environnements.

    Ne lit QUE les `chunk_N_pheno_<env>.npz`, une ligne par genome. Les
    `chunk_N.npz` de _agg_lab portent une ligne par evenement de fin de vie et
    ne s'alignent donc pas d'un environnement a l'autre ; les _adapt_* testent
    une permutation et pas un phenotype.
    """
    out = []
    for f in glob.glob(os.path.join(data_dir, "chunk_*_pheno_*.npz")):
        m = re.fullmatch(r"chunk_(\d+)_pheno_(.+)\.npz", os.path.basename(f))
        if m:
            out.append((m.group(2), int(m.group(1)), f))
    return sorted(out)


def filtre(fichiers, motifs):
    """Garde les environnements correspondant a un motif (glob ou nom exact)."""
    return [t for t in fichiers
            if any(fnmatch.fnmatch(t[0], m) for m in motifs)]


def charge(fichiers, noms, extra=None, survivants=False):
    """(X, chunk, env, genome, colonnes, extra), tous environnements empiles.

    `extra` est une colonne lue SANS entrer dans la PCA -- de quoi colorer le
    nuage par une variable qu'on ne veut pas voir influencer les axes.
    """
    lignes, chunks, envs, genomes, colonnes, sup = [], [], [], [], None, []
    for env, chunk, f in fichiers:
        d = np.load(f)
        dispo = [n for n in noms if n in d.files]
        if colonnes is None:
            colonnes = dispo
            manquantes = [n for n in noms if n not in dispo]
            if manquantes:
                print(f"  [info] absentes de ces donnees : {', '.join(manquantes)}")
        elif dispo != colonnes:
            print(f"  [info] {env} chunk {chunk} : colonnes differentes, ignore")
            d.close()
            continue
        bloc = np.stack([np.asarray(d[n], float) for n in colonnes], axis=1)
        garde = slice(None)
        if survivants and "died" in d.files:
            # PC1 est massivement l'axe de mortalite : mourir tot tire ensemble
            # age, energy_end et repro_rate, et ecrase les axes comportementaux.
            # Restreindre aux survivants retire ce facteur commun au lieu de le
            # laisser dominer.
            garde = np.asarray(d["died"]) < 0.5
            bloc = bloc[garde]
        lignes.append(bloc)
        chunks.append(np.full(len(bloc), chunk))
        envs.append(np.full(len(bloc), env, dtype=object))
        g = (np.asarray(d["genome"]) if "genome" in d.files
             else np.arange(len(d[colonnes[0]])))
        genomes.append(g[garde])
        sup.append((np.asarray(d[extra], float)[garde]) if extra and extra in d.files
                   else np.full(len(bloc), np.nan))
        d.close()
    if not lignes:
        return (None,) * 6
    return (np.concatenate(lignes), np.concatenate(chunks),
            np.concatenate(envs), np.concatenate(genomes), colonnes,
            np.concatenate(sup))


def pivot(X, chunk, env, genome, colonnes):
    """Une ligne par (chunk, genome), une colonne par (environnement, mesure).

    C'est le phenotype au sens large : la NORME DE REACTION du genome, ce qu'il
    fait dans chacun des contextes. Deux raisons de preferer ce format au format
    long.

    D'abord l'independance : en format long le meme genome fournit une ligne par
    environnement, donc les lignes sont correlees et la PCA melange la variation
    ENTRE genomes avec la variation ENTRE contextes, sans moyen de les separer.

    Ensuite le sens : un genome bon partout et un genome bon seulement en amas
    deviennent ici deux directions distinctes de l'espace, alors qu'en format
    long ils se confondent avec deux nuages qui se recouvrent.

    Prix a payer : un genome doit avoir une valeur valide dans TOUS les
    environnements pour fournir une ligne.
    """
    envs = sorted(set(env.tolist()))
    cles = sorted({(int(c), int(g)) for c, g in zip(chunk, genome)})
    rang = {k: i for i, k in enumerate(cles)}
    p = len(colonnes)
    W = np.full((len(cles), len(envs) * p), np.nan)
    for j, e in enumerate(envs):
        m = np.flatnonzero(env == e)
        for r in m:
            W[rang[(int(chunk[r]), int(genome[r]))], j * p:(j + 1) * p] = X[r]
    noms = [f"{e}:{n}" for e in envs for n in colonnes]
    return W, np.array([k[0] for k in cles]), noms


def prepare(X, chunk, env, genome, colonnes, sup):
    """Retire les colonnes constantes et les lignes a NaN, en le disant.

    Les NaN ne sont pas repartis au hasard : greediness est indefinie pour un
    agent qui n'a jamais vu de ressource. Les jeter elimine donc precisement les
    phenotypes les plus atypiques -- ceux qui pourraient former un mode a part.
    Le compte est affiche pour que le biais soit visible, pas silencieux.
    """
    garde = []
    for j, nom in enumerate(colonnes):
        col = X[:, j]
        fini = col[np.isfinite(col)]
        if fini.size == 0 or np.allclose(fini, fini[0]):
            print(f"  [info] {nom} : constante, retiree de la PCA")
        else:
            garde.append(j)
    X, colonnes = X[:, garde], [colonnes[j] for j in garde]

    ok = np.isfinite(X).all(axis=1)
    perdus = int((~ok).sum())
    if perdus:
        # nommer les colonnes responsables : en format large une seule d'entre
        # elles peut eliminer la majorite des lignes, et la retirer avec --vars
        # vaut souvent mieux que de perdre les genomes
        coupables = [(colonnes[j], int((~np.isfinite(X[:, j])).sum()))
                     for j in range(X.shape[1])]
        coupables = sorted([c for c in coupables if c[1]],
                           key=lambda c: -c[1])[:4]
        detail = ", ".join(f"{n} ({k})" for n, k in coupables)
        print(f"  [attention] {perdus} lignes sur {len(X)} "
              f"({100*perdus/len(X):.0f} %) ecartees pour NaN. Colonnes en "
              f"cause : {detail}. Ce ne sont pas des lignes quelconques -- "
              "greediness est indefinie quand l'agent n'a jamais vu de "
              "ressource, t_explore quand il n'a jamais mange.")
    return X[ok], chunk[ok], env[ok], genome[ok], colonnes, sup[ok]


def pca(X):
    """(scores, axes, part de variance, moyenne, ecart-type).

    SVD sur la matrice centree-reduite : les vecteurs singuliers droits sont les
    axes principaux, et les valeurs singulieres au carre les variances.
    """
    moy, ect = X.mean(0), X.std(0)
    ect = np.where(ect > 0, ect, 1.0)
    Z = (X - moy) / ect
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    part = S ** 2 / (S ** 2).sum()
    return U * S, Vt, part, moy, ect


def shuffles_en_chunks(exp_dir, chunk_size):
    """Chunks ou une permutation a eu lieu, s'il y a un journal de shuffle."""
    try:
        from simulation.utils.utils_sim import load_shuffle_log
    except Exception:
        return []
    log = load_shuffle_log(exp_dir) or []
    return sorted({int(e["step"]) // chunk_size for e in log})


def trace(scores, chunk, env, colonnes, axes, part, titre, sortie,
          couleur=None, nom_couleur=None, trois_d=False):
    """Un plan (ou un volume) commun a tous les environnements du groupe.

    Environnement par FORME, variable continue par COULEUR : les deux
    informations coexistent, la ou une legende de couleurs par environnement
    interdisait d'en afficher une seconde.
    """
    n_comp = min(3 if trois_d else 2, scores.shape[1])
    trois_d = trois_d and n_comp >= 3
    haut = max(6.0, 0.24 * len(colonnes))
    fig = plt.figure(figsize=(19, haut))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.45, 1])
    ax = fig.add_subplot(gs[0], projection="3d") if trois_d else fig.add_subplot(gs[0])

    noms = sorted(set(env.tolist())) if env is not None else []
    cmap = plt.get_cmap("tab10")
    sc, poignees = None, []

    def points(m, **kw):
        xs = [scores[m, k] for k in range(n_comp)]
        return ax.scatter(*xs, **kw)

    if noms:
        for i, nom in enumerate(noms):
            m = env == nom
            mk = MARQUEURS[i % len(MARQUEURS)]
            if couleur is not None:
                sc = points(m, c=couleur[m], cmap="viridis", marker=mk,
                            s=20, alpha=.6, linewidths=0)
                poignees.append(plt.Line2D([], [], marker=mk, ls="", ms=7,
                                           color="#555555", label=nom))
            else:
                points(m, color=cmap(i % 10), marker=mk, s=18, alpha=.5,
                       linewidths=0, label=nom)
        # barycentre de chaque environnement : c'est l'ecart entre eux qui dit
        # si le contexte deplace le comportement
        for i, nom in enumerate(noms):
            m = env == nom
            c = [scores[m, k].mean() for k in range(n_comp)]
            ax.scatter(*c, marker=MARQUEURS[i % len(MARQUEURS)], s=230,
                       color=cmap(i % 10), edgecolors="black", linewidths=1.5,
                       zorder=6)
        ax.legend(handles=poignees or None, fontsize=8, loc="best",
                  title="environment", markerscale=1.3)
    else:
        sc = points(slice(None), c=couleur, cmap="viridis", s=18, alpha=.55,
                    linewidths=0) if couleur is not None else \
             points(slice(None), color="#1D3557", s=18, alpha=.55, linewidths=0)
    if sc is not None and couleur is not None:
        fig.colorbar(sc, ax=ax, label=nom_couleur, fraction=.035, pad=.10)

    ax.set_xlabel(f"PC1 ({100*part[0]:.0f} %)")
    ax.set_ylabel(f"PC2 ({100*part[1]:.0f} %)")
    if trois_d:
        ax.set_zlabel(f"PC3 ({100*part[2]:.0f} %)")
        ax.view_init(elev=18, azim=-58)
    else:
        ax.grid(alpha=.25)
    ax.set_title("Phenotype space, one point per genome"
                 + ("" if noms else " (all measures, all environments)")
                 + "\nlarge markers: per-environment centroid", fontsize=10)

    # --- charges : sans elles les axes ne veulent rien dire ---
    ax = fig.add_subplot(gs[1])
    y = np.arange(len(colonnes))
    larg = .8 / n_comp
    couleurs_pc = ("#1D3557", "#E76F51", "#2A9D8F")
    for k in range(n_comp):
        ax.barh(y + (k - (n_comp - 1) / 2) * larg, axes[k], height=larg * .92,
                color=couleurs_pc[k], label=f"PC{k+1}")
    ax.set_yticks(y)
    ax.set_yticklabels(colonnes, fontsize=8 if len(colonnes) > 12 else 9)
    ax.axvline(0, color="black", lw=.9)
    ax.set_title("Loadings", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="x")

    # --- variance expliquee ---
    ax = fig.add_subplot(gs[2])
    k = np.arange(1, len(part) + 1)
    ax.bar(k, 100 * part, color="#8D99AE")
    ax.plot(k, 100 * np.cumsum(part), "o-", color="#1D3557", ms=4)
    ax.set_xlabel("component"); ax.set_ylabel("% variance")
    ax.set_title("Explained variance", fontsize=10)
    ax.set_xticks(k); ax.grid(alpha=.25)

    fig.suptitle(titre, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)
    fig.savefig(sortie, dpi=140)
    plt.close(fig)
    print(f"Figure saved: {sortie}")


def trace_html(scores, env, colonnes, part, titre, sortie,
               couleur=None, nom_couleur="chunk"):
    """La meme PCA en 3D interactive, dans un fichier HTML.

    Complement du PNG, pas remplacement : en PCA toute l'information est dans
    les coordonnees, donc une vue 3D n'ajoute rien sur l'ensemble des plans 2D.
    Ce qu'elle apporte est perceptif -- la rotation donne la parallaxe, donc
    leve l'ambiguite de profondeur d'une projection figee. A utiliser pour
    CHERCHER la structure ; pour la montrer, le PNG reste le bon format.

    Meme patron que genealogy.plot_clade_pca_html, qui fait deja cela sur les
    genotypes.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [info] plotly absent : pas de version interactive")
        return
    if scores.shape[1] < 3:
        print("  [info] moins de 3 composantes : pas de version 3D")
        return

    fig = go.Figure()
    noms = sorted(set(env.tolist())) if env is not None else [None]
    formes = ("circle", "square", "diamond", "cross", "x", "circle-open")
    for i, nom in enumerate(noms):
        m = slice(None) if nom is None else (env == nom)
        marqueur = dict(size=3.2, opacity=.75,
                        symbol=formes[i % len(formes)])
        if couleur is not None:
            marqueur.update(color=couleur[m], colorscale="Viridis",
                            showscale=(i == 0),
                            colorbar=dict(title=nom_couleur))
        fig.add_trace(go.Scatter3d(
            x=scores[m, 0], y=scores[m, 1], z=scores[m, 2], mode="markers",
            marker=marqueur, name=nom or "genomes"))

    fig.update_layout(
        title=f"{titre} — var. {100*part[0]:.0f}/{100*part[1]:.0f}/"
              f"{100*part[2]:.0f} %",
        scene=dict(xaxis_title=f"PC1 ({100*part[0]:.0f} %)",
                   yaxis_title=f"PC2 ({100*part[1]:.0f} %)",
                   zaxis_title=f"PC3 ({100*part[2]:.0f} %)"),
        legend=dict(title="environment"))
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)
    fig.write_html(sortie)
    print(f"Interactive saved: {sortie}")


def trace_html_sur(*args, **kw):
    """trace_html, mais un echec n'emporte pas la passe.

    Les PNG sont deja ecrits quand on arrive ici, et les groupes suivants
    restent a traiter : une surprise d'API plotly ne doit pas les faire perdre.
    """
    try:
        trace_html(*args, **kw)
    except Exception as e:
        print(f"  [info] version interactive abandonnee ({type(e).__name__}: {e})")


def deplacement_par_genome(scores, env, genome, chunk):
    """De combien un MEME genome bouge-t-il d'un environnement a l'autre ?

    Compare, pour chaque genome d'un chunk, l'ecart type de ses positions entre
    environnements a l'ecart type de la population dans un environnement donne.
    Un rapport proche de 1 signifie que le contexte deplace autant le
    comportement que le genotype -- donc qu'un point de la figure en dit autant
    sur l'environnement que sur l'agent.
    """
    intra = []
    for c in np.unique(chunk):
        for g in np.unique(genome[chunk == c]):
            m = (chunk == c) & (genome == g)
            if m.sum() > 1:
                intra.append(scores[m, :2].std(axis=0))
    if not intra:
        return None
    intra = np.mean(intra, axis=0)
    inter = np.array([scores[env == n, :2].std(axis=0)
                      for n in sorted(set(env.tolist()))]).mean(axis=0)
    return intra, inter


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier d'experience, son replay, ou lab_data")
    p.add_argument("-o", "--out", default=None,
                   help="dossier de sortie (defaut <source>/fig)")
    p.add_argument("--vars", nargs="+", default=list(VARS_DEFAUT),
                   help=f"mesures retenues (defaut : {' '.join(VARS_DEFAUT)})")
    p.add_argument("--envs", nargs="+", default=None, metavar="ENV",
                   help="restreindre a ces environnements. Motifs glob acceptes "
                        "(ex. 'alone_*' lowres). Defaut : tous ceux trouves")
    p.add_argument("--color", default=None, metavar="COL",
                   help="colonne du npz servant a colorer le nuage, SANS entrer "
                        "dans la PCA. Defaut : une couleur par environnement. "
                        "`died` teste si PC1 n'est qu'un axe de survie, ce qui "
                        "arrive des que la mortalite tire toutes les mesures "
                        "ensemble")
    p.add_argument("--html", action="store_true",
                   help="ecrire AUSSI une version 3D interactive (plotly) a "
                        "cote du PNG. A ouvrir dans un navigateur : la rotation "
                        "leve l'ambiguite de profondeur d'une projection figee. "
                        "Pour explorer ; le PNG reste ce qui s'imprime")
    p.add_argument("--3d", dest="trois_d", action="store_true",
                   help="nuage en PC1-PC2-PC3 au lieu du plan. A lire avec la "
                        "part de variance : une 3e composante a 8 %% n'ajoute "
                        "pas grand-chose et rend la lecture plus difficile")
    p.add_argument("--survivants", action="store_true",
                   help="ne garder que les agents vivants a la fin (died = 0). "
                        "PC1 est sinon massivement l'axe de MORTALITE -- mourir "
                        "tot tire ensemble age, energy_end et repro_rate -- et "
                        "ecrase les axes comportementaux")
    p.add_argument("--large", action="store_true",
                   help="UNE LIGNE PAR GENOME, colonnes = mesures x "
                        "environnements (sa norme de reaction). En format long "
                        "-- le defaut -- le meme genome fournit une ligne par "
                        "environnement, donc les lignes sont correlees et la "
                        "PCA melange variation entre genomes et variation entre "
                        "contextes. Exige une valeur valide dans TOUS les "
                        "environnements")
    p.add_argument("--groupes", action="store_true",
                   help="un plan par groupe : `solo` (alone_* et lowres) et "
                        "`social` (clones_* et figurants_*). Les axes sont "
                        "communs a l'interieur d'un groupe, donc ses "
                        "environnements y sont comparables")
    p.add_argument("--nom", default=None, metavar="NOM",
                   help="nom du fichier de sortie, pour un groupement ad hoc "
                        "fait a la main avec --envs")
    p.add_argument("--par-env", action="store_true",
                   help="une PCA SEPAREE par environnement, avec ses propres "
                        "axes, au lieu d'un plan commun. Decrit mieux la "
                        "variation interne d'un contexte, mais les positions ne "
                        "sont plus comparables d'une figure a l'autre")
    a = p.parse_args()

    data_dir = lab_data_de(a.source)
    if data_dir is None:
        raise SystemExit(f"Aucun chunk_*.npz sous {a.source}")
    fichiers = familles(data_dir)
    if not fichiers:
        raise SystemExit(
            f"Aucun chunk_*_pheno_*.npz sous {data_dir}.\n"
            "Ces fichiers sont ecrits depuis l'ajout de _save_pheno : refaire "
            "la passe de rejeu pour les produire.")
    if a.envs:
        fichiers = filtre(fichiers, a.envs)
        if not fichiers:
            raise SystemExit(f"Aucun environnement parmi {a.envs}")
    trouves = sorted({e for e, _, _ in fichiers})
    print(f"donnees : {data_dir}")
    print(f"{len(trouves)} environnement(s) : {', '.join(trouves)}")

    fig_dir = a.out or os.path.join(a.source, "fig")
    if a.groupes:
        # Un plan par groupe : les axes sont communs A L'INTERIEUR d'un groupe,
        # donc les environnements y sont comparables entre eux -- ce qui n'est
        # pas le cas d'une figure a l'autre.
        for nom, motifs in GROUPES.items():
            sous = filtre(fichiers, motifs)
            if not sous:
                print(f"\n=== {nom} : aucun environnement, saute")
                continue
            noms = sorted({e for e, _, _ in sous})
            print(f"\n=== {nom} : {', '.join(noms)}")
            analyse(sous, a, fig_dir, titre_suffixe=f"   [{nom}]",
                    fichier=f"pca_phenotypes_{nom}.png")
        return

    if a.par_env:
        # Une PCA PAR environnement, axes propres a chacun. Utile pour decrire
        # la variation interne d'un contexte -- mais les positions ne sont alors
        # plus comparables d'une figure a l'autre, les axes ayant tourne. Pour
        # comparer des environnements, c'est le mode par defaut qu'il faut.
        for nom in trouves:
            print(f"\n=== {nom}")
            sous = [t for t in fichiers if t[0] == nom]
            analyse(sous, a, fig_dir, titre_suffixe=f"   [{nom}]",
                    fichier=f"pca_phenotypes_{nom}.png", par_chunk=True)
        return

    analyse(fichiers, a, fig_dir, titre_suffixe=f"   [{a.nom}]" if a.nom else "",
            fichier=f"pca_phenotypes_{a.nom or 'all_envs'}.png")


def analyse(fichiers, a, fig_dir, titre_suffixe, fichier, par_chunk=False):
    """Une PCA sur l'ensemble donne, plus sa figure.

    `par_chunk` colore par chunk au lieu de colorer par environnement : dans une
    PCA a un seul environnement, la couleur par environnement serait uniforme.
    """
    X, chunk, env, genome, colonnes, sup = charge(fichiers, a.vars, a.color,
                                                 survivants=a.survivants)
    if X is None:
        print("  rien a analyser")
        return
    if a.large:
        X, chunk, colonnes = pivot(X, chunk, env, genome, colonnes)
        env = genome = sup = np.zeros(len(X))
        env = None
    X, chunk, env2, genome, colonnes, sup = prepare(
        X, chunk, env if env is not None else np.zeros(len(X), dtype=object),
        genome if not a.large else np.zeros(len(X)), colonnes, sup)
    env = None if a.large else env2
    if X.shape[0] < 3 or X.shape[1] < 2:
        print(f"  {X.shape[0]} genomes x {X.shape[1]} mesures : "
              "pas de quoi faire une PCA, saute")
        return

    scores, axes, part, _, _ = pca(X)
    # Regle usuelle : au moins ~10 observations par variable. En dessous les axes
    # sont pilotes par quelques points et ne se reproduisent pas d'un tirage a
    # l'autre.
    if X.shape[0] < 10 * X.shape[1]:
        print(f"  [attention] {X.shape[0]} lignes pour {X.shape[1]} mesures "
              f"({X.shape[0]/X.shape[1]:.1f} par variable) : axes instables. "
              "Refaire le rejeu avec plus de genomes (-n 0).")
    print(f"  {X.shape[0]} lignes x {X.shape[1]} mesures  "
          f"PC1 {100*part[0]:.0f} %  PC2 {100*part[1]:.0f} %  "
          f"(PC1+PC2 {100*part[:2].sum():.0f} %)")
    for i in (0, 1):
        ordre = np.argsort(-np.abs(axes[i]))[:3]
        detail = ", ".join(f"{colonnes[j]} {axes[i][j]:+.2f}" for j in ordre)
        print(f"    PC{i+1} porte surtout : {detail}")

    if env is not None:
        print("\n  barycentre par environnement (PC1, PC2) :")
        for nom in sorted(set(env.tolist())):
            m = env == nom
            print(f"    {nom:<28} ({scores[m,0].mean():+6.2f}, "
                  f"{scores[m,1].mean():+6.2f})   n={int(m.sum())}")

    d = None if a.large else deplacement_par_genome(scores, env, genome, chunk)
    if d is not None:
        intra, inter = d
        print(f"\n  dispersion d'un MEME genome entre environnements : "
              f"PC1 {intra[0]:.2f}  PC2 {intra[1]:.2f}")
        print(f"  dispersion de la population dans un environnement : "
              f"PC1 {inter[0]:.2f}  PC2 {inter[1]:.2f}")
        print("  (rapport proche de 1 -> le contexte deplace autant que le genotype)")

    couleur, nom_couleur = None, a.color
    if a.color is not None and not a.large:
        couleur = sup
    elif a.color is not None and a.large:
        print("  [info] --color ignore en format large : une ligne y couvre "
              "tous les environnements, la variable n'y a pas de valeur unique")
        couleur, nom_couleur = chunk.astype(float), "chunk"
    elif par_chunk or a.large:
        couleur, nom_couleur = chunk.astype(float), "chunk"
    trace(scores, chunk, env, colonnes, axes, part,
          ("Lab phenotypes — PCA, fixed axes" if par_chunk else
           "Lab phenotypes — one PCA across all environments, fixed axes")
          + titre_suffixe,
          os.path.join(fig_dir, fichier),
          couleur=couleur, nom_couleur=nom_couleur, trois_d=a.trois_d)
    if a.html:
        trace_html_sur(scores, env, colonnes, part,
                   ("Lab phenotypes" + titre_suffixe).strip(),
                   os.path.join(fig_dir, os.path.splitext(fichier)[0] + ".html"),
                   couleur=couleur, nom_couleur=nom_couleur or "chunk")


if __name__ == "__main__":
    main()
