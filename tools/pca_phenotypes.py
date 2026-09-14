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
               "greediness", "repro_rate", "voisinage")

# Regroupements par defaut de --groupes. La coupure est SOLITAIRE / SOCIAL et
# non high_res / low_res : low_res n'a qu'un agent vivant (n_agents_max=2, cf.
# launch_env_low_res), il appartient donc au meme monde que les alone_*. Les
# melanger aux env sociaux ferait de PC1 un axe "y a-t-il quelqu'un", ce qui est
# vrai mais tautologique -- voisinage vaut 0 par construction d'un cote.
GROUPES = {
    "solo":   ("alone_*", "lowres"),
    "social": ("clones_*", "figurants_*"),
}


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


def charge(fichiers, noms, extra=None):
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
        lignes.append(bloc)
        chunks.append(np.full(len(bloc), chunk))
        envs.append(np.full(len(bloc), env, dtype=object))
        genomes.append(np.asarray(d["genome"]) if "genome" in d.files
                       else np.arange(len(bloc)))
        sup.append(np.asarray(d[extra], float) if extra and extra in d.files
                   else np.full(len(bloc), np.nan))
        d.close()
    if not lignes:
        return (None,) * 6
    return (np.concatenate(lignes), np.concatenate(chunks),
            np.concatenate(envs), np.concatenate(genomes), colonnes,
            np.concatenate(sup))


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
        print(f"  [attention] {perdus} genomes sur {len(X)} "
              f"({100*perdus/len(X):.0f} %) ecartes pour NaN. Ce ne sont pas des "
              "lignes quelconques : greediness est indefinie quand l'agent n'a "
              "jamais vu de ressource.")
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
          couleur=None, nom_couleur=None):
    """Un plan, tous les environnements.

    Couleur par ENVIRONNEMENT et non par chunk : la question posee ici est
    "chaque contexte occupe-t-il une region propre", et un degrade temporel la
    masquerait. `--color` bascule sur n'importe quelle colonne si besoin.
    """
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8),
                            gridspec_kw={"width_ratios": [2.3, 1.25, 1]})
    ax = axs[0]
    noms = sorted(set(env.tolist()))
    if couleur is not None:
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=couleur, cmap="viridis",
                        s=14, alpha=.5, linewidths=0)
        fig.colorbar(sc, ax=ax, label=nom_couleur, fraction=.04)
    else:
        cmap = plt.get_cmap("tab10")
        for i, nom in enumerate(noms):
            m = env == nom
            ax.scatter(scores[m, 0], scores[m, 1], s=14, alpha=.45,
                       linewidths=0, color=cmap(i % 10), label=nom)
        ax.legend(fontsize=8, markerscale=1.8, loc="best", title="environment")

    # Le barycentre de chaque environnement, en noir : c'est l'ecart entre eux
    # qui dit si le contexte deplace le comportement.
    cmap = plt.get_cmap("tab10")
    for i, nom in enumerate(noms):
        m = env == nom
        ax.scatter(scores[m, 0].mean(), scores[m, 1].mean(), marker="X", s=210,
                   color=cmap(i % 10), edgecolors="black", linewidths=1.3, zorder=5)
    ax.set_xlabel(f"PC1 ({100*part[0]:.0f} %)")
    ax.set_ylabel(f"PC2 ({100*part[1]:.0f} %)")
    ax.set_title("Phenotype space, one point per genome per environment\n"
                 "X: per-environment centroid", fontsize=10)
    ax.grid(alpha=.25)

    # --- charges : sans elles les axes ne veulent rien dire ---
    ax = axs[1]
    y = np.arange(len(colonnes))
    ax.barh(y - .2, axes[0], height=.38, color="#1D3557", label="PC1")
    ax.barh(y + .2, axes[1], height=.38, color="#E76F51", label="PC2")
    ax.set_yticks(y); ax.set_yticklabels(colonnes, fontsize=9)
    ax.axvline(0, color="black", lw=.9)
    ax.set_title("Loadings", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="x")

    # --- variance expliquee ---
    ax = axs[2]
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
    X, chunk, env, genome, colonnes, sup = charge(fichiers, a.vars, a.color)
    if X is None:
        print("  rien a analyser")
        return
    X, chunk, env, genome, colonnes, sup = prepare(
        X, chunk, env, genome, colonnes, sup)
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

    print("\n  barycentre par environnement (PC1, PC2) :")
    for nom in sorted(set(env.tolist())):
        m = env == nom
        print(f"    {nom:<28} ({scores[m,0].mean():+6.2f}, "
              f"{scores[m,1].mean():+6.2f})   n={int(m.sum())}")

    d = deplacement_par_genome(scores, env, genome, chunk)
    if d is not None:
        intra, inter = d
        print(f"\n  dispersion d'un MEME genome entre environnements : "
              f"PC1 {intra[0]:.2f}  PC2 {intra[1]:.2f}")
        print(f"  dispersion de la population dans un environnement : "
              f"PC1 {inter[0]:.2f}  PC2 {inter[1]:.2f}")
        print("  (rapport proche de 1 -> le contexte deplace autant que le genotype)")

    couleur, nom_couleur = None, a.color
    if a.color is not None:
        couleur = sup
    elif par_chunk:
        couleur, nom_couleur = chunk.astype(float), "chunk"
    trace(scores, chunk, env, colonnes, axes, part,
          ("Lab phenotypes — PCA, fixed axes" if par_chunk else
           "Lab phenotypes — one PCA across all environments, fixed axes")
          + titre_suffixe,
          os.path.join(fig_dir, fichier),
          couleur=couleur, nom_couleur=nom_couleur)


if __name__ == "__main__":
    main()
