"""PCA des phenotypes de lab : ou va la population dans l'espace des mesures.

    python -m simulation.tools.pca_phenotypes <exp_dir>
    python -m simulation.tools.pca_phenotypes <exp_dir> --vars age greediness voisinage

Les axes sont ajustes UNE SEULE FOIS sur tous les chunks empiles, puis chaque
chunk y est projete. C'est ce qui rend la trajectoire lisible : avec une PCA
par chunk, les axes tourneraient d'une image a l'autre et un deplacement
apparent ne voudrait rien dire.

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
    """{geometrie: [(chunk, fichier)]}, "" pour la serie sans geometrie.

    Le motif exige soit rien soit `_env_<nom>` : les suffixes _lowres et
    _adapt_* sont donc ecartes d'eux-memes, et c'est voulu -- l'un ne porte pas
    les memes colonnes, l'autre teste une permutation et pas un phenotype.
    """
    out = {}
    for f in glob.glob(os.path.join(data_dir, "chunk_*.npz")):
        m = re.fullmatch(r"chunk_(\d+)(?:_env_(.+))?\.npz", os.path.basename(f))
        if m:
            out.setdefault(m.group(2) or "", []).append((int(m.group(1)), f))
    return {k: sorted(v) for k, v in out.items()}


def charge(fichiers, noms, extra=None):
    """(X, chunk, colonnes, extra) : une ligne par genome, une colonne par mesure.

    `extra` est une colonne lue SANS entrer dans la PCA -- de quoi colorer le
    nuage par une variable qu'on ne veut pas voir influencer les axes.
    """
    lignes, chunks, colonnes, sup = [], [], None, []
    for chunk, f in fichiers:
        d = np.load(f)
        dispo = [n for n in noms if n in d.files]
        if colonnes is None:
            colonnes = dispo
            manquantes = [n for n in noms if n not in dispo]
            if manquantes:
                print(f"  [info] absentes de ces donnees : {', '.join(manquantes)}")
        elif dispo != colonnes:
            print(f"  [info] chunk {chunk} : colonnes differentes, ignore")
            d.close()
            continue
        bloc = np.stack([np.asarray(d[n], float) for n in colonnes], axis=1)
        lignes.append(bloc)
        chunks.append(np.full(len(bloc), chunk))
        sup.append(np.asarray(d[extra], float) if extra and extra in d.files
                   else np.full(len(bloc), np.nan))
        d.close()
    if not lignes:
        return None, None, None, None
    return (np.concatenate(lignes), np.concatenate(chunks), colonnes,
            np.concatenate(sup))


def prepare(X, chunk, colonnes, sup):
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
    return X[ok], chunk[ok], colonnes, sup[ok]


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


def trace(scores, chunk, colonnes, axes, part, titre, sortie, shuffles=(),
          couleur=None, nom_couleur="chunk"):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6),
                            gridspec_kw={"width_ratios": [2.1, 1.25, 1]})

    # --- 1) le nuage, colore par chunk, et la trajectoire du barycentre ---
    ax = axs[0]
    c = chunk if couleur is None else couleur
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=c, cmap="viridis",
                    s=16, alpha=.55, linewidths=0)
    uniq = np.unique(chunk)
    cx = np.array([scores[chunk == c, 0].mean() for c in uniq])
    cy = np.array([scores[chunk == c, 1].mean() for c in uniq])
    ax.plot(cx, cy, "-", color="black", lw=1.4, zorder=3, alpha=.8)
    ax.scatter(cx, cy, c=uniq, cmap="viridis", s=95, edgecolors="black",
               linewidths=1.1, zorder=4)
    # les permutations, la ou elles tombent sur la trajectoire : c'est la
    # perturbation dont on veut savoir si le nuage revient
    for s in shuffles:
        if s in uniq:
            i = int(np.where(uniq == s)[0][0])
            ax.scatter(cx[i], cy[i], marker="D", s=190, facecolors="none",
                       edgecolors="#E63946", linewidths=2.0, zorder=5)
    if len(uniq) > 1:
        ax.annotate("", xy=(cx[-1], cy[-1]), xytext=(cx[-2], cy[-2]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    ax.set_xlabel(f"PC1 ({100*part[0]:.0f} %)")
    ax.set_ylabel(f"PC2 ({100*part[1]:.0f} %)")
    ax.set_title(f"Phenotype space, one point per genome (colour: {nom_couleur})\n"
                 "black: per-chunk centroid   red: channel shuffle", fontsize=10)
    ax.grid(alpha=.25)
    fig.colorbar(sc, ax=ax, label=nom_couleur, fraction=.04)

    # --- 2) les charges : sans elles les axes ne veulent rien dire ---
    ax = axs[1]
    y = np.arange(len(colonnes))
    ax.barh(y - .2, axes[0], height=.38, color="#1D3557", label="PC1")
    ax.barh(y + .2, axes[1], height=.38, color="#E76F51", label="PC2")
    ax.set_yticks(y); ax.set_yticklabels(colonnes, fontsize=9)
    ax.axvline(0, color="black", lw=.9)
    ax.set_title("Loadings", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="x")

    # --- 3) variance expliquee ---
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier d'experience, son replay, ou lab_data")
    p.add_argument("-o", "--out", default=None,
                   help="dossier de sortie (defaut <source>/fig)")
    p.add_argument("--vars", nargs="+", default=list(VARS_DEFAUT),
                   help=f"mesures retenues (defaut : {' '.join(VARS_DEFAUT)})")
    p.add_argument("--color", default="chunk", metavar="COL",
                   help="colonne du npz servant a colorer le nuage, SANS entrer "
                        "dans la PCA (defaut chunk). `died` teste si PC1 n'est "
                        "qu'un axe de survie, ce qui arrive des que la mortalite "
                        "tire toutes les mesures ensemble")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="pas par chunk, pour situer les permutations "
                        "(defaut : lu dans config.json, sinon 1000)")
    a = p.parse_args()

    data_dir = lab_data_de(a.source)
    if data_dir is None:
        raise SystemExit(f"Aucun chunk_*.npz sous {a.source}")
    print(f"donnees : {data_dir}")

    chunk_size = a.chunk_size
    if chunk_size is None:
        for c in (os.path.join(a.source, "config.json"),
                  os.path.join(data_dir, "..", "..", "config.json")):
            if os.path.isfile(c):
                chunk_size = json.load(open(c)).get("chunk_size")
                break
    chunk_size = chunk_size or 1000
    shuffles = shuffles_en_chunks(a.source, chunk_size)

    fig_dir = a.out or os.path.join(a.source, "fig")
    fams = familles(data_dir)
    if not fams:
        raise SystemExit("Aucune serie phenotypique (chunk_N.npz ou chunk_N_env_*.npz)")

    for geo, fichiers in sorted(fams.items()):
        nom = geo or "reference"
        print(f"\n=== {nom} : {len(fichiers)} chunk(s)")
        extra = None if a.color == "chunk" else a.color
        X, chunk, colonnes, sup = charge(fichiers, a.vars, extra)
        if X is None:
            continue
        X, chunk, colonnes, sup = prepare(X, chunk, colonnes, sup)
        if X.shape[0] < 3 or X.shape[1] < 2:
            print(f"  [info] {X.shape[0]} genomes x {X.shape[1]} mesures : "
                  "pas de quoi faire une PCA, saute")
            continue
        scores, axes, part, _, _ = pca(X)
        # Regle usuelle : au moins ~10 observations par variable. En dessous les
        # axes sont pilotes par quelques points et ne se reproduisent pas d'un
        # tirage a l'autre.
        if X.shape[0] < 10 * X.shape[1]:
            print(f"  [attention] {X.shape[0]} genomes pour {X.shape[1]} mesures "
                  f"({X.shape[0]/X.shape[1]:.1f} par variable) : les axes sont "
                  "instables. Relancer le rejeu avec plus de genomes (-n 0).")
        if len(np.unique(chunk)) < 4:
            print(f"  [attention] {len(np.unique(chunk))} chunk(s) : la "
                  "trajectoire du barycentre ne dit rien d'une dynamique.")
        print(f"  {X.shape[0]} genomes x {X.shape[1]} mesures  "
              f"PC1 {100*part[0]:.0f} %  PC2 {100*part[1]:.0f} %  "
              f"(PC1+PC2 {100*part[:2].sum():.0f} %)")
        for i in (0, 1):
            ordre = np.argsort(-np.abs(axes[i]))[:3]
            detail = ", ".join(f"{colonnes[j]} {axes[i][j]:+.2f}" for j in ordre)
            print(f"    PC{i+1} porte surtout : {detail}")
        suffixe = f"_{geo}" if geo else ""
        trace(scores, chunk, colonnes, axes, part,
              f"Lab phenotypes — PCA on pooled chunks, fixed axes"
              + (f"   [{geo}]" if geo else ""),
              os.path.join(fig_dir, f"pca_phenotypes{suffixe}.png"),
              shuffles=shuffles,
              couleur=None if extra is None else sup, nom_couleur=a.color)


if __name__ == "__main__":
    main()
