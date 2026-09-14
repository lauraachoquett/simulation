"""Descendance potentielle en fonction de la duree de vie, un point par genome.

    python -m simulation.tools.plot_repro_vs_age ~/fusion
    python -m simulation.tools.plot_repro_vs_age ~/fusion/lab_data -o fig/repro.png

`repro_ready` compte les fois ou le seuil de reproduction est atteint, AVANT le
plafond de places libres : c'est donc le nombre de descendants qu'un genome
aurait eus. Le tracer contre l'age separe les deux facons de l'obtenir -- mieux
se nourrir, ou simplement vivre plus longtemps -- que le total seul confond.

La couleur donne le pas de simulation : si les points recents montent au-dessus
de la tendance des anciens A AGE EGAL, c'est un gain de strategie et non de
longevite.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


def lab_data_de(chemin):
    """Le dossier lab_data, qu'on donne la fusion, l'experience ou son replay."""
    for candidat in (chemin,
                     os.path.join(chemin, "lab_data"),
                     os.path.join(chemin, "replay", "lab_data")):
        if glob.glob(os.path.join(candidat, "chunk_*.npz")):
            return candidat
    return None


def charge(data_dir, chunk_size):
    """{geometrie: (age, repro, step)}, "" pour la serie sans geometrie.

    Le motif accepte `chunk_N.npz` et `chunk_N_env_<geo>.npz`, et ecarte donc
    _lowres (l'env low_res ne porte pas repro_ready, il mesure l'exploration et
    non la fecondite) comme _adapt_* et _pheno_*.

    Une entree par geometrie : depuis que la simulation en joue plusieurs, il
    n'existe plus de serie non suffixee, et une lecture qui n'attend que
    `chunk_N.npz` ne trouve plus rien du tout.
    """
    par_geo = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz"))):
        m = re.fullmatch(r"chunk_(\d+)(?:_env_(.+))?\.npz", os.path.basename(f))
        if not m:
            continue
        d = np.load(f)
        if "repro_ready" not in d.files:
            d.close()
            continue
        geo = m.group(2) or ""
        a, r, s_ = par_geo.setdefault(geo, ([], [], []))
        n = len(d["age"])
        a.append(np.asarray(d["age"], float))
        r.append(np.asarray(d["repro_ready"], float))
        s_.append(np.full(n, int(m.group(1)) * chunk_size, float))
        d.close()
    return {g: tuple(np.concatenate(v) for v in trois)
            for g, trois in par_geo.items()}


def court(v):
    """50 000 -> 50k, 1 200 000 -> 1.2M. Les titres de vignette sont le premier
    element a devenir illisible quand on en demande beaucoup."""
    v = float(v)
    if v >= 1e6:
        return f"{v/1e6:.1f}M".replace(".0M", "M")
    if v >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def casiers(age, n_x=26):
    """Bords des casiers : entiers en y, reguliers en x.

    repro_ready est un COMPTE : des casiers a cheval sur les entiers
    melangeraient 1 et 2 descendants dans la meme case et liseraient la densite
    de travers.
    """
    bx = np.linspace(0, max(age.max(), 1), n_x + 1)
    return bx


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="dossier de fusion, d'experience, ou lab_data")
    p.add_argument("-o", "--out", default=None,
                   help="fichier de sortie (defaut <source>/fig/repro_vs_age.png)")
    p.add_argument("--pas", type=int, default=None, metavar="N",
                   help="largeur des tranches en PAS de simulation (ex. 200000). "
                        "Tranches comparables entre elles et entre experiences, "
                        "contrairement a --tranches dont la duree depend de la "
                        "cadence des checkpoints. Les tranches vides sont retirees")
    p.add_argument("--tranches", type=int, default=6,
                   help="nombre de vignettes, decoupees en tranches de PAS "
                        "d'effectif egal (defaut %(default)s)")
    p.add_argument("--repro-max", type=int, default=None,
                   help="dernier casier de descendance. Defaut : le quantile 99, "
                        "pas le maximum -- quelques valeurs rares etiraient l'axe "
                        "et ecrasaient la masse. Le compte du titre dit combien "
                        "sont au-dela")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="courbure de l'echelle de couleur (defaut %(default)s). "
                        "PLUS HAUT = PLUS CLAIR : 1.0 donne une echelle lineaire, "
                        "0.3 sature vite vers le sombre")
    p.add_argument("--clip", type=float, default=0.97,
                   help="quantile des cases occupees ou l'echelle sature "
                        "(defaut %(default)s). PLUS HAUT = PLUS CLAIR : 1.0 "
                        "borne au maximum, donc seule la case la plus peuplee "
                        "est noire")
    p.add_argument("--cmap", default="magma_r",
                   help="palette (defaut %(default)s). Plus douces : Blues, "
                        "YlOrRd, rocket_r. Ajouter _r inverse le sens")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="pas par chunk (defaut : lu dans config.json, sinon 1000)")
    a = p.parse_args()

    data_dir = lab_data_de(a.source)
    if data_dir is None:
        print(f"Aucun chunk_*.npz sous {a.source}")
        return

    chunk_size = a.chunk_size
    if chunk_size is None:
        for c in (os.path.join(a.source, "config.json"),
                  os.path.join(os.path.dirname(data_dir), "..", "config.json")):
            if os.path.isfile(c):
                chunk_size = json.load(open(c)).get("chunk_size")
                break
    if chunk_size is None:
        chunk_size = 1000
        print("[info] chunk_size non trouve, 1000 suppose (--chunk-size pour forcer)")

    series = charge(data_dir, chunk_size)
    if not series:
        print(f"{data_dir} : aucun fichier ne porte repro_ready. Ces donnees "
              "sont anterieures a son ajout, il faut rejouer le lab.")
        return

    for geo, (age, repro, step) in sorted(series.items()):
        trace_serie(a, age, repro, step, geo)


def trace_serie(a, age, repro, step, geo):
    """Une figure pour une geometrie. `geo` vide = serie sans geometrie."""
    titre_geo = f"   [{geo}]" if geo else ""
    print(f"\n=== {geo or 'reference'}")
    nuls = int((repro == 0).sum())
    print(f"{len(age)} genomes, {len(np.unique(step))} instants, "
          f"{nuls} a zero descendant ({100*nuls/len(age):.0f} %)")

    if a.pas:
        # Tranches de LARGEUR FIXE en pas de simulation : comparables entre
        # elles et entre experiences, contrairement aux tranches a effectif egal
        # dont la duree depend de la cadence des checkpoints. Les tranches vides
        # sont retirees plus bas, sinon elles donneraient des vignettes blanches.
        bornes = np.arange(0, step.max() + a.pas, a.pas, dtype=float)
        bornes = bornes[bornes <= step.max() + a.pas]
        bornes[0] -= 1
    else:
        # Tranches a EFFECTIF egal et non a duree egale : les checkpoints ne sont
        # pas forcement reguliers, et une tranche vide ferait une vignette blanche.
        pas_uniques = np.unique(step)
        k = min(a.tranches, len(pas_uniques))
        bornes = np.quantile(step, np.linspace(0, 1, k + 1))
        bornes[0] -= 1                           # inclure le premier instant
    k = len(bornes) - 1

    r_max = (a.repro_max if a.repro_max is not None
             else int(max(np.ceil(np.quantile(repro, 0.99)), 1)))
    hors = int((repro > r_max).sum())
    by = np.arange(-0.5, r_max + 1.5)            # un casier par entier
    bx = casiers(age)

    # Toutes les vignettes partagent l'echelle de couleur, sinon deux images
    # d'allure identique porteraient des effectifs differents. On normalise en
    # FRACTION de la tranche : les tranches n'ont pas le meme nombre de genomes.
    grilles = []
    for i in range(k):
        m = (step > bornes[i]) & (step <= bornes[i + 1])
        if not m.any():        # tranche vide : une vignette blanche n'apprend rien
            continue
        h, _, _ = np.histogram2d(age[m], repro[m], bins=[bx, by])
        grilles.append((h / max(m.sum(), 1), int(m.sum()),
                        step[m].min(), step[m].max(), bornes[i], bornes[i + 1]))
    k = len(grilles)
    if not k:
        print("  aucune tranche non vide")
        return
    # Plafond au quantile des cases NON VIDES, pas au maximum : une seule case
    # tres peuplee -- le paquet a zero descendant, souvent -- compressait tout le
    # reste dans le pale. Et echelle en racine : les densites s'etalent sur deux
    # ordres de grandeur, une echelle lineaire n'en montre que le sommet.
    occupees = np.concatenate([g[0][g[0] > 0].ravel() for g in grilles
                               if (g[0] > 0).any()])
    vmax = float(np.quantile(occupees, a.clip)) if occupees.size else 1.0
    norm = mcolors.PowerNorm(gamma=a.gamma, vmin=0, vmax=vmax)
    cmap = plt.get_cmap(a.cmap).copy()
    cmap.set_bad("white")            # une case vide n'est pas une densite faible

    # Grille a peu pres carree, et non 3 colonnes fixes : a 12 tranches une
    # grille 3x4 donne des vignettes hautes et etroites dont les titres se
    # chevauchent. constrained_layout reserve la place des titres et de la barre
    # de couleur, ce que bbox_inches seul ne fait pas.
    cols = min(k, max(3, int(np.ceil(np.sqrt(k)))))
    lignes = -(-k // cols)
    petit = k > 6
    fig, axes = plt.subplots(lignes, cols,
                             figsize=(3.9 * cols, max(3.3 * lignes, 4.4)),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    for ax in axes.ravel()[k:]:
        ax.axis("off")

    for i, (h, n, s0, s1, b0, b1) in enumerate(grilles):
        ax = axes[i // cols][i % cols]
        im = ax.pcolormesh(bx, by, np.where(h.T > 0, h.T, np.nan), cmap=cmap,
                           norm=norm, shading="flat")
        # mediane par casier d'age : la tendance, sur les valeurs exactes
        m = (step > b0) & (step <= b1)
        centres, med = [], []
        for j in range(len(bx) - 1):
            sel = m & (age >= bx[j]) & (age < bx[j + 1])
            if sel.sum() >= 4:
                centres.append((bx[j] + bx[j + 1]) / 2)
                med.append(np.median(repro[sel]))
        if centres:
            # lisere blanc : la mediane passe sur les cases les plus sombres,
            # ou un trait bleu uni disparait
            ax.plot(centres, med, color="#1D3557", lw=2.0, zorder=3,
                    path_effects=[pe.Stroke(linewidth=3.6, foreground="white"),
                                  pe.Normal()])
        ax.set_title(f"{court(s0)}–{court(s1)}"
                     + ("" if petit else f"   ({n} genomes)"),
                     fontsize=8.5 if petit else 10)
        ax.grid(alpha=.18, zorder=0)

    # une seule paire d'etiquettes d'axes : repetee sur chaque vignette elle
    # mange la place quand il y en a beaucoup
    for ax in axes[-1]:
        ax.set_xlabel("Lifespan in the lab (steps)", fontsize=9 if petit else 10)
    for ligne in axes:
        ligne[0].set_ylabel("Potential offspring", fontsize=9 if petit else 10)
    if petit:
        for ax in axes.ravel():
            ax.tick_params(labelsize=8)

    barre = fig.colorbar(im, ax=axes, fraction=.025)
    barre.set_label("Fraction of genomes in panel", fontsize=9)

    echelle = (f"colour: power \u03b3={a.gamma:g}, "
               f"clipped at p{100*a.clip:.0f}")
    if petit:
        fig.suptitle(f"Potential offspring against lifespan, over time{titre_geo}"
                     f"   —   {len(age)} genomes total   —   {echelle}",
                     fontsize=12)
    else:
        fig.suptitle(f"Potential offspring against lifespan, over time{titre_geo}\n"
                     f"{len(age)} genomes — {nuls} with none "
                     f"({100*nuls/len(age):.0f} %)"
                     + (f" — {hors} above {r_max}, off scale" if hors else "")
                     + f" — shared scale, {echelle}",
                     fontsize=12.5)

    suffixe = f"_{geo}" if geo else ""
    sortie = a.out or os.path.join(a.source, "fig", f"repro_vs_age{suffixe}.png")
    if a.out and geo:        # plusieurs geometries dans un seul --out
        base, ext = os.path.splitext(a.out)
        sortie = f"{base}{suffixe}{ext}"
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {sortie}")


if __name__ == "__main__":
    main()
