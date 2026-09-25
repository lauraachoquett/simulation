"""Les geometries de test et les conditions sociales, comparees.

    python -m simulation.tools.plot_geometries <fusion|exp_dir>              # alone
    python -m simulation.tools.plot_geometries <dir> --condition clones
    python -m simulation.tools.plot_geometries <dir> --par-env               # les 3 conditions
    python -m simulation.tools.plot_geometries <dirA> <dirB> --labels v1 v2  # deux expes

Par defaut : une courbe par geometrie, pour une condition. Avec --par-env : une
ligne par geometrie, les trois conditions superposees.
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

from simulation.utils.plots import _band, _chunks_dans

LARGEUR_LEGENDE = 1.7      # pouces reserves a droite pour la legende

# (cle dans alone_vs_*, cle dans *_summary, titre, unite)
MESURES = [("age", "duree_vie", "Lifespan", "steps"),
           ("greediness", "greediness", "Greediness  G = Cr/Tr", "ratio"),
           ("mean_speed", "mouvement", "Movement", "/step")]
NOMS = {"scatter40_s0": "scattered", "patch8x5_s0": "patchy",
        "blob1x40_s1": "single blob"}
COND = {"alone": "alone", "clones": "identical clones",
        "figurants": "inert peers"}
COULEUR_COND = {"alone": "#4C4C4C", "clones": "#C1121F", "figurants": "#1D5C8F"}


def data_dir_de(chemin):
    for c in (os.path.join(chemin, "replay", "lab_data"),
              os.path.join(chemin, "lab_data"), chemin):
        if glob.glob(os.path.join(c, "chunk_*_summary.json")):
            return c
    return None


def geometries(data_dir):
    """{stem: [resumes]} de l'agent seul, une entree par geometrie."""
    out = {}
    for f in glob.glob(os.path.join(data_dir, "chunk_*_env_*_summary.json")):
        m = re.fullmatch(r"chunk_\d+_env_(.+)_summary\.json", os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def _tries(files):
    return sorted(files, key=lambda f: int(
        re.search(r"chunk_(\d+)", os.path.basename(f)).group(1)))


def serie(data_dir, geo, condition, bornes, pas):
    """(x en chunks, [dict de dispersion par chunk], prefixe) pour une condition.

    `alone` est lu dans les resumes, les conditions sociales dans les fichiers
    apparies alone_vs_<condition>, seuls a les porter.
    """
    if condition == "alone":
        files = _chunks_dans(
            [f for f in glob.glob(
                os.path.join(data_dir, f"chunk_*_env_{geo}_summary.json"))], bornes, pas)
        S = [json.load(open(f)) for f in _tries(files)]
        return np.array([s["chunk"] for s in S]), S, None
    files = _chunks_dans(glob.glob(os.path.join(
        data_dir, f"chunk_*_env_{geo}_alone_vs_{condition}.json")), bornes, pas)
    P = [json.load(open(f)) for f in _tries(files)]
    return np.array([p["chunk"] for p in P]), [p["metrics"] for p in P], condition


def serie_pheno(data_dir, geo, condition, bornes, pas):
    """Par chunk et par MODE : mediane et quartiles, calcules genome par genome.

    Les resumes agregent toute la population : dans un env a un seul amas, ils
    melangent ceux qui ont mange et ceux qui n'ont jamais mange, et la mediane
    tombe dans le creux entre les deux modes.
    """
    marque = f"{condition}_{geo}" if geo else condition
    files = _chunks_dans(glob.glob(os.path.join(
        data_dir, f"chunk_*_pheno_{marque}.npz")), bornes, pas)
    x, par_mode = [], {True: [], False: []}
    for f in _tries(files):
        with np.load(f) as z:
            if "ever_ate" not in z.files:
                continue
            d = {c: np.asarray(z[c], float) for c in z.files}
        mange = np.nan_to_num(d["ever_ate"]) > .5
        x.append(int(re.search(r"chunk_(\d+)", os.path.basename(f)).group(1)))
        for mode in (True, False):
            m = mange if mode else ~mange
            e = {}
            for k_vs, *_ in MESURES:
                v = d.get(k_vs, np.array([]))
                v = v[m] if len(v) else v
                v = v[np.isfinite(v)]
                e[k_vs] = ((np.median(v), np.percentile(v, 25), np.percentile(v, 75))
                           if len(v) >= 3 else (np.nan, np.nan, np.nan))
            e["_n"] = int(m.sum())
            par_mode[mode].append(e)
    return np.array(x), par_mode


def trace_mode(ax, x, entrees, cle, couleur, style, label, chunk_size):
    m = np.array([e[cle][0] for e in entrees])
    lo = np.array([e[cle][1] for e in entrees])
    hi = np.array([e[cle][2] for e in entrees])
    ok = ~np.isnan(m)
    if not ok.any():
        return
    xs = x * chunk_size
    ax.plot(xs[ok], m[ok], color=couleur, lw=2, ls=style, marker="o", ms=3.5,
            label=label)
    ax.fill_between(xs[ok], lo[ok], hi[ok], color=couleur, alpha=.13)


def trace(ax, x, S, prefix, couleur, label, chunk_size):
    """Mediane et p25-p75 d'une mesure ; rien si la mesure manque partout."""
    m, lo, hi = _band(S, prefix)
    ok = ~np.isnan(m)
    if not ok.any():
        return False
    xs = x * chunk_size
    ax.plot(xs[ok], m[ok], marker="o", ms=3.5, color=couleur, label=label)
    okb = ~(np.isnan(lo) | np.isnan(hi))
    ax.fill_between(xs[okb], lo[okb], hi[okb], color=couleur, alpha=.15)
    return True


def separe(a, mesures, data_dir, geos, fig_dir):
    """Une ligne par geometrie, deux courbes par panneau : les deux modes."""
    modes = [(True, "ate at least once", "-"), (False, "never ate", "--")]
    nl, nc = len(geos), len(mesures)
    fig, axes = plt.subplots(nl, nc, squeeze=False, sharex=True,
                             figsize=(5.4 * nc, 3.9 * nl))
    cmap = plt.get_cmap(a.cmap)
    couleurs = [cmap(t) for t in np.linspace(.12, .82, len(geos))]
    for i, geo in enumerate(geos):
        x, par_mode = serie_pheno(data_dir, geo, a.condition, a.chunks, a.pas)
        if not len(x):
            print(f"  [info] pas de phenotypes pour {geo} en {a.condition}")
            continue
        for mode, etiquette, style in modes:
            for j, (k_vs, k_res, titre, unite) in enumerate(mesures):
                ax = axes[i][j]
                trace_mode(ax, x, par_mode[mode], k_vs, couleurs[i], style,
                           etiquette if (i == 0 and j == 0) else None, a.chunk_size)
                ax.set_title(f"{NOMS.get(geo, geo)} — {titre}", fontsize=11)
                ax.set_ylabel(unite)
                ax.grid(alpha=.3)
                if k_vs == "greediness":
                    ax.set_ylim(0, 1)
                if i == nl - 1:
                    ax.set_xlabel("simulation step")
    frac = 1 - LARGEUR_LEGENDE / fig.get_figwidth()
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, title="mode", frameon=False, loc="center left",
               bbox_to_anchor=(frac + .01, .5))
    quoi = "alone" if a.condition == "alone" else f"with {COND[a.condition]}"
    fig.suptitle(f"Focal agent {quoi}, split by mode "
                 "(median and p25–p75 within each mode)", fontsize=13)
    fig.tight_layout(rect=[0, 0, frac, .94])
    out = a.out or os.path.join(fig_dir, f"lab_modes_{a.condition}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


def compare(a, mesures):
    """Plusieurs experiences superposees : une couleur par experience."""
    sources, dirs = [], []
    for src in a.sources:
        d = data_dir_de(src)
        if d is None:
            print(f"  [info] pas de lab_data sous {src}, ignore")
            continue
        sources.append(src)
        dirs.append(d)
    noms = a.labels or [os.path.basename(os.path.normpath(s)) for s in sources]
    geos = [g for g in a.geos if any(geometries(d).get(g) for d in dirs)]
    if not geos:
        raise SystemExit("aucune geometrie commune")

    cmap = plt.get_cmap(a.cmap)
    couleurs = [cmap(t) for t in np.linspace(.12, .82, len(sources))]
    une = len(mesures) == 1          # une mesure : geometries en colonnes
    nl, nc = (1, len(geos)) if une else (len(geos), len(mesures))
    fig, axes = plt.subplots(nl, nc, squeeze=False, sharex=True, sharey=une,
                             figsize=(5.4 * nc, (4.4 if une else 3.9) * nl))
    for i, geo in enumerate(geos):
        for (d, nom, couleur) in zip(dirs, noms, couleurs):
            x, S, pref = serie(d, geo, a.condition, a.chunks, a.pas)
            if not len(x):
                continue
            for j, (k_vs, k_res, titre, unite) in enumerate(mesures):
                ax = axes[0][i] if une else axes[i][j]
                Sk = S if a.condition == "alone" else [s.get(k_vs, {}) for s in S]
                trace(ax, x, Sk, k_res if a.condition == "alone" else pref,
                      couleur, nom, a.chunk_size)
                ax.set_title(NOMS.get(geo, geo) if une
                             else f"{NOMS.get(geo, geo)} — {titre}", fontsize=11)
                ax.set_ylabel(unite)
                ax.grid(alpha=.3)
                if k_vs == "greediness":
                    ax.set_ylim(0, 1)
                if une or i == nl - 1:
                    ax.set_xlabel("simulation step")
    frac = 1 - LARGEUR_LEGENDE / fig.get_figwidth()
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, title="experiment", frameon=False, loc="center left",
               bbox_to_anchor=(frac + .01, .5))
    quoi = "alone" if a.condition == "alone" else f"with {COND[a.condition]}"
    titres = " · ".join(t for _, _, t, _ in mesures)
    fig.suptitle(f"Experiments compared — focal agent {quoi} — {titres}",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, frac, .94])
    out = a.out or os.path.join(sources[0], "fig",
                                f"lab_compare_{a.condition}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sources", nargs="+", metavar="DIR",
                   help="un dossier, ou plusieurs a superposer")
    p.add_argument("--labels", nargs="+", default=None,
                   help="noms des courbes quand plusieurs dossiers sont donnes")
    p.add_argument("--condition", default="alone", choices=list(COND))
    p.add_argument("--mesures", nargs="+", default=None,
                   metavar="M", choices=[m[0] for m in MESURES],
                   help="mesures a tracer parmi age, greediness, mean_speed "
                        "(defaut : les trois)")
    p.add_argument("--separer", action="store_true",
                   help="separer ceux qui ont mange au moins une fois des autres "
                        "(lu genome par genome, pas dans les resumes)")
    p.add_argument("--par-env", dest="par_env", action="store_true",
                   help="une ligne par geometrie, les trois conditions ensemble")
    p.add_argument("--geos", nargs="+", default=list(NOMS),
                   help="ordre des geometries, du disperse au groupe")
    p.add_argument("--chunks", type=int, nargs=2, default=None,
                   metavar=("DEBUT", "FIN"))
    p.add_argument("--pas-chunks", dest="pas", type=int, default=0, metavar="N")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=1000)
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    mesures = ([m for m in MESURES if m[0] in a.mesures] if a.mesures
               else list(MESURES))
    a.source = a.sources[0]
    if len(a.sources) > 1:
        return compare(a, mesures)
    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de lab_data sous {a.source}")
    par_geo = geometries(data_dir)
    geos = [g for g in a.geos if g in par_geo] + \
           [g for g in sorted(par_geo) if g not in a.geos]
    if not geos:
        raise SystemExit(f"aucun resume par geometrie dans {data_dir}")
    fig_dir = os.path.join(a.source, "fig")

    if a.separer:
        return separe(a, mesures, data_dir, geos, fig_dir)
    if a.par_env:
        conds = ["alone"] + [c for c in ("figurants", "clones") if glob.glob(
            os.path.join(data_dir, f"chunk_*_env_*_alone_vs_{c}.json"))]
        # une seule mesure : les geometries en colonnes, c'est plus lisible
        une = len(mesures) == 1
        nl, nc = (1, len(geos)) if une else (len(geos), len(mesures))
        fig, axes = plt.subplots(nl, nc, squeeze=False,
                                 figsize=(5.4 * nc, (4.4 if une else 3.9) * nl),
                                 sharex=True, sharey=une)
        for i, geo in enumerate(geos):
            for cond in conds:
                # alone est lu dans le MEME fichier apparie que la condition
                # sociale : sinon on comparerait deux definitions de la mesure
                famille = cond if cond != "alone" else (
                    conds[1] if len(conds) > 1 else "alone")
                x, S, _ = serie(data_dir, geo, famille, a.chunks, a.pas)
                pref = cond
                if famille == "alone":
                    pref = None
                if not len(x):
                    continue
                for j, (k_vs, k_res, titre, unite) in enumerate(mesures):
                    ax = axes[0][i] if une else axes[i][j]
                    if pref is None:          # aucun fichier apparie : resumes
                        trace(ax, x, S, k_res, COULEUR_COND[cond], COND[cond],
                              a.chunk_size)
                    else:
                        Sk = [s.get(k_vs, {}) for s in S]
                        trace(ax, x, Sk, pref, COULEUR_COND[cond], COND[cond],
                              a.chunk_size)
                    ax.set_title(NOMS.get(geo, geo) if une
                                 else f"{NOMS.get(geo, geo)} — {titre}", fontsize=11)
                    ax.set_ylabel(unite)
                    ax.grid(alpha=.3)
                    if une or i == len(geos) - 1:
                        ax.set_xlabel("simulation step")
                    if k_vs == "greediness":
                        ax.set_ylim(0, 1)
        titres = " · ".join(t for _, _, t, _ in mesures)
        fig.suptitle(f"Social conditions within each test environment — {titres}"
                     r" (median and p25–p75 over genomes)", fontsize=13)
        frac = 1 - LARGEUR_LEGENDE / fig.get_figwidth()
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, title="condition", frameon=False,
                   loc="center left", bbox_to_anchor=(frac + .01, .5))
        fig.tight_layout(rect=[0, 0, frac, .93])
        nom = ("lab_conditions_par_env.png" if len(mesures) == len(MESURES)
               else f"lab_conditions_par_env_{'_'.join(m[0] for m in mesures)}.png")
        out = a.out or os.path.join(fig_dir, nom)
    else:
        cmap = plt.get_cmap(a.cmap)
        couleurs = [cmap(t) for t in np.linspace(.12, .82, len(geos))]
        fig, axes = plt.subplots(1, len(mesures),
                                 figsize=(5.4 * len(mesures), 4.6), squeeze=False)
        axes = axes[0]
        for geo, couleur in zip(geos, couleurs):
            x, S, pref = serie(data_dir, geo, a.condition, a.chunks, a.pas)
            if not len(x):
                print(f"  [info] rien pour {geo} en {a.condition}")
                continue
            for ax, (k_vs, k_res, titre, unite) in zip(axes, mesures):
                Sk = S if a.condition == "alone" else [s.get(k_vs, {}) for s in S]
                trace(ax, x, Sk, k_res if a.condition == "alone" else pref,
                      couleur, NOMS.get(geo, geo), a.chunk_size)
                ax.set_title(titre)
                ax.set_ylabel(unite)
                ax.set_xlabel("simulation step")
                ax.grid(alpha=.3)
        for ax, (k_vs, *_) in zip(axes, mesures):
            if k_vs == "greediness":
                ax.set_ylim(0, 1)
        frac = 1 - LARGEUR_LEGENDE / fig.get_figwidth()
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, title="test environment", frameon=False,
                   loc="center left", bbox_to_anchor=(frac + .01, .5))
        quoi = ("alone" if a.condition == "alone"
                else f"with {COND[a.condition]}")
        fig.suptitle(f"Focal agent {quoi}, across test geometries "
                     "(median and p25–p75 over genomes)", fontsize=13)
        fig.tight_layout(rect=[0, 0, frac, .93])
        suffixe = ("" if len(mesures) == len(MESURES)
                   else "_" + "_".join(m[0] for m in mesures))
        out = a.out or os.path.join(
            fig_dir, f"lab_geometries_{a.condition}{suffixe}.png")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
