import graphviz
import matplotlib.pyplot as plt  # used by plot_full_genealogy_robust for colormap
import matplotlib.colors as mcolors
import numpy as np
import os
import warnings
import jax
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import glob 
import json
import re 
from simulation.data_class import COLOR_BY_ID,LABELS
from simulation.utils.utils_sim import build_id_timeline

GROUPS = ["encoder", "lstm_input", "lstm_recurrent", "controller"]


def block_edges(n, start_step=0, n_target=2000, cut_steps=()):
    """Bornes de blocs pour reduire une serie de `n` pas a ~`n_target` points.

    Une figure de 1200 px ne resout pas plus de ~1200 points ; en tracer 1,6 M
    coute lineairement en longueur de run pour un resultat identique. On agrege
    donc par blocs, ce qui rend le cout des figures CONSTANT.

    Les bornes sont forcees a tomber sur les `cut_steps` (les shuffles) : un bloc
    a cheval sur deux configs melangerait deux identites de ressources et
    recevrait une seule couleur via build_id_timeline.
    """
    coupes = sorted({0, n} | {int(s) - start_step for s in cut_steps
                              if 0 < int(s) - start_step < n})
    par_bloc = max(1, n // max(n_target, 1))
    edges = []
    for a, b in zip(coupes[:-1], coupes[1:]):
        k = max(1, round((b - a) / par_bloc))
        edges.extend(a + (b - a) * i // k for i in range(k))
    edges.append(n)
    return np.array(sorted(set(edges)))


def block_apply(arr, edges, how="mean"):
    """Agrege arr (n,) ou (n, c) par blocs.

    how = 'mean' (niveaux), 'sum' (comptes), ou 'nanmean' pour les series a
    trous -- un seul NaN suffirait a effacer tout un bloc avec np.mean.

    Les COMPTES doivent etre sommes, jamais moyennes : pour un ratio k/n, moyenner
    les ratios donnerait le meme poids a un bloc ou 3 agents voient et a un ou 300
    voient. On somme n et k separement, on divise ensuite.
    """
    a = np.asarray(arr, dtype=float)
    plat = a.ndim == 1
    if plat:
        a = a[:, None]
    if len(edges) < 2:                      # serie vide -> aucun bloc
        return np.array([]) if plat else np.empty((0, a.shape[1]))
    f = {"mean": np.mean, "sum": np.sum, "nanmean": np.nanmean}[how]
    with warnings.catch_warnings():         # bloc entierement NaN -> NaN, sans bruit
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.stack([f(a[lo:hi], axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    return out[:, 0] if plat else out


def block_steps(edges, start_step=0):
    """Position en x de chaque bloc : son centre."""
    return start_step + 0.5 * (edges[:-1] + edges[1:])


def plot_evolution(pop_history, res_history, exp_dir,shuffle_log,initial_order_ids,start_step=0,steps=None):
    plot_evolution_png(pop_history, res_history, exp_dir,shuffle_log,initial_order_ids,start_step=start_step,steps=steps)
    plot_evolution_html(pop_history, res_history, exp_dir,shuffle_log,initial_order_ids,start_step=start_step,steps=steps)

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def plot_evolution_png(pop_history, res_history, exp_dir, shuffle_log,
                       initial_order_ids, start_step=0, steps=None):
    generations = (np.asarray(steps) if steps is not None
                   else np.arange(start_step, start_step + len(pop_history)))

    fig, ax_evo_agents = plt.subplots(figsize=(12, 6))
    ax_evo_agents.set_xlabel('Steps')
    ax_evo_agents.set_ylabel('Population size', color='tab:red')
    ax_evo_agents.plot(generations, pop_history, color='tab:red', linewidth=2)
    ax_evo_agents.tick_params(axis='y', labelcolor='tab:red')
    ax_evo_agents.grid(True, alpha=0.3)

    ax_evo_res = ax_evo_agents.twinx()
    ax_evo_res.set_ylabel('Resources amount', color='tab:green')

    res = np.asarray(res_history)
    if res.ndim == 1:
        res = res[:, None]
    n_types = res.shape[1]

    id_timeline = build_id_timeline(generations, shuffle_log, initial_order_ids)  # (T, n_types)

    for k in range(n_types):
        y      = res[:, k]
        ids_k  = id_timeline[:, k]
        points = np.array([generations, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = [COLOR_BY_ID[i] for i in ids_k[:-1]]        # 1 couleur par segment
        ax_evo_res.add_collection(LineCollection(segments, colors=seg_colors, linewidth=2))

    # LineCollection ne cadre pas seule
    ax_evo_res.set_xlim(generations[0], generations[-1])
    ax_evo_res.set_ylim(0, res.max() * 1.05)
    ax_evo_res.tick_params(axis='y', labelcolor='tab:green')

    # légende = les IDENTITÉS (couleur fixe), pas les canaux
    present_ids = sorted(set(int(i) for i in id_timeline.ravel()))
    handles = [Line2D([0], [0], color=COLOR_BY_ID[i], lw=2, label=LABELS[i]) for i in present_ids]
    ax_evo_res.legend(handles=handles, loc='upper right')

    ax_evo_agents.set_title('Simulation dynamic')
    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig'); os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, 'plot_evo.png')); plt.close()

from matplotlib.collections import LineCollection


def plot_consumption(pop_history, consumed_history, exp_dir, shuffle_log,
                     initial_order_ids, start_step=0, window=1, name_fig='plot_conso',
                     steps=None):
    """Comme plot_evolution, mais l'axe droit porte le FLUX consomme par type
    (unites retirees de la grille par step) au lieu du stock present.

    window > 1 -> moyenne glissante sur `window` steps (courbe lissee)."""
    plot_consumption_png(pop_history, consumed_history, exp_dir, shuffle_log,
                         initial_order_ids, start_step=start_step, window=window,
                         name_fig=name_fig, steps=steps)
    plot_consumption_html(pop_history, consumed_history, exp_dir, shuffle_log,
                          initial_order_ids, start_step=start_step, window=window,
                          name_fig=name_fig, steps=steps)


def _smooth_consumption(consumed, window):
    """Moyenne glissante centree par canal. window=1 -> renvoie le brut.

    On divise par le nombre de points REELLEMENT dans la fenetre plutot que par
    `window` : sinon les ~window/2 premiers et derniers points sont tires vers 0
    par le zero-padding de np.convolve, ce qui ferait un faux effondrement de la
    conso aux deux bouts du graphe."""
    consumed = np.asarray(consumed, dtype=float)
    if consumed.ndim == 1:
        consumed = consumed[:, None]
    if window <= 1 or len(consumed) < window:
        return consumed
    kernel = np.ones(window)
    counts = np.convolve(np.ones(len(consumed)), kernel, mode='same')   # (T,)
    return np.stack(
        [np.convolve(consumed[:, k], kernel, mode='same') / counts
         for k in range(consumed.shape[1])],
        axis=1,
    )


def plot_consumption_png(pop_history, consumed_history, exp_dir, shuffle_log,
                         initial_order_ids, start_step=0, window=1, name_fig='plot_conso',
                         steps=None):
    generations = (np.asarray(steps) if steps is not None
                   else np.arange(start_step, start_step + len(pop_history)))

    fig, ax_pop = plt.subplots(figsize=(12, 6))
    ax_pop.set_xlabel('Steps')
    ax_pop.set_ylabel('Population size', color='tab:red')
    ax_pop.plot(generations, pop_history, color='tab:red', linewidth=2)
    ax_pop.tick_params(axis='y', labelcolor='tab:red')
    ax_pop.grid(True, alpha=0.3)

    ax_conso = ax_pop.twinx()
    ax_conso.set_ylabel('Resources consumed / step', color='tab:green')

    conso = _smooth_consumption(consumed_history, window)
    n_types = conso.shape[1]

    id_timeline = build_id_timeline(generations, shuffle_log, initial_order_ids)  # (T, n_types)

    for k in range(n_types):
        y      = conso[:, k]
        ids_k  = id_timeline[:, k]
        points = np.array([generations, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = [COLOR_BY_ID[i] for i in ids_k[:-1]]        # 1 couleur par segment
        ax_conso.add_collection(LineCollection(segments, colors=seg_colors, linewidth=2))

    # LineCollection ne cadre pas seule ; max(.., 1) car la conso peut etre nulle partout
    ax_conso.set_xlim(generations[0], generations[-1])
    ax_conso.set_ylim(0, max(conso.max(), 1.0) * 1.05)
    ax_conso.tick_params(axis='y', labelcolor='tab:green')

    # légende = les IDENTITÉS (couleur fixe), pas les canaux
    present_ids = sorted(set(int(i) for i in id_timeline.ravel()))
    handles = [Line2D([0], [0], color=COLOR_BY_ID[i], lw=2, label=LABELS[i]) for i in present_ids]
    ax_conso.legend(handles=handles, loc='upper right')

    ax_pop.set_title(_consumption_title(window))
    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig'); os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'{name_fig}.png')); plt.close()


def _consumption_title(window):
    return ('Resource consumption' if window <= 1
            else f'Resource consumption (rolling mean, {window} steps)')


def plot_consumption_html(pop_history, consumed_history, exp_dir, shuffle_log,
                          initial_order_ids, start_step=0, window=1, name_fig='plot_conso',
                          steps=None):
    generations = (np.asarray(steps) if steps is not None
                   else np.arange(start_step, start_step + len(pop_history)))
    T = len(generations)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=generations, y=pop_history, name='Agents',
                   line=dict(color='red', width=2)),
        secondary_y=False,
    )

    conso = _smooth_consumption(consumed_history, window)
    n_types = conso.shape[1]

    id_timeline = build_id_timeline(generations, shuffle_log, initial_order_ids)  # (T, n_types)

    def rgba(i):
        r, g, b, a = mcolors.to_rgba(COLOR_BY_ID[i])
        return f"rgba({r*255:.0f},{g*255:.0f},{b*255:.0f},{a:.2f})"

    seen = set()                                  # chaque id une seule fois en légende
    for k in range(n_types):
        ids_k = id_timeline[:, k]
        change = np.where(np.diff(ids_k) != 0)[0] + 1
        bounds = [0, *change.tolist(), T]

        for s, e in zip(bounds[:-1], bounds[1:]):
            i   = int(ids_k[s])
            sl  = slice(s, min(e + 1, T))         # +1 point pour raccorder sans trou
            show = i not in seen
            seen.add(i)
            fig.add_trace(
                go.Scatter(
                    x=generations[sl], y=conso[sl, k],
                    name=LABELS[i], legendgroup=LABELS[i], showlegend=show,
                    line=dict(color=rgba(i), width=2),
                ),
                secondary_y=True,
            )

    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='Population size', secondary_y=False,
                     title_font=dict(color='red'), tickfont=dict(color='red'))
    fig.update_yaxes(title_text='Resources consumed / step', secondary_y=True,
                     title_font=dict(color='green'), tickfont=dict(color='green'))
    fig.update_layout(title=_consumption_title(window))

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'{name_fig}.html'))


def plot_prob_eat_given_seen(pop_history, n_seen, n_eaten, exp_dir, shuffle_log,
                             initial_order_ids, start_step=0, window=100,
                             name_fig='plot_prob_eat', steps=None):
    """P(manger le type k | type k dans le champ de vision), une courbe par identite.

    n_seen / n_eaten : (T, n_types), comptes d'agents par pas (cf.
    compute_seen_eaten_chunk). On agrege sur `window` pas puis on divise :
    lisser le RAPPORT donnerait le meme poids a un pas ou 3 agents voient et a
    un pas ou 300 voient."""
    plot_prob_eat_png(pop_history, n_seen, n_eaten, exp_dir, shuffle_log,
                      initial_order_ids, start_step, window, name_fig, steps)
    plot_prob_eat_html(pop_history, n_seen, n_eaten, exp_dir, shuffle_log,
                       initial_order_ids, start_step, window, name_fig, steps)


def _pooled_ratio(n_seen, n_eaten, window):
    """(T, n_types) : k/n agreges sur une fenetre glissante centree.

    NaN la ou personne n'a rien vu -- P n'y est pas defini, et forcer 0 ferait
    croire a un evitement parfait alors qu'il n'y a aucune donnee."""
    n = np.asarray(n_seen, dtype=float)
    k = np.asarray(n_eaten, dtype=float)
    if n.ndim == 1:
        n, k = n[:, None], k[:, None]
    if window > 1 and len(n) >= window:
        kern = np.ones(window)
        pool = lambda a: np.stack(
            [np.convolve(a[:, c], kern, mode='same') for c in range(a.shape[1])], axis=1)
        n, k = pool(n), pool(k)
    return np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)


def plot_prob_eat_png(pop_history, n_seen, n_eaten, exp_dir, shuffle_log,
                      initial_order_ids, start_step=0, window=100,
                      name_fig='plot_prob_eat', steps=None):
    generations = (np.asarray(steps) if steps is not None
                   else np.arange(start_step, start_step + len(pop_history)))
    p = _pooled_ratio(n_seen, n_eaten, window)
    n_types = p.shape[1]

    fig, ax_pop = plt.subplots(figsize=(12, 6))
    ax_pop.set_xlabel('Steps')
    ax_pop.set_ylabel('Population size', color='tab:red')
    ax_pop.plot(generations, pop_history, color='tab:red', linewidth=1.2, alpha=0.45)
    ax_pop.tick_params(axis='y', labelcolor='tab:red')
    ax_pop.grid(True, alpha=0.3)

    ax_p = ax_pop.twinx()
    ax_p.set_ylabel('P(eat | in field of view)')

    id_timeline = build_id_timeline(generations, shuffle_log, initial_order_ids)

    for k in range(n_types):
        y     = p[:, k]
        ids_k = id_timeline[:, k]
        points = np.array([generations, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # un segment touchant un NaN n'est pas trace par LineCollection -> les
        # trous (personne n'a rien vu) restent visibles au lieu d'etre interpoles
        seg_colors = [COLOR_BY_ID[i] for i in ids_k[:-1]]
        ax_p.add_collection(LineCollection(segments, colors=seg_colors, linewidth=2))

    ax_p.set_xlim(generations[0], generations[-1])
    # add_collection met a jour les limites de donnees mais ne les APPLIQUE pas :
    # sans ce declenchement l'axe reste au defaut (0, 1) et des probabilites toutes
    # faibles s'ecrasent en bas du cadre. Meme piege que dans plot_evolution.
    # bottom=0 garde le zero comme ancrage : sinon une courbe oscillant entre 0.02
    # et 0.05 remplit tout le graphe et parait tres instable.
    ax_p.autoscale_view()
    ax_p.set_ylim(bottom=0)
    ax_p.axhline(0, color='grey', lw=0.6, alpha=0.5)

    present_ids = sorted(set(int(i) for i in id_timeline.ravel()))
    handles = [Line2D([0], [0], color=COLOR_BY_ID[i], lw=2, label=LABELS[i])
               for i in present_ids]
    handles.append(Line2D([0], [0], color='tab:red', lw=1.2, alpha=0.45,
                          label='population'))
    ax_p.legend(handles=handles, loc='upper right', fontsize=8)

    ax_pop.set_title('P(eat | resource in field of view)'
                     + (f'  —  pooled over {window} steps' if window > 1 else ''))
    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig'); os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'{name_fig}.png')); plt.close()


def plot_prob_eat_html(pop_history, n_seen, n_eaten, exp_dir, shuffle_log,
                       initial_order_ids, start_step=0, window=100,
                       name_fig='plot_prob_eat', steps=None):
    generations = (np.asarray(steps) if steps is not None
                   else np.arange(start_step, start_step + len(pop_history)))
    T = len(generations)
    p = _pooled_ratio(n_seen, n_eaten, window)
    n_types = p.shape[1]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=generations, y=pop_history, name='Agents',
                   line=dict(color='red', width=1.2), opacity=0.45),
        secondary_y=True,
    )

    id_timeline = build_id_timeline(generations, shuffle_log, initial_order_ids)

    def rgba(i):
        r, g, b, a = mcolors.to_rgba(COLOR_BY_ID[i])
        return f"rgba({r*255:.0f},{g*255:.0f},{b*255:.0f},{a:.2f})"

    seen = set()
    for k in range(n_types):
        ids_k = id_timeline[:, k]
        change = np.where(np.diff(ids_k) != 0)[0] + 1
        bounds = [0, *change.tolist(), T]
        for s, e in zip(bounds[:-1], bounds[1:]):
            i   = int(ids_k[s])
            sl  = slice(s, min(e + 1, T))
            show = i not in seen
            seen.add(i)
            fig.add_trace(
                go.Scatter(
                    x=generations[sl], y=p[sl, k],
                    name=LABELS[i], legendgroup=LABELS[i], showlegend=show,
                    line=dict(color=rgba(i), width=2),
                ),
                secondary_y=False,
            )

    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='P(eat | in field of view)', secondary_y=False)
    fig.update_yaxes(title_text='Population size', secondary_y=True,
                     title_font=dict(color='red'), tickfont=dict(color='red'))
    fig.update_layout(title='P(eat | resource in field of view)'
                            + (f' — pooled over {window} steps' if window > 1 else ''))

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'{name_fig}.html'))


def plot_prob_eat_over_life(curves, n_pooled, k_pooled, wilson, exp_dir, chunk, tag,
                            label='poison', mapping='', baseline=None):
    """Une courbe par agent au fil du TEMPS, + la mediane.

    curves : (B, n_bins) — P(manger | en vue) de chaque agent dans chaque
    tranche de temps (cf. LabMixin.prob_eat_over_life). NaN la ou un agent n'a
    pas rencontre le type dans la tranche -> agregation par nanmedian.

    Chaque agent est compare a lui-meme sur un axe qui ne depend PAS de ses
    choix : pas de biais de composition, contrairement a un axe indexe sur le
    nombre deja mange, qui monte a droite meme sans apprentissage.

    Lire la mediane en premier : les pentes par agent reposent sur peu
    d'evenements, donc la fraction d'agents decroissants est un signal plus
    faible que le deplacement de la mediane."""
    c = np.asarray(curves, dtype=float)
    if c.size == 0:
        return
    B, n_bins = c.shape
    x = np.arange(n_bins)

    # pente par agent, sur ses tranches renseignees uniquement
    pentes = []
    for ci in c:
        ok = np.isfinite(ci)
        if ok.sum() >= 2:
            pentes.append(np.polyfit(x[ok], ci[ok], 1)[0])
    pentes = np.array(pentes)
    frac_dec = float(np.mean(pentes < 0)) if pentes.size else float('nan')
    n_pentes = pentes.size
    # test du signe contre 50% : ecart-type binomial sous H0
    z = (frac_dec - 0.5) / (0.5 / np.sqrt(n_pentes)) if n_pentes else 0.0

    fig, ax = plt.subplots(figsize=(8.5, 5))
    color = COLOR_BY_ID[[i for i, l in enumerate(LABELS) if l == label][0]]

    for ci in c:                                    # faisceau individuel
        ax.plot(x, ci, color=color, alpha=0.13, lw=1, zorder=2)

    # Courbe agregee : ratio POOLE, pas la mediane des ratios par agent. Avec
    # une centaine de rencontres par agent et par tranche et p ~ 0.03, chaque
    # valeur individuelle est un multiple de 1/n : la mediane saute entre ces
    # niveaux discrets et bouge de +/-30% sans qu'il se passe rien.
    n_p = np.asarray(n_pooled, dtype=float)
    k_p = np.asarray(k_pooled, dtype=float)
    p, lo_w, hi_w = wilson(k_p, n_p)
    ok = n_p > 0
    ax.fill_between(x[ok], lo_w[ok], hi_w[ok], color=color, alpha=0.22, zorder=3,
                    label='Wilson 95%')
    ax.plot(x[ok], p[ok], 'o-', color=color, lw=2.5, ms=6, zorder=4,
            label='pooled over agents')

    # Baseline tracee COMME UNE COURBE, sur les memes tranches de vie.
    # Une ligne horizontale (poolee sur tout le rollout) laisserait ouverte une
    # explication concurrente : au debut de chaque rollout l'etat LSTM est remis
    # a zero (model.reset_b), donc une partie de la chute pourrait n'etre que la
    # convergence de la memoire recurrente, pas un apprentissage sur le poison.
    # Si la baseline est plate et que la condition permutee plonge, l'ambiguite
    # tombe ; si elle plonge aussi, c'est l'ECART entre les deux qu'il faut lire.
    if baseline is not None and len(baseline) == 2 and np.sum(baseline[0]) > 0:
        bn = np.asarray(baseline[0], dtype=float)
        bk = np.asarray(baseline[1], dtype=float)
        bp_tot = bk.sum() / bn.sum()
        if bn.shape == n_p.shape and (bn > 0).all():
            bp, blo, bhi = wilson(bk, bn)
            ax.fill_between(x, blo, bhi, color='grey', alpha=0.15, zorder=1)
            ax.plot(x, bp, 's--', color='grey', lw=1.6, ms=4, zorder=2,
                    label=f'baseline, no permutation ({bp_tot:.3f} overall)')
        else:                       # tranches indisponibles -> repli sur le niveau
            ax.axhline(bp_tot, color='grey', ls='--', lw=1.4, zorder=1,
                       label=f'baseline, no permutation ({bp_tot:.3f})')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{100*i//n_bins}–{100*(i+1)//n_bins}%\nn={int(v)}'
                        for i, v in zip(x, n_p)], fontsize=8)
    ax.set_xlabel(f"fraction of the agent's own lifetime")
    ax.set_ylabel(f'P(eat {label} | in field of view)')
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    # La MEDIANE est le signal le plus net : les pentes par agent reposent sur
    # peu d'evenements chacune, donc `frac_dec` bouge moins que la mediane meme
    # quand l'adaptation est franche. On affiche les deux.
    # variation RELATIVE signee : negative = l'agent mange moins en fin de vie
    first, last = p[ok][0], p[ok][-1]
    var = (last - first) / first if first > 0 else float('nan')
    # Meme variation pour la baseline : si elle chute autant, la baisse observee
    # n'est pas specifique a la permutation (echauffement du LSTM plutot
    # qu'apprentissage sur le poison).
    ligne_base = ''
    if baseline is not None and len(baseline) == 2:
        bn = np.asarray(baseline[0], dtype=float)
        bk = np.asarray(baseline[1], dtype=float)
        if bn.shape == n_p.shape and (bn > 0).all():
            bpc = bk / bn
            bvar = (bpc[-1] - bpc[0]) / bpc[0] if bpc[0] > 0 else float('nan')
            ligne_base = (f'\nbaseline {bpc[0]:.3f} → {bpc[-1]:.3f}'
                          + (f'  ({bvar:+.0%})' if np.isfinite(bvar) else ''))
    ax.text(0.02, 0.96,
            f'{first:.3f} → {last:.3f}'
            + (f'  ({var:+.0%})' if np.isfinite(var) else '')
            + f'\n{frac_dec:.0%} des {n_pentes} agents décroissent  (50% = hasard, z={z:+.1f})'
            + ligne_base,
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', alpha=0.85))
    ax.set_title(f'Within-life adaptation — {tag} (chunk {chunk})', fontsize=11,
                 pad=34 if mapping else 6)
    if mapping:
        ax.text(0.5, 1.005, mapping, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=8.5, family='monospace', color='dimgrey')

    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig', 'adapt', tag)
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'prob_eat_{label}_over_life_chunk_{chunk}.png'))
    plt.close()


def plot_memory_ablation(n_full, k_full, n_abl, k_abl, wilson, exp_dir, chunk, tag,
                         label='poison', mapping='', baseline=None):
    """Meme rollout avec et sans memoire intra-vie, le long de la vie.

    L'agent recoit (obs, last_action, reward, etat LSTM) : signature RL2. En
    coupant les trois canaux temporels sur LES MEMES genomes et LES MEMES cles,
    seule l'information intra-vie change.

    C'est le controle decisif :
      - la baisse disparait sans memoire -> c'est bien de l'apprentissage ;
      - elle persiste -> elle vient d'autre chose (depletion des ressources,
        physiologie), et l'interpretation "l'agent apprend" tombe.
    """
    nf, kf = np.asarray(n_full, float), np.asarray(k_full, float)
    na, ka = np.asarray(n_abl, float), np.asarray(k_abl, float)
    if nf.size == 0 or na.size == 0:
        return
    x = np.arange(len(nf))
    color = COLOR_BY_ID[[i for i, l in enumerate(LABELS) if l == label][0]]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    lignes = []
    for n_, k_, nom, sty, alpha in [(nf, kf, 'mémoire intacte', 'o-', 1.0),
                                    (na, ka, 'mémoire coupée', 's--', 0.6)]:
        ok = n_ > 0
        if not ok.any():
            continue
        p, lo, hi = wilson(k_, n_)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=0.18 * alpha, zorder=2)
        ax.plot(x[ok], p[ok], sty, color=color, lw=2.5, ms=6, alpha=alpha, zorder=4,
                label=nom)
        var = (p[ok][-1] - p[ok][0]) / p[ok][0] if p[ok][0] > 0 else float('nan')
        lignes.append(f'{nom:<16} {p[ok][0]:.3f} → {p[ok][-1]:.3f}'
                      + (f'  ({var:+.0%})' if np.isfinite(var) else ''))

    if baseline is not None and len(baseline) == 2 and np.sum(baseline[0]) > 0:
        bn, bk = np.asarray(baseline[0], float), np.asarray(baseline[1], float)
        if bn.shape == nf.shape and (bn > 0).all():
            ax.plot(x, bk / bn, ':', color='grey', lw=1.5, zorder=1,
                    label='baseline, no permutation')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{100*i//len(nf)}–{100*(i+1)//len(nf)}%\nn={int(v)}'
                        for i, v in zip(x, nf)], fontsize=8)
    ax.set_xlabel("fraction of the agent's own lifetime")
    ax.set_ylabel(f'P(eat {label} | in field of view)')
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.text(0.02, 0.96, '\n'.join(lignes), transform=ax.transAxes,
            ha='left', va='top', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', alpha=0.85))
    ax.set_title(f'Memory ablation — {tag} (chunk {chunk})', fontsize=11,
                 pad=34 if mapping else 6)
    if mapping:
        ax.text(0.5, 1.005, mapping, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=8.5, family='monospace', color='dimgrey')

    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig', 'adapt', tag)
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'memory_ablation_chunk_{chunk}.png'))
    plt.close()


def _excess_ci(n_p, k_p, n_b, k_b, z=1.96):
    """(exces, lo, hi) : difference de deux proportions et son IC 95%.

    exces = p_permute - p_baseline. Zero = l'agent se comporte comme sans
    permutation. L'IC est celui de la DIFFERENCE (variances additionnees), pas
    la juxtaposition des deux IC : c'est lui qui dit si l'ecart est reel.
    """
    n_p, k_p = np.asarray(n_p, float), np.asarray(k_p, float)
    n_b, k_b = np.asarray(n_b, float), np.asarray(k_b, float)
    ok = (n_p > 0) & (n_b > 0)
    d = np.full(n_p.shape, np.nan)
    lo = np.full(n_p.shape, np.nan)
    hi = np.full(n_p.shape, np.nan)
    pp = np.divide(k_p, n_p, out=np.zeros_like(n_p), where=ok)
    pb = np.divide(k_b, n_b, out=np.zeros_like(n_b), where=ok)
    se = np.sqrt(np.divide(pp * (1 - pp), n_p, out=np.zeros_like(n_p), where=ok)
                 + np.divide(pb * (1 - pb), n_b, out=np.zeros_like(n_b), where=ok))
    d[ok] = (pp - pb)[ok]
    lo[ok] = d[ok] - z * se[ok]
    hi[ok] = d[ok] + z * se[ok]
    return d, lo, hi


def plot_prob_eat_excess(per_type, baseline_per_type, exp_dir, chunk, tag,
                         mapping='', suffix='', titre=''):
    """Ecart `permute - non permute` par type, le long de la vie.

    Les agents mangent moins en vieillissant, quelle que soit la ressource : sur
    les courbes brutes les trois types descendent, ce qui empeche de conclure.
    Cette baisse d'appetit touche AUSSI la condition non permutee, donc la
    soustraction l'annule. Ce qui reste est imputable a la seule permutation.

    Zero = l'agent se comporte comme si de rien n'etait. Au-dessus = il mange ce
    qu'il ne devrait pas (il est piege) ; en dessous = il evite a tort une
    ressource devenue comestible. Un ecart qui se resorbe au fil de la vie EST
    l'adaptation recherchee.
    """
    ids = sorted(per_type)
    if not ids:
        return
    n_bins = len(next(iter(per_type.values()))[0])
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    lignes = []
    trace = False
    for i in ids:
        b = baseline_per_type.get(i)
        if b is None:
            continue
        d, lo, hi = _excess_ci(*per_type[i], *b)
        ok = np.isfinite(d)
        if not ok.any():
            continue
        trace = True
        c = COLOR_BY_ID[i]
        ax.fill_between(x[ok], lo[ok], hi[ok], color=c, alpha=0.18, zorder=2)
        ax.plot(x[ok], d[ok], 'o-', color=c, lw=2.5, ms=6, zorder=4,
                label=LABELS[i])
        # le rapport en annexe : "5x trop" parle mieux que "+0.113" quand les
        # niveaux absolus sont petits
        n_p, k_p = (np.asarray(v, float) for v in per_type[i])
        n_b, k_b = (np.asarray(v, float) for v in b)
        pb0 = k_b[ok][0] / n_b[ok][0] if n_b[ok][0] else np.nan
        pp0 = k_p[ok][0] / n_p[ok][0] if n_p[ok][0] else np.nan
        fois = f'  ({pp0/pb0:.1f}x)' if np.isfinite(pb0) and pb0 > 0 else ''
        lignes.append(f'{LABELS[i]:<7} {d[ok][0]:+.3f} → {d[ok][-1]:+.3f}{fois}')

    if not trace:
        plt.close(); return

    ax.axhline(0, color='black', lw=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{100*i//n_bins}–{100*(i+1)//n_bins}%' for i in x])
    ax.set_xlabel("fraction of the agent's own lifetime")
    ax.set_ylabel('P(eat X | X visible) : permuted − baseline')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.text(0.02, 0.96, '\n'.join(lignes), transform=ax.transAxes,
            ha='left', va='top', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', alpha=0.85))
    ax.set_title(f'Permutation-specific excess{titre} — {tag} (chunk {chunk})',
                 fontsize=11, pad=34 if mapping else 6)
    if mapping:
        ax.text(0.5, 1.005, mapping, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=8.5, family='monospace', color='dimgrey')

    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig', 'adapt', tag)
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'excess_over_life{suffix}_chunk_{chunk}.png'))
    plt.close()


RATIO_MIN_EVENTS = 10


def _ratio_ci(n_p, k_p, n_b, k_b, z=1.96, min_events=RATIO_MIN_EVENTS):
    """(rapport, lo, hi) : p_permute / p_baseline et son IC 95%.

    L'IC est calcule sur le LOG du rapport puis ramene par exponentielle. Le
    rapport n'est pas symetrique (2x et 0.5x sont deux ecarts egaux mais
    opposes) ; son log l'est, et c'est sur cette echelle que l'approximation
    normale tient. Par la methode delta, Var(log p) = (1-p)/(n p), soit ~1/k
    quand p est petit : c'est le NOMBRE D'EVENEMENTS, pas la taille de
    l'echantillon, qui fixe la precision.

    D'ou le masquage : sous `min_events` reussites d'un cote ou de l'autre, le
    rapport est domine par le bruit du denominateur et n'est pas trace. Un
    denominateur nul donnerait un rapport infini -- il vaut mieux un trou dans
    la courbe qu'un point faux.
    """
    n_p, k_p = np.asarray(n_p, float), np.asarray(k_p, float)
    n_b, k_b = np.asarray(n_b, float), np.asarray(k_b, float)
    ok = (n_p > 0) & (n_b > 0) & (k_p >= min_events) & (k_b >= min_events)
    r = np.full(n_p.shape, np.nan)
    lo = np.full(n_p.shape, np.nan)
    hi = np.full(n_p.shape, np.nan)
    if not ok.any():
        return r, lo, hi
    pp = k_p[ok] / n_p[ok]
    pb = k_b[ok] / n_b[ok]
    se = np.sqrt((1 - pp) / k_p[ok] + (1 - pb) / k_b[ok])
    lr = np.log(pp / pb)
    r[ok] = np.exp(lr)
    lo[ok] = np.exp(lr - z * se)
    hi[ok] = np.exp(lr + z * se)
    return r, lo, hi


def plot_prob_eat_ratio(per_type, baseline_per_type, exp_dir, chunk, tag,
                        mapping='', suffix='', titre=''):
    """Rapport `permute / non permute` par type, le long de la vie.

    Pourquoi un rapport et pas la difference de `plot_prob_eat_excess` : les
    agents mangent moins en vieillissant, et cette baisse d'appetit MULTIPLIE
    les deux conditions au lieu de leur retirer la meme quantite. Une
    soustraction n'annule qu'un effet additif ; face a une division elle laisse
    l'exces se contracter tout seul. Sur les figures ablatees -- ou la politique
    est pourtant une fonction de `obs` seule, donc figee -- les trois types se
    contractaient d'un facteur quasi identique (~0.24) : un seul facteur commun,
    pas trois adaptations.

    Le rapport, lui, est sans dimension : si l'environnement multiplie les deux
    conditions par f(t), f se simplifie et la courbe reste plate.

    1 = l'agent se comporte comme sans permutation. Au-dessus = il mange ce
    qu'il ne devrait pas (il est piege) ; en dessous = il evite a tort une
    ressource devenue comestible. Une courbe qui converge vers 1 au fil de la
    vie EST l'adaptation recherchee -- et cette fois la convergence ne peut plus
    venir d'un effondrement d'echelle.

    Contrepartie : quand la baseline est minuscule le rapport devient tres
    bruite. D'ou l'echelle log et le masquage de `_ratio_ci`.
    """
    ids = sorted(per_type)
    if not ids:
        return
    n_bins = len(next(iter(per_type.values()))[0])
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    lignes = []
    trace = False
    for i in ids:
        b = baseline_per_type.get(i)
        if b is None:
            continue
        r, lo, hi = _ratio_ci(*per_type[i], *b)
        ok = np.isfinite(r)
        if not ok.any():
            lignes.append(f'{LABELS[i]:<7} (trop peu d\'evenements)')
            continue
        trace = True
        c = COLOR_BY_ID[i]
        ax.fill_between(x[ok], lo[ok], hi[ok], color=c, alpha=0.18, zorder=2)
        ax.plot(x[ok], r[ok], 'o-', color=c, lw=2.5, ms=6, zorder=4,
                label=LABELS[i])
        # trous eventuels : on signale combien de tranches ont ete masquees
        manque = int((~ok).sum())
        note = f'  [{manque} tranche(s) masquee(s)]' if manque else ''
        lignes.append(f'{LABELS[i]:<7} {r[ok][0]:5.2f}x → {r[ok][-1]:5.2f}x{note}')

    if not trace:
        plt.close(); return

    ax.axhline(1, color='black', lw=1.2, zorder=3)
    ax.set_yscale('log')
    # graduations lisibles en facteurs, pas en puissances de 10
    ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4, 10, 25])
    ax.set_yticklabels(['0.1x', '0.25x', '0.5x', '1x', '2x', '4x', '10x', '25x'])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([f'{100*i//n_bins}–{100*(i+1)//n_bins}%' for i in x])
    ax.set_xlabel("fraction of the agent's own lifetime")
    ax.set_ylabel('P(eat X | X visible) : permuted / baseline')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.text(0.02, 0.96, '\n'.join(lignes), transform=ax.transAxes,
            ha='left', va='top', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', alpha=0.85))
    ax.set_title(f'Permutation-specific ratio{titre} — {tag} (chunk {chunk})',
                 fontsize=11, pad=34 if mapping else 6)
    if mapping:
        ax.text(0.5, 1.005, mapping, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=8.5, family='monospace', color='dimgrey')

    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig', 'adapt', tag)
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'ratio_over_life{suffix}_chunk_{chunk}.png'))
    plt.close()


def plot_prob_eat_over_life_by_type(per_type, baseline_per_type, wilson, exp_dir,
                                    chunk, tag, mapping=''):
    """P(manger X | X en vue) pour LES TROIS identites, le long de la vie.

    per_type / baseline_per_type : {id_ressource: (n, k)} par tranche de vie,
    pour l'env permute et pour l'env non permute.

    Repond a deux questions d'un coup :

      - la permutation deplace le poison sur le canal d'une autre ressource ;
        celle-ci est-elle davantage consommee en retour ?
      - la baisse du poison en fin de vie est-elle de la SELECTIVITE, ou juste
        un appetit general qui retombe (les agents mangent tot puis explorent) ?
        Si les trois courbes descendent ensemble, c'est l'appetit ; si seul le
        poison descend, c'est de la selectivite.

    C'est cette comparaison entre types qui tranche, et non un axe en "ressources
    deja mangees" : ce dernier est un cumul de la variable mesuree, donc ses bins
    de droite ne contiennent que les gros mangeurs et la courbe monte meme sans
    aucun apprentissage.
    """
    ids = sorted(per_type)
    if not ids:
        return
    n_bins = len(next(iter(per_type.values()))[0])
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    lignes = []
    for i in ids:
        n_i, k_i = (np.asarray(v, dtype=float) for v in per_type[i])
        ok = n_i > 0
        if not ok.any():
            continue
        p, lo, hi = wilson(k_i, n_i)
        c = COLOR_BY_ID[i]
        ax.fill_between(x[ok], lo[ok], hi[ok], color=c, alpha=0.18, zorder=2)
        ax.plot(x[ok], p[ok], 'o-', color=c, lw=2.5, ms=6, zorder=4,
                label=f'{LABELS[i]} (permuted)')
        var = (p[ok][-1] - p[ok][0]) / p[ok][0] if p[ok][0] > 0 else float('nan')
        lignes.append(f'{LABELS[i]:<7} {p[ok][0]:.3f} → {p[ok][-1]:.3f}'
                      + (f'  ({var:+.0%})' if np.isfinite(var) else ''))

        b = baseline_per_type.get(i)
        if b is not None:
            bn, bk = (np.asarray(v, dtype=float) for v in b)
            if (bn > 0).all():
                ax.plot(x, bk / bn, 's--', color=c, lw=1.3, ms=3.5, alpha=0.55,
                        zorder=3, label=f'{LABELS[i]} (baseline)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{100*i//n_bins}–{100*(i+1)//n_bins}%' for i in x])
    ax.set_xlabel("fraction of the agent's own lifetime")
    ax.set_ylabel('P(eat X | X in field of view)')
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=7.5, ncol=2)
    ax.text(0.02, 0.96, '\n'.join(lignes), transform=ax.transAxes,
            ha='left', va='top', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', alpha=0.85))
    ax.set_title(f'Selectivity over life — {tag} (chunk {chunk})', fontsize=11,
                 pad=34 if mapping else 6)
    if mapping:
        ax.text(0.5, 1.005, mapping, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=8.5, family='monospace', color='dimgrey')

    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig', 'adapt', tag)
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'selectivity_over_life_chunk_{chunk}.png'))
    plt.close()


def _eaten_by_identity(eaten, ids_by_channel):
    """Reindexe (B, n_types) de CANAL vers IDENTITE de ressource.

    Indispensable : la rotation change ce que porte chaque canal, donc comparer
    canal a canal comparerait des ressources differentes."""
    eaten = np.asarray(eaten, dtype=float)
    return {int(i): eaten[:, k] for k, i in enumerate(ids_by_channel)}


def plot_eaten_by_type_boxplot(eaten, ids_by_channel, exp_dir, chunk, tag,
                               mapping='', available=None,
                               baseline=None, baseline_ids=None,
                               baseline_available=None):
    """Boxplot du nombre de ressources mangees par agent, une boite par type.

    eaten          : (B, n_types) total mange par chaque agent, indexe par CANAL
    ids_by_channel : id de ressource porte par chaque canal DANS CETTE ROTATION
    available      : dict {id_ressource: presentes sur la grille} -> plafond
    baseline       : (B, n_types) idem pour l'env NON permute (rot0), apparie
                     genome par genome ; baseline_ids = ses ids par canal
    baseline_available : plafond de l'env baseline (peut differer de `available`,
                     la croissance a l'init dependant du canal)
    """
    by_id = _eaten_by_identity(eaten, ids_by_channel)
    base_by_id = (_eaten_by_identity(baseline, baseline_ids)
                  if baseline is not None else None)

    ids = sorted(by_id)                                  # good, medium, poison
    labels = [LABELS[i] for i in ids]
    B = np.asarray(eaten).shape[0]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    rng = np.random.default_rng(0)

    def draw(series, positions, alpha, hatch):
        bp = ax.boxplot(series, positions=positions, widths=0.3,
                        patch_artist=True, showmeans=True, manage_ticks=False,
                        medianprops=dict(color='black', lw=2))
        for patch, i in zip(bp['boxes'], ids):
            patch.set_facecolor(COLOR_BY_ID[i])
            patch.set_alpha(alpha)
            if hatch:
                patch.set_hatch(hatch)
        # nuage de points : avec peu d'agents la boite seule cache la distribution
        for x, col in zip(positions, series):
            ax.plot(x + rng.normal(0, 0.035, len(col)), col, 'o', ms=2.5,
                    color='black', alpha=0.3, zorder=3)

    def ceiling(avail, positions):
        """Un plafond AU-DESSUS DE CHAQUE boite : baseline et env permute n'ont
        pas forcement le meme disponible."""
        if avail is None:
            return
        for x, i in zip(positions, ids):
            ax.hlines(avail[i], x - 0.16, x + 0.16, color='grey',
                      ls='--', lw=1.2, zorder=2)

    centers = np.arange(len(ids), dtype=float)
    if base_by_id is None:
        draw([by_id[i] for i in ids], centers, 0.55, None)
        ceiling(available, centers)
    else:
        draw([base_by_id[i] for i in ids], centers - 0.19, 0.25, '//')
        draw([by_id[i] for i in ids],      centers + 0.19, 0.65, None)
        ceiling(baseline_available, centers - 0.19)
        ceiling(available,          centers + 0.19)
        ax.plot([], [], 's', color='grey', alpha=0.35, markersize=9,
                markeredgecolor='black', label='baseline (rot0, no permutation)')
        ax.plot([], [], 's', color='grey', alpha=0.8, markersize=9,
                markeredgecolor='black', label=f'{tag} (permuted)')

    if available is not None:
        ax.plot([], [], color='grey', ls='--', lw=1.2, label='available on the grid')

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(ids) - 0.4)
    ax.set_ylabel('Resources eaten per agent')
    # marge en haut : sinon les plafonds les plus hauts passent sous la legende
    tops = [np.max(v) for v in by_id.values()]
    for av in (available, baseline_available):
        if av is not None:
            tops += list(av.values())
    ax.set_ylim(0, max(tops) * 1.22)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Resources eaten by type — {tag} (chunk {chunk}, n={B})',
                 fontsize=11, pad=34 if mapping else 6)
    if mapping:
        # monospace : les colonnes ch0/ch1/ch2 des deux lignes doivent s'aligner
        ax.text(0.5, 1.005, mapping, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8.5, family='monospace', color='dimgrey')

    plt.tight_layout()
    path = os.path.join(exp_dir, 'fig', 'adapt', tag)
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'eaten_by_type_chunk_{chunk}.png'))
    plt.close()


BURN_IN_STEPS = 2000      # transitoire initial ecarte du portrait de phase


def plot_phase_portrait_png(pop_history, res_history, exp_dir, cfg, start_step=0,
                            fixed_point=None, steps=None):
    res     = np.asarray(res_history)                      # (T, n_types)
    delta_e = np.array([r.delta_energy for r in cfg.resources])
    good    = np.where(delta_e > 0)[0]
    R_full  = res[:, good].sum(axis=1)                     # (T,) total bonnes ressources

    # Le burn-in se compte en STEPS, pas en indices : sur des donnees agregees par
    # blocs un indice ne vaut plus un pas, et decouper a l'indice 2000 jetterait
    # tout le run.
    pos = (np.asarray(steps) if steps is not None
           else np.arange(start_step, start_step + len(pop_history)))
    garde = pos >= BURN_IN_STEPS
    if garde.sum() < 2:
        return

    R = R_full[garde]
    N = np.asarray(pop_history)[garde]
    steps = pos[garde]

    # Segments consécutifs pour colorer la ligne par le temps
    points = np.array([R, N]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))

    lc = LineCollection(segments, cmap='viridis', linewidth=2)
    lc.set_array(steps[:-1])           # couleur = step
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Step')

    # Début (cercle) et fin (croix)
    ax.scatter(R[0], N[0], color='black', s=60, zorder=3, label='Start')
    ax.scatter(R[-1], N[-1], color='red', marker='X', s=80, zorder=3, label='End')

    # Point fixe théorique si fourni : (R*, N*)
    if fixed_point is not None:
        ax.scatter(*fixed_point, color='blue', marker='*', s=200,
                   zorder=3, label='Fixed point')

    ax.set_xlabel('Resources amount')
    ax.set_ylabel('Population size')
    ax.set_title('Phase portrait (resources vs population)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.autoscale()                     # LineCollection ne fixe pas les limites seule

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, 'plot_phase.png'))
    plt.close()
    
            

def plot_evolution_html(pop_history, res_history, exp_dir, shuffle_log,
                        initial_order_ids, start_step=0, steps=None):
    generations = (np.asarray(steps) if steps is not None
                   else np.arange(start_step, start_step + len(pop_history)))
    T = len(generations)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=generations, y=pop_history, name='Agents',
                   line=dict(color='red', width=2)),
        secondary_y=False,
    )

    res = np.asarray(res_history)
    if res.ndim == 1:
        res = res[:, None]
    n_types = res.shape[1]

    id_timeline = build_id_timeline(generations, shuffle_log, initial_order_ids)  # (T, n_types)

    # rgba par id (une seule fois)
    def rgba(i):
        r, g, b, a = mcolors.to_rgba(COLOR_BY_ID[i])
        return f"rgba({r*255:.0f},{g*255:.0f},{b*255:.0f},{a:.2f})"

    seen = set()                                  # pour n'afficher chaque id qu'une fois en légende
    for k in range(n_types):
        ids_k = id_timeline[:, k]
        # bornes des runs où l'id est constant
        change = np.where(np.diff(ids_k) != 0)[0] + 1
        bounds = [0, *change.tolist(), T]

        for s, e in zip(bounds[:-1], bounds[1:]):
            i   = int(ids_k[s])
            sl  = slice(s, min(e + 1, T))         # +1 point pour raccorder les segments sans trou
            show = i not in seen
            seen.add(i)
            fig.add_trace(
                go.Scatter(
                    x=generations[sl], y=res[sl, k],
                    name=LABELS[i], legendgroup=LABELS[i], showlegend=show,
                    line=dict(color=rgba(i), width=2),
                ),
                secondary_y=True,
            )

    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='Population size', secondary_y=False,
                     title_font=dict(color='red'), tickfont=dict(color='red'))
    fig.update_yaxes(title_text='Resources amount', secondary_y=True,
                     title_font=dict(color='green'), tickfont=dict(color='green'))
    fig.update_layout(title='Simulation dynamic')

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, 'plot_evo.html'))


def plot_current_config(current_grid_res, grid_walls,pos, alive, exp_dir, resources,name_fig='sim'):
    plot_current_config_png(current_grid_res, grid_walls,pos, alive, exp_dir,resources, name_fig)

def plot_current_config_png(current_grid_res, grid_walls, pos, alive, exp_dir, resources, name_fig='sim'):
    fig, ax = plt.subplots(figsize=(5, 5))

    res = np.asarray(current_grid_res)                    # (n_types, L, L)
    n_types = res.shape[0]

    # repli en grille de labels : 0 = vide, k+1 = type k
    present = res.sum(axis=0) > 0                          # (L, L) : y a-t-il une ressource ?
    label   = np.where(present, res.argmax(axis=0) + 1, 0)  # (L, L)

    # une couleur par type (indice 0 = blanc = vide)
    ids = [r.id for r in resources]
    colors = [COLOR_BY_ID[id] for id in ids]
    colors = ["white"] + colors
    cmap_res = mcolors.ListedColormap(colors[:n_types + 1])

    # bornes décalées de 0.5 -> chaque entier tombe au centre de sa couleur
    ax.imshow(label, cmap=cmap_res, vmin=-0.5, vmax=n_types + 0.5,
              interpolation="nearest")

    
    
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    _,grid_h, grid_w = current_grid_res.shape
    cell_px = min(bbox.width / grid_w, bbox.height / grid_h)
    cell_pts = cell_px * 72 / fig.dpi

    for i in range(len(alive)):
        if alive[i]:
            ax.plot(pos[i, 1], pos[i, 0], 's',
                    color="red",
                    markersize=cell_pts)

    ax.set_title('Configuration')
    ax.axis("off")

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig/sim_grid/')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_sim_chunk_{name_fig}.png'))
    plt.close()

def plot_current_config_html(current_grid_res, pos, alive, exp_dir, name_fig=''):
    """Plot the initial grid configuration with agent positions."""
    grid_h, grid_w = current_grid_res.shape

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=current_grid_res,
        colorscale=[[0, 'white'], [1, '#4CAF50']],
        zmin=0, zmax=1,
        showscale=False,
    ))

    x_agents = [pos[i, 1] for i in range(len(alive)) if alive[i]]
    y_agents = [pos[i, 0] for i in range(len(alive)) if alive[i]]
    colors_agents = ['purple' if i == 1 else 'red' for i in range(len(alive)) if alive[i]]

    fig.add_trace(go.Scatter(
        x=x_agents, y=y_agents,
        mode='markers',
        marker=dict(symbol='square', color=colors_agents, size=10),
        name='Agents',
    ))

    fig.update_layout(
        title='Configuration initiale (t=0)',
        xaxis=dict(showticklabels=False, range=[-0.5, grid_w - 0.5]),
        yaxis=dict(showticklabels=False, scaleanchor='x', range=[grid_h - 0.5, -0.5]),
        width=500, height=500,
    )

    path_save_fig = os.path.join(exp_dir, 'fig/sim_grid/')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_sim_chunk_{name_fig}.html'))
    





def plot_full_genealogy_robust(outputs):
    agents = outputs.agents
    parent_ids = np.array(agents.parent_id)
    born_steps = np.array(agents.born_step)
    alive = np.array(agents.alive)
    
    T, N = born_steps.shape
    
    nodes_by_step = {}
    edges = set()
    
    def add_node(idx, step):
        if step not in nodes_by_step:
            nodes_by_step[step] = set()
        nodes_by_step[step].add(idx)
        
    # 1. Capture des agents initiaux (t=0)
    for idx in range(1, N):
        if alive[0, idx] == 1:
            add_node(idx, born_steps[0, idx])
            
    # 2. Détection infaillible de toutes les naissances
    for t in range(1, T):
        # Une naissance = le step de naissance a changé par rapport à l'étape précédente
        changed_indices = np.where(born_steps[t] != born_steps[t-1])[0]
        
        for idx in changed_indices:
            if idx == 0:
                continue # On exclut l'agent 0 mort par défaut
                
            c_b_step = born_steps[t, idx]
            p_idx = parent_ids[t, idx]
            
            if p_idx == 0:
                continue
                
            p_b_step = born_steps[t, p_idx]
            
            # Ajouter l'enfant et le parent
            add_node(idx, c_b_step)
            add_node(p_idx, p_b_step)
            
            # Créer le lien généalogique
            edges.add((f"A{p_idx}_T{p_b_step}", f"A{idx}_T{c_b_step}"))

    # 3. Identification des survivants finaux pour le style
    survivors = set()
    last_step = T - 1
    for idx in range(1, N):
        if alive[last_step, idx] == 1:
            survivors.add((idx, born_steps[last_step, idx]))

    # 4. Tracé Graphviz
    dot = graphviz.Digraph(engine='dot', format='pdf')
    dot.attr(rankdir='LR', nodesep='0.15', ranksep='0.4')
    
    if not nodes_by_step:
        return dot
        
    max_step = max(nodes_by_step.keys())
    cmap = plt.cm.Blues_r
    norm = mcolors.Normalize(vmin=0, vmax=max_step if max_step > 0 else 1)

    for step, indices in nodes_by_step.items():
        rgba = cmap(norm(step))
        hex_color = mcolors.to_hex(rgba)
        
        with dot.subgraph() as s:
            s.attr(rank='same')
            for idx in indices:
                node_id = f"A{idx}_T{step}"
                node_label = f"A_{idx}\nStep {step}"
                
                is_survivor = (idx, step) in survivors
                border_color = '#2ca02c' if is_survivor else 'black' # Vert si survivant
                pen_width = '3' if is_survivor else '1'
                
                s.node(node_id, 
                       label=node_label, 
                       style='filled,rounded', 
                       fillcolor=hex_color, 
                       color=border_color,
                       penwidth=pen_width,
                       shape='box', 
                       fontsize='9', 
                       fontcolor='white' if step < max_step/2 else 'black')

    for parent, child in edges:
        dot.edge(parent, child, arrowsize='0.6')

    dot.render('arbre_genealogique_complet', view=True)
    return dot


### MOVEMENT
def compute_mean_movement_chunk(outputs, n):
    """Déplacement moyen (en cellules) par transition, moyenné sur les agents.
    Ne compte QUE les vraies transitions intra-vie (même individu vivant aux 2 pas).
    Grille bornée : pas de correction toroïdale."""
    positions = np.array(outputs.position)   # (T, N, 2)
    alive     = np.array(outputs.alive)      # (T, N)
    born      = np.array(outputs.born_step)  # (T, N)

    delta     = positions[1:] - positions[:-1]           # (T-1, N, 2)
    magnitude = np.sqrt((delta ** 2).sum(axis=-1))       # (T-1, N)

    # même individu, vivant à t ET t+1  ->  exclut naissances, morts, recyclages
    same = (alive[:-1] == 1) & (alive[1:] == 1) & (born[:-1] == born[1:])   # (T-1, N)

    n_valid  = same.sum(axis=1)
    mean_mov = np.where(n_valid > 0,
                        (magnitude * same).sum(axis=1) / n_valid, 0.0)
    return mean_mov   # (T-1,)

def plot_mean_movement(mov_history, exp_dir, start_step=0, name_fig='sim', steps=None):
    plot_mean_movement_png(mov_history, exp_dir, start_step=start_step, name_fig=name_fig, steps=steps)
    plot_mean_movement_html(mov_history, exp_dir, start_step=start_step, name_fig=name_fig, steps=steps)

def plot_mean_movement_png(mov_history, exp_dir, start_step=0, name_fig='sim', steps=None):
    """Plot mean movement of the population over time."""
    steps = (np.asarray(steps) if steps is not None
             else np.arange(start_step, start_step + len(mov_history)))

    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Mean movement (cell)')
    ax.plot(steps, mov_history, color='tab:blue', linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Mean population movement')

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_movement_{name_fig}.png'))
    plt.close()

def plot_mean_movement_html(mov_history, exp_dir,
                            start_step=0, name_fig='sim', steps=None):
    """Plot mean movement, points colored by resource presence."""
    steps = (np.asarray(steps) if steps is not None
             else np.arange(start_step, start_step + len(mov_history)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=mov_history,
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            showscale=True,
            size=5,
            colorbar=dict(title='Resources'),
        ),
        name='Mean movement',
    ))
    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='Mean movement (steps)')
    fig.update_layout(title='Mean population movement')

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_movement_{name_fig}.html'))



    
## LIFETIME
def compute_lifetime_chunk(outputs, cfg):
    alive = np.array(outputs.alive)
    born  = np.array(outputs.born_step)
    step  = np.array(outputs.step)

    a_t, a_tp1 = alive[:-1], alive[1:]
    b_t, b_tp1 = born[:-1], born[1:]

    death_event = (a_t == 1) & ((a_tp1 == 0) | (b_tp1 != b_t))

    death_step = np.broadcast_to(step[:-1][:, None], death_event.shape)[death_event]
    ages       = (step[:-1][:, None] - b_t)[death_event]
    return ages,death_step         # deux tableaux 1D, longueur = nb de morts

def plot_lifetime_vs_step(death_steps, lifetimes, exp_dir, cfg,name_fig='sim'):
    plot_lifetime_vs_step_png(death_steps, lifetimes, exp_dir,cfg, name_fig)
    plot_lifetime_vs_step_html(death_steps, lifetimes, exp_dir,cfg, name_fig)

def plot_lifetime_vs_step_png(death_steps, lifetimes, exp_dir, cfg,name_fig='sim'):
    death_steps = np.asarray(death_steps, dtype=float)
    lifetimes   = np.asarray(lifetimes, dtype=float)

    time_to_die_without_eating = cfg.starting_energy/cfg.energy_decay + cfg.time_to_die
    
    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Death step')
    ax.set_ylabel('Age at death time (steps)')
    ax.scatter(death_steps, lifetimes, s=4, alpha=0.8,
               color='tab:green', edgecolors='none')
    ax.grid(True, alpha=0.3)
    ax.set_title("Agents age at death time")
    ax.axhline(time_to_die_without_eating, color='tab:red', linewidth=1,
               linestyle='--', label=f'Minimum age ({time_to_die_without_eating:.0f} steps)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_lifetime_scatter_{name_fig}.png'), dpi=120)
    plt.close()

def plot_lifetime_vs_step_html(death_steps, lifetimes, exp_dir,cfg, name_fig='sim'):
    death_steps = np.asarray(death_steps, dtype=float)
    lifetimes   = np.asarray(lifetimes, dtype=float)

    nbins   = 100
    edges   = np.linspace(death_steps.min(), death_steps.max(), nbins + 1)
    time_to_die_without_eating = cfg.starting_energy/cfg.energy_decay + cfg.time_to_die

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=death_steps, y=lifetimes,
        mode='markers',
        marker=dict(color='green', size=5, opacity=0.8),
        name='Death age',
    ))

    fig.add_hline(y=time_to_die_without_eating, line_dash='dash', line_color='red',
                  annotation_text=f'Minimum age ({time_to_die_without_eating:.0f} steps)',
                  annotation_position='top left')

    fig.update_xaxes(title_text='Death Step')
    fig.update_yaxes(title_text='Death age (steps)')
    fig.update_layout(title="Agents age at death time")

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_lifetime_scatter_{name_fig}.html'))
    
    
## LIFE EXPECTANCY
def compute_life_expectancy(death_steps, lifetimes, bin_width=100):
    """Médiane de survie par cohorte : regroupe par step de naissance
    (largeur bin_width) et renvoie (centres, âge médian à la mort, effectif)."""
    death_steps = np.asarray(death_steps, dtype=float)
    lifetimes   = np.asarray(lifetimes, dtype=float)
    born        = death_steps - lifetimes

    edges  = np.arange(born.min(), born.max() + bin_width, bin_width)
    n_bins = len(edges) - 1
    idx    = np.clip(np.digitize(born, edges) - 1, 0, n_bins - 1)   # bin de chaque individu

    med_age = np.full(n_bins, np.nan)
    count   = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        vals = lifetimes[idx == b]
        count[b] = vals.size
        if vals.size:
            med_age[b] = np.median(vals)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, med_age, count
 
def plot_life_expectancy(death_steps, lifetimes, exp_dir, bin_width=100, name_fig='sim'):
    plot_life_expectancy_png(death_steps, lifetimes, exp_dir, bin_width, name_fig)
    plot_life_expectancy_html(death_steps, lifetimes, exp_dir, bin_width, name_fig)

def plot_life_expectancy_png(death_steps, lifetimes, exp_dir, bin_width=10, name_fig='sim'):
    centers, med_age, _ = compute_life_expectancy(death_steps, lifetimes, bin_width)
    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Born step')
    ax.set_ylabel('Life expectancy (steps)')
    ax.scatter(centers, med_age, color='tab:purple')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Life expectancy with lifetime average over {bin_width} agents")

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_life_exp_{name_fig}.png'), dpi=120)
    plt.close()

def plot_life_expectancy_html(death_steps, lifetimes, exp_dir, bin_width=10, name_fig='sim'):
    centers, med_age, _ = compute_life_expectancy(death_steps, lifetimes, bin_width)


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=centers, y=med_age,
        mode='markers',
        marker=dict(color='purple', size=5, opacity=0.8),
        name='Born step',
    ))
    fig.update_xaxes(title_text='Born step')
    fig.update_yaxes(title_text='Life expectancy (steps)')
    fig.update_layout(title=f"Life expectancy with lifetime average over {bin_width} agents")

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_life_exp_{name_fig}.html'))
    



# =====================================================================
#  Bande de dispersion — remplace moy ± std
# =====================================================================
BAND = "iqr"          # "minmax" | "iqr" | "std"   <- change ici partout



def _get(S, prefix, key, fallback=None):
    """Lecture tolerante : les resumes ecrits AVANT l'ajout des quantiles n'ont
    que moy/std. Cle absente -> NaN (bande non tracee), avec repli optionnel
    sur une autre cle pour la ligne centrale (on garde l'historique)."""
    out = []
    for s in S:
        v = s.get(f"{prefix}_{key}")
        if v is None and fallback is not None:
            v = s.get(f"{prefix}_{fallback}")
        out.append(np.nan if v is None else float(v))
    return np.array(out, dtype=float)
 
 
def _band(S, prefix, band=None):
    """Retourne (centre, bas, haut) pour tracer une mesure au fil des chunks.
 
      "minmax" : enveloppe totale [min, max]  — jamais hors du support, mais
                 s'elargit mecaniquement avec le nombre d'agents.
      "iqr"    : [p25, p75], 50% central      — borne ET stable avec n.
      "std"    : moy ± 1σ                     — peut sortir du support.
 
    Le centre est la MEDIANE pour minmax/iqr (coherent avec des quantiles,
    robuste aux queues), la moyenne pour std. Sur un chunk trop ancien pour
    avoir les quantiles, le centre retombe sur la moyenne.
    """
    band = band or BAND
    if band == "std":
        m = _get(S, prefix, "moy")
        return m, m - _get(S, prefix, "std"), m + _get(S, prefix, "std")
    m = _get(S, prefix, "p50", fallback="moy")
    if band == "iqr":
        return m, _get(S, prefix, "p25"), _get(S, prefix, "p75")
    return m, _get(S, prefix, "min"), _get(S, prefix, "max")
 
 
def _plot_band(ax, x, S, prefix, color="C0", label="median", band=None):
    m, lo, hi = _band(S, prefix, band)
    ok = ~np.isnan(m)
    if ok.any():
        ax.plot(x[ok], m[ok], marker="o", color=color, label=label)
    okb = ~(np.isnan(lo) | np.isnan(hi))          # chunks sans quantiles : pas de bande
    if okb.any():
        ax.fill_between(x[okb], lo[okb], hi[okb], alpha=0.22, color=color,
                        label={"std": "±1σ", "iqr": "p25–p75"}.get(band or BAND, "min–max"))
    return m
 
 
def plot_lab_metrics(exp_dir, suffix=""):
    """Évolution des métriques d'un env de lab, chunk après chunk.
    suffix="" -> high_res ; suffix="adapt_rot1" -> l'env adapt rot1, etc."""
    data_dir = os.path.join(exp_dir, "lab_data")
    fig_dir  = os.path.join(exp_dir, "fig")

    tag = f"_{suffix}" if suffix else ""
    pattern = rf"chunk_\d+{re.escape(tag)}_summary\.json"        # match EXACT de cette famille

    files = sorted(glob.glob(os.path.join(data_dir, f"chunk_*{tag}_summary.json")),
                   key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1)))
    files = [f for f in files
             if re.fullmatch(pattern, os.path.basename(f))]      # exclut les autres suffixes
    if not files:
        print(f"No summary to plot (suffix={suffix!r}).")
        return
 
    S = [json.load(open(f)) for f in files]
    x = np.array([s["chunk"] for s in S])
 
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
 
    specs = [
        (axes[0, 0], "duree_vie",    "Lifespan",              "steps"),
        (axes[0, 1], "consommation", "Mean intake",           "/step"),
        (axes[0, 2], "mouvement",    "Mean motion",           "/step"),
        (axes[1, 0], "greediness",   "Greediness  G = Cr/Tr", "ratio"),
    ]
    for ax, prefix, title, unit in specs:
        _plot_band(ax, x, S, prefix)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
    axes[1, 0].set_ylim(0, 1)          # G est un ratio borne
 
    # Mortalité : fractions, pas de dispersion (ce sont deja des proportions)
    ax = axes[1, 1]
    mur  = np.array([s["frac_mort_mur"]  for s in S])
    faim = np.array([s["frac_mort_faim"] for s in S])
    # deduit des donnees et non de cfg : marche aussi sur les runs deja faits
    if np.any(mur > 0):
        ax.plot(x, mur, marker="o", color="C3", label="wall deaths")
    ax.plot(x, faim, marker="s", color="C1", label="starvation deaths")
    ax.set_title("Cause of death (fraction of tested agents)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
 
    # Score d'adaptation : Δe moyen de ce qui est mange moins Δe moyen de ce qui
    # est vu. 0 = mange sans choisir, >0 = choisit mieux que l'offre. Absent des
    # runs anterieurs a son ajout -> case laissee vide.

    # Gain net par pas de faim : signe conserve, donc le poison coute. Mesure la
    # QUANTITE recoltee la ou elle sert, quand adapt_score compare des
    # compositions. Les deux sont traces cote a cote pour pouvoir diverger.
    ax = axes[1, 2]
    if any("adapt_gain_p50" in s for s in S):
        _plot_band(ax, x, S, "adapt_gain")
        ax.axhline(0, color="black", lw=1)
        ax.set_title("Net gain / hungry step")
        ax.set_ylabel("energy / step")
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    for ax in axes.ravel():
        ax.set_xlabel("chunk")
    axes[0, 0].legend(loc="best")
 
    fig.tight_layout()
    out = os.path.join(fig_dir, f"lab_metrics_evolution{tag}.png")   # nom de sortie distinct
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")
 
 
def _test_signes(v):
    """k positifs sur m non nuls, et p bilateral exact (binomiale 1/2)."""
    from math import comb
    v = v[np.isfinite(v)]
    nz = v[v != 0]
    m = nz.size
    if m == 0:
        return 0, 0, float("nan")
    k = int((nz > 0).sum())
    d = abs(k - m / 2)
    p = sum(comb(m, i) for i in range(m + 1) if abs(i - m / 2) >= d) / 2**m
    return k, m, min(p, 1.0)


def _nulle_somme(gain, n_perm=5000, seed=0):
    """Test apparie par retournement de signes, sur la SOMME des ecarts.

    Sous l'hypothese nulle, le signe de chaque ecart apparie est arbitraire.
    Les amplitudes sont conservees, donc une dispersion qui grandit au fil du
    run gonfle la nulle autant que l'observe.

    La somme et non le max : le max ne repose que sur UN genome, et il suffit
    qu'il garde son signe (une chance sur deux) pour que la nulle l'egale. Un
    effet porte par plusieurs genomes n'y est jamais detectable.
    """
    g = gain[np.isfinite(gain)]
    g = g[g != 0]
    if g.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sommes = (g * rng.choice([-1.0, 1.0], size=(n_perm, g.size))).sum(axis=1)
    obs = float(g.sum())
    return obs, float((sommes >= obs).mean())


def plot_memory_gain_hist(exp_dir, tag="memory", suffix="", env_titre="",
                          metric="age", bins=30):
    """Distribution, par genome, de (avec memoire - sans memoire).

    La mediane et l'IQR effacent les queues : un effet reel sur une poignee de
    genomes y est invisible. L'histogramme le montre.
    """
    data_dir = os.path.join(exp_dir, "lab_data")
    fig_dir  = os.path.join(exp_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(data_dir, f"chunk_*_{tag}_pergenome.npz")),
                   key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1)))
    files = [f for f in files
             if re.fullmatch(rf"chunk_\d+_{re.escape(tag)}_pergenome\.npz",
                             os.path.basename(f))]
    if not files:
        print(f"No per-genome memory data to plot (tag={tag!r}).")
        return

    cle = f"gain_{metric}"
    chunks, series = [], []
    for f in files:
        d = np.load(f)
        if cle not in d.files:
            continue
        chunks.append(int(re.search(r"chunk_(\d+)", f).group(1)))
        series.append(np.asarray(d[cle], dtype=float))
    if not series:
        print(f"No '{cle}' in per-genome files.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # 1) histogramme du dernier chunk
    ax = axes[0]
    v = series[-1][np.isfinite(series[-1])]
    if v.size:
        ax.hist(v, bins=bins, color="C0", alpha=.8)
        ax.axvline(0, color="black", lw=1.5)
        ax.axvline(np.median(v), color="C3", lw=1.5, ls="--",
                   label=f"mediane {np.median(v):+.1f}")
        ax.legend(loc="best")
    k, m, p = _test_signes(series[-1])
    ax.set_title(f"chunk {chunks[-1]} — n={v.size}\n"
                 f"signes : {k}/{m} positifs, p={p:.3f}")
    ax.set_xlabel(f"{metric} : avec memoire − sans")
    ax.set_ylabel("nb de genomes")

    # 2) tous les chunks empiles : une queue rare ressort ici
    ax = axes[1]
    tout = np.concatenate([x[np.isfinite(x)] for x in series]) if series else np.array([])
    if tout.size:
        ax.hist(tout, bins=bins * 2, color="C4", alpha=.8)
        ax.axvline(0, color="black", lw=1.5)
        pos = float((tout > 0).mean())
        ax.set_title(f"tous chunks — n={tout.size}, {100*pos:.0f}% > 0")
    ax.set_xlabel(f"{metric} : avec memoire − sans")

    # 3) evolution : mediane, IQR, et les extremes
    ax = axes[2]
    x   = np.array(chunks, dtype=float)
    med = np.array([np.nanmedian(v) if np.isfinite(v).any() else np.nan for v in series])
    p25 = np.array([np.nanpercentile(v, 25) if np.isfinite(v).any() else np.nan for v in series])
    p75 = np.array([np.nanpercentile(v, 75) if np.isfinite(v).any() else np.nan for v in series])
    mx  = np.array([np.nanmax(v) if np.isfinite(v).any() else np.nan for v in series])
    pvals = []
    for f in files:
        d = np.load(f)
        if cle in d.files:
            pvals.append(_nulle_somme(np.asarray(d[cle], float))[1])
        else:
            pvals.append(float("nan"))

    ok = ~np.isnan(med)
    if ok.any():
        ax.fill_between(x[ok], p25[ok], p75[ok], alpha=.25, color="C0", label="p25-p75")
        ax.plot(x[ok], med[ok], marker="o", color="C0", label="mediane")
        ax.plot(x[ok], mx[ok], marker=".", ls=":", color="C3", label="max")
    ax.axhline(0, color="black", lw=1.5)
    dernier_p = pvals[-1] if pvals else float("nan")
    ax.set_title(f"evolution du gain — p(somme) = {dernier_p:.3f} au dernier chunk")
    ax.set_xlabel("chunk")
    ax.legend(loc="best")

    for a in axes:
        a.grid(alpha=.3)
    fig.suptitle(f"Gain de la memoire, par genome — {metric} — {env_titre}", y=1.02)
    fig.tight_layout()
    out = os.path.join(fig_dir, f"lab_memory_gain_hist_{metric}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")


def plot_replay_top_gain(data_dir, chunk, suffix="", fig_dir=None):
    """Les genomes au plus fort gain, rejoues sur beaucoup de graines.

    Boite centree sur du positif = la memoire aide vraiment ce genome.
    Boite centree sur zero = l'ecart initial venait d'une seule action qui a
    bascule, pas d'un apprentissage.
    """
    f = os.path.join(data_dir, f"chunk_{chunk}_replay{suffix}.npz")
    if not os.path.exists(f):
        print(f"No replay data at {f}")
        return
    d = np.load(f)
    gains, obs, gen = d["gains"], d["observe"], d["genome"]

    fig, ax = plt.subplots(figsize=(1.6 * len(obs) + 4, 5))
    donnees = [g[np.isfinite(g)] for g in gains]
    pos = np.arange(1, len(donnees) + 1)
    garde = [i for i, g in enumerate(donnees) if g.size]
    if garde:
        ax.boxplot([donnees[i] for i in garde], positions=pos[garde], widths=.6,
                   flierprops=dict(marker=".", markersize=4, alpha=.6))
    ax.plot(pos, obs, "o", color="C3", ms=8, zorder=5,
            label="gain observe au depart")
    ax.axhline(0, color="black", lw=1.5)
    for i, g in enumerate(donnees):
        if g.size:
            ax.annotate(f"{100*(g > 0).mean():.0f}% > 0",
                        (pos[i], ax.get_ylim()[0]), ha="center", va="bottom",
                        fontsize=8, color="0.3")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"genome {int(b)}" for b in gen], fontsize=9)
    ax.set_ylabel("lifespan : with memory − without")
    ax.set_title(f"Top-gain genomes replayed over {gains.shape[1]} seeds "
                 f"— chunk {chunk}{suffix}")
    ax.legend(loc="best")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig_dir = fig_dir or data_dir
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"replay_top_gain_chunk_{chunk}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")


CORR_METRIQUES = ["age", "mean_rew", "mean_speed", "energy_end",
                  "greediness", "adapt_gain"]
CORR_TITRES = {"age": "lifespan", "mean_rew": "intake /step",
               "mean_speed": "motion /step", "energy_end": "final energy",
               "greediness": "greediness", "adapt_gain": "net gain /step"}


def plot_invasion(pop, oracles, exp_dir, start_step=0, steps=None,
                  invasion_start=None, cible=None):
    """Frequence des envahisseurs au cours du temps.

    C'est LA figure du test : si la frequence monte, discriminer paie ; si elle
    stagne ou retombe, l'environnement ne le recompense pas.
    """
    pop = np.asarray(pop, dtype=float)
    ora = np.asarray(oracles, dtype=float)
    x = (np.asarray(steps) if steps is not None
         else np.arange(start_step, start_step + len(pop)))
    freq = np.divide(ora, pop, out=np.zeros_like(ora), where=pop > 0)

    fig, (h, b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    h.plot(x, 100 * freq, color="#C1121F", lw=2, label="envahisseurs")
    if cible is not None:
        h.axhline(100 * cible, color="0.4", ls="--", lw=1,
                  label=f"cible d'injection ({100*cible:.0f}%)")
    if invasion_start is not None:
        for ax in (h, b):
            ax.axvline(invasion_start, color="0.3", ls=":", lw=1.4)
        h.annotate("injection", (invasion_start, h.get_ylim()[1]),
                   xytext=(6, -12), textcoords="offset points", fontsize=9, color="0.3")
    h.set_ylabel("part de la population (%)")
    h.set_ylim(bottom=0)
    h.grid(alpha=.3); h.legend(loc="best")
    h.set_title("Test d'invasion — frequence des oracles")

    b.plot(x, pop, color="0.35", lw=1.2, label="population totale")
    b.plot(x, ora, color="#C1121F", lw=1.2, label="oracles")
    b.set_ylabel("effectif"); b.set_xlabel("Steps")
    b.grid(alpha=.3); b.legend(loc="best", fontsize=9)

    fig.tight_layout()
    path = os.path.join(exp_dir, "fig"); os.makedirs(path, exist_ok=True)
    out = os.path.join(path, "plot_invasion.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"Figure saved: {out}")


def plot_inner_loss(perte, exp_dir, start_step=0, steps=None):
    """Erreur de prediction de la boucle interne, au fil du run.

    Elle ne dit PAS a elle seule qu'il y a effet Baldwin : la boucle interne
    tourne des le premier pas. Ce qui compte est sa BAISSE au fil des
    generations -- l'evolution fournirait alors une initialisation depuis
    laquelle predire coute moins cher.
    """
    y = np.asarray(perte, dtype=float)
    x = (np.asarray(steps) if steps is not None
         else np.arange(start_step, start_step + len(y)))
    # les blocs ou personne n'a mange sont des NaN : matplotlib les saute
    fini = np.isfinite(y)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x[fini], y[fini], color="#1D3557", lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("erreur quadratique par bouchee")
    ax.set_title("Boucle interne — erreur de prediction de la valeur des canaux")
    ax.grid(alpha=.3, which="both")

    fig.tight_layout()
    path = os.path.join(exp_dir, "fig"); os.makedirs(path, exist_ok=True)
    out = os.path.join(path, "plot_inner_loss.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"Figure saved: {out}")


def plot_metric_pairs(par_genome, exp_dir, chunk, suffix="", titre=""):
    """Chaque metrique du lab en fonction de chaque autre, sur les agents.

    Triangle bas : les nuages. Diagonale : la distribution marginale. Triangle
    haut : le rho de Spearman, qui resume ce que le nuage symetrique montrerait.
    Spearman et non Pearson : `age` sature au plafond du rollout et `adapt_gain`
    a des queues lourdes.
    """
    cles = [k for k in CORR_METRIQUES if k in par_genome]
    M = np.column_stack([np.asarray(par_genome[k], dtype=float) for k in cles])
    ok = np.isfinite(M).all(axis=1)
    M = M[ok]
    n = M.shape[0]
    if n < 5:
        print(f"Pair plot : seulement {n} agents complets, saute.")
        return

    R = np.apply_along_axis(lambda c: np.argsort(np.argsort(c)), 0, M).astype(float)
    C = np.corrcoef(R, rowvar=False)
    d = len(cles)
    noms = [CORR_TITRES.get(k, k) for k in cles]

    fig, axes = plt.subplots(d, d, figsize=(2.25 * d, 2.25 * d),
                             squeeze=False)
    cmap = plt.get_cmap("RdBu_r")
    for i in range(d):
        for j in range(d):
            ax = axes[i][j]
            if i == j:
                ax.hist(M[:, i], bins=18, color="0.6", edgecolor="white",
                        linewidth=.5)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(M[:, j], M[:, i], s=16, alpha=.7, color="C0",
                           edgecolor="none")
            else:
                r = C[i, j]
                ax.set_facecolor(cmap((r + 1) / 2))
                ax.text(.5, .5, f"{r:+.2f}", ha="center", va="center",
                        fontsize=13, transform=ax.transAxes,
                        color="white" if abs(r) > .55 else "0.15")
                ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(labelsize=7)
            if i != d - 1 or i == j:
                ax.set_xticklabels([])
            if j != 0 or i == j:
                ax.set_yticklabels([])
            if i == d - 1:
                ax.set_xlabel(noms[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(noms[i], fontsize=9)
            ax.grid(alpha=.2)

    fig.suptitle(f"Lab metrics, pairwise — chunk {chunk}"
                 + (f"  |  {titre}" if titre else "") + f"    ({n} agents)",
                 fontsize=13, y=.995)
    fig.tight_layout()
    fig_dir = os.path.join(exp_dir, "fig", "correlations")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"metric_pairs_chunk_{chunk}{suffix}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")


_SOMMETS = {"medium": (0.0, 0.0), "poison": (1.0, 0.0),
            "good": (0.5, np.sqrt(3) / 2)}


def _bary(p_good, p_medium, p_poison):
    """Proportions -> coordonnees du triangle equilateral.

    medium en (0,0), poison en (1,0), good en (0.5, sqrt(3)/2) : chaque
    coordonnee est donc la moyenne des sommets ponderee par les proportions.
    """
    return p_poison + p_good / 2.0, (np.sqrt(3) / 2.0) * p_good


def _teinte_id(nom):
    """Couleur du sommet, alpha ignore (COLOR_BY_ID[1] a 8 chiffres hex)."""
    from simulation.data_class import LABELS, COLOR_BY_ID
    c = COLOR_BY_ID[LABELS.index(nom)]
    return c[:7] if len(c) == 9 else c


def _cadre_simplex(ax, etiquettes=True):
    """Triangle, grille interne, graduations et sommets. Partage par toutes les
    figures en simplex pour qu'elles restent lisibles de la meme facon."""
    coins = np.array([_SOMMETS["medium"], _SOMMETS["poison"], _SOMMETS["good"],
                      _SOMMETS["medium"]])
    ax.plot(coins[:, 0], coins[:, 1], color="0.25", lw=1.4, zorder=2)

    for f in np.arange(0.2, 1.0, 0.2):
        for a, b in (((f, 1 - f, 0), (f, 0, 1 - f)),      # good constant
                     ((1 - f, f, 0), (0, f, 1 - f)),      # medium constant
                     ((1 - f, 0, f), (0, 1 - f, f))):     # poison constant
            x, y = zip(_bary(*a), _bary(*b))
            ax.plot(x, y, color="0.85", lw=0.7, zorder=1)

    if etiquettes:
        # une composante par cote, sur le bord ou naissent ses iso-lignes
        for f in np.arange(0.2, 1.0, 0.2):
            xg, yg = _bary(f, 0, 1 - f)                    # good, cote droit
            ax.text(xg + .028, yg + .008, f"{100*f:.0f}", ha="left", va="center",
                    fontsize=8, color=_teinte_id("good"))
            xp, yp = _bary(0, 1 - f, f)                    # poison, cote bas
            ax.text(xp, yp - .028, f"{100*f:.0f}", ha="center", va="top",
                    fontsize=8, color=_teinte_id("poison"))
            xm, ym = _bary(1 - f, f, 0)                    # medium, cote gauche
            ax.text(xm - .028, ym + .008, f"{100*f:.0f}", ha="right", va="center",
                    fontsize=8, color=_teinte_id("medium"))

    for nom, (dx, dy, ha, va) in (("good", (0, .040, "center", "bottom")),
                                  ("medium", (-.045, -.030, "right", "top")),
                                  ("poison", (.045, -.030, "left", "top"))):
        x0, y0 = _SOMMETS[nom]
        ax.text(x0 + dx, y0 + dy, nom, ha=ha, va=va, fontsize=13, weight="bold",
                color=_teinte_id(nom))

    ax.set_aspect("equal")
    ax.axis("off")
    marge = .09
    ax.set_xlim(-marge, 1 + marge)
    ax.set_ylim(-marge, np.sqrt(3) / 2 + marge)


def plot_food_simplex(eaten, ids, age, disponible, exp_dir, chunk,
                      suffix="", titre="", fig_dir=None, parent=None,
                      age_max=None):
    """Composition du regime de chaque agent, dans un simplex good/medium/poison.

    `eaten` est indexe par CANAL ; `ids` donne l'identite de chaque canal. On
    reordonne par identite, sinon les sommets seraient faux des qu'un shuffle a
    eu lieu.
    """
    from simulation.data_class import LABELS, COLOR_BY_ID

    eaten = np.asarray(eaten, dtype=float)
    if eaten.shape[1] != 3:
        print(f"Simplex : {eaten.shape[1]} ressources, il en faut 3.")
        return

    # canal -> identite, puis colonnes dans l'ordre good, medium, poison
    par_id = {int(i): k for k, i in enumerate(ids)}
    ordre = [par_id[LABELS.index(n)] for n in ("good", "medium", "poison")]
    e = eaten[:, ordre]
    dispo_ok = disponible is not None and np.asarray(disponible).size == 3
    dispo = np.asarray(disponible, dtype=float)[ordre] if dispo_ok else None

    total = e.sum(axis=1)
    ok = total > 0                       # rien mange -> composition indefinie
    p = e[ok] / total[ok, None]
    n_exclus = int((~ok).sum())

    fig, ax = plt.subplots(figsize=(7.5, 7))
    _cadre_simplex(ax)

    # agents
    if p.size:
        x, y = _bary(p[:, 0], p[:, 1], p[:, 2])
        c = np.asarray(age, dtype=float)[ok]
        fini = np.isfinite(c)
        # echelle fixe quand age_max est donne : sinon chaque figure a sa
        # propre normalisation et deux chunks ne sont plus comparables a l'oeil
        sc = ax.scatter(x[fini], y[fini], c=c[fini], cmap="viridis", s=55,
                        edgecolor="white", linewidth=.6, zorder=4,
                        vmin=0 if age_max else None,
                        vmax=age_max if age_max else None)
        cb = fig.colorbar(sc, ax=ax, shrink=.62, pad=.02)
        cb.set_label("lifespan (steps)", fontsize=10)
        if (~fini).any():
            ax.scatter(x[~fini], y[~fini], c="0.7", s=55, edgecolor="white",
                       linewidth=.6, zorder=3)

    # le parent, quand on trace le nuage de ses enfants
    if parent is not None:
        pp = np.asarray(parent, dtype=float)[ordre]
        if pp.sum() > 0:
            xp, yp = _bary(*(pp / pp.sum()))
            ax.scatter([xp], [yp], marker="*", s=340, color="#C1121F",
                       edgecolor="white", linewidth=.8, zorder=6, label="parent")

    # composition OFFERTE : la preference se lit comme l'ecart a ce point
    if dispo_ok and dispo.sum() > 0:
        d = dispo / dispo.sum()
        xd, yd = _bary(*d)
        ax.scatter([xd], [yd], marker="o", s=190, facecolor="none",
                   edgecolor="black", linewidth=2.0, zorder=5,
                   label="available on the grid")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", frameon=False, fontsize=9,
                  bbox_to_anchor=(-.02, 1.0))

    sous = f"{int(ok.sum())} agents"
    if n_exclus:
        sous += f"  ({n_exclus} exclus : rien mange)"
    ax.set_title(f"Diet composition — chunk {chunk}"
                 + (f"  |  {titre}" if titre else "") + f"\n{sous}",
                 fontsize=12)

    fig_dir = fig_dir or os.path.join(exp_dir, "fig", "simplex")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"food_simplex_chunk_{chunk}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")


def plot_energy_expectation(resources, out_path, niveaux=7):
    """Esperance d'energie par item mange, pour chaque composition de regime.

    E[de] = somme_i p_i * de_i : lineaire sur le simplex, donc un degrade dont
    la ligne zero separe les regimes rentables des regimes deficitaires. Ne
    depend que de data_class, aucune simulation n'est necessaire.
    """
    from simulation.data_class import LABELS

    par_id = {r.id: r for r in resources}
    if set(par_id) != {0, 1, 2}:
        print(f"Esperance : il faut les 3 identites, reçu {sorted(par_id)}.")
        return
    de = np.array([par_id[LABELS.index(n)].delta_energy
                   for n in ("good", "medium", "poison")], dtype=float)

    # maillage barycentrique regulier
    N = 200
    i, j = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
    garde = (i + j) <= N
    pg, pp = i[garde] / N, j[garde] / N
    pm = 1.0 - pg - pp
    x, y = _bary(pg, pm, pp)
    E = pg * de[0] + pm * de[1] + pp * de[2]

    vmax = float(np.abs(de).max())
    fig, ax = plt.subplots(figsize=(9.2, 7))
    tcf = ax.tricontourf(x, y, E, levels=np.linspace(-vmax, vmax, 41),
                         cmap="RdYlGn", vmin=-vmax, vmax=vmax, zorder=0)
    cs = ax.tricontour(x, y, E, levels=niveaux, colors="0.30", linewidths=.6,
                       zorder=1)
    ax.clabel(cs, fmt="%+.2f", fontsize=7.5, inline=True)
    z = ax.tricontour(x, y, E, levels=[0.0], colors="black", linewidths=2.2,
                      zorder=3)
    ax.clabel(z, fmt={0.0: "break-even"}, fontsize=9, inline=True)

    _cadre_simplex(ax)
    cb = fig.colorbar(tcf, ax=ax, shrink=.72, pad=.10,
                      ticks=np.linspace(-vmax, vmax, 5))
    cb.set_label("expected energy per item eaten", fontsize=10)

    detail = "   ".join(f"{n} {par_id[LABELS.index(n)].delta_energy:+g}"
                        for n in ("good", "medium", "poison"))
    ax.set_title("Expected energy of a diet\n" + detail, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


EVO_METRIQUES = ["age", "mean_rew", "mean_speed", "energy_end",
                 "greediness", "adapt_score", "adapt_gain"]
EVO_TITRES = {"age": "Lifespan (steps)", "mean_rew": "Consumption /step",
              "mean_speed": "Movement /step", "energy_end": "Final energy",
              "greediness": "Greediness  G = Cr/Tr",
              "adapt_gain": "Net gain / hungry step"}


def plot_evolvability(data_dir, chunk, fig_dir=None):
    """Un panneau par mesure : en x les parents, en y la boite des enfants.

    Le point rouge est le parent lui-meme, evalue dans le meme environnement.
    Boite au-dessus du point = la mutation ameliore en moyenne.
    """
    f = os.path.join(data_dir, f"chunk_{chunk}.npz")
    if not os.path.exists(f):
        print(f"No evolvability data at {f}")
        return
    d = np.load(f, allow_pickle=True)
    etiq = [str(x) for x in d["etiquettes"]]
    presentes = [k for k in EVO_METRIQUES
                 if k in EVO_TITRES and f"enfants_{k}" in d.files]
    if not presentes:
        print("No metric in evolvability data.")
        return

    n_col = 3
    n_lig = -(-len(presentes) // n_col)
    fig, axes = plt.subplots(n_lig, n_col, figsize=(5.2 * n_col, 4 * n_lig),
                             squeeze=False)
    axes = axes.ravel()

    for ax, k in zip(axes, presentes):
        enf = np.asarray(d[f"enfants_{k}"], dtype=float)      # (P, M)
        par = np.asarray(d[f"parent_{k}"],  dtype=float)      # (P,)
        donnees = [e[np.isfinite(e)] for e in enf]
        pos = np.arange(1, len(donnees) + 1)
        garde = [i for i, e in enumerate(donnees) if e.size]
        if garde:
            ax.boxplot([donnees[i] for i in garde],
                       positions=pos[garde], widths=.6, showfliers=True,
                       flierprops=dict(marker=".", markersize=3, alpha=.5))
        ok = np.isfinite(par)
        if ok.any():
            ax.plot(pos[ok], par[ok], "o", color="C3", ms=7, zorder=5,
                    label="parent")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(EVO_TITRES.get(k, k))
        ax.set_xticks(pos)
        ax.set_xticklabels(etiq, fontsize=9)
        ax.set_xlabel("parent")
        ax.grid(alpha=.3, axis="y")
        if k == "adapt_gain":
            ax.axhline(0, color="black", lw=1)

    for ax in axes[len(presentes):]:
        ax.axis("off")
    fig.suptitle(f"Evolvability — chunk {chunk} — "
                 f"{enf.shape[1]} mutated offspring per parent", y=1.0)
    fig.tight_layout()
    fig_dir = fig_dir or data_dir
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"evolvability_chunk_{chunk}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")


def plot_lab_exploration(exp_dir):
    """Évolution de l'EXPLORATION (env low_res) chunk après chunk.
    Deux panneaux :
      - frac_found_food : part des agents qui trouvent au moins une ressource
      - explore_time    : délai moyen (± 1σ) avant la 1re ressource, calculé sur
                          les seuls agents qui ont mangé (grandeur conditionnelle).
    """
    data_dir = os.path.join(exp_dir, "lab_data")
    fig_dir = os.path.join(exp_dir, "fig")
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*_lowres_summary.json")),
                   key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1)))
    if not files:
        print("No low_res summary to plot.")
        return
 
    S = [json.load(open(f)) for f in files]
    x = np.array([s["chunk"] for s in S])
 
    frac = np.array([s["frac_found_food"] for s in S])
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
 
    axes[0].plot(x, frac, marker="o", color="C2")
    axes[0].set_title("Fraction of agents that found food")
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
 
    _plot_band(axes[1], x, S, "explore_time")
    axes[1].set_title("Time to first resource (exploration)")
 
    _plot_band(axes[2], x, S, "greediness")
    axes[2].set_title("Greediness  G = Cr/Tr")
    axes[2].set_ylabel("ratio")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best")
    axes[1].set_ylabel("steps")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
 
    for ax in axes:
        ax.set_xlabel("chunk")
 
    fig.tight_layout()
    out = os.path.join(fig_dir, "lab_exploration_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")
 
 
def plot_alone_vs_clones(exp_dir, tag="alone_vs_clones",
                         prefixes=("alone", "clones"),
                         labels=("alone", "clones (median of peers)"),
                         titre="Focal agent alone vs among identical clones",
                         fname="lab_alone_vs_clones_evolution.png"):
    """Évolution d'une comparaison APPARIÉE, chunk après chunk.

    Un sous-graphe par métrique, deux courbes : les deux conditions comparées.
    Générique — sert à l'effet des pairs (seul vs clones) comme à l'ablation de
    mémoire (intacte vs coupée), qui partagent exactement la même structure de
    données. `tag` sélectionne la famille de fichiers, `labels` les clés de
    dispersion écrites par la comparaison correspondante."""
    data_dir = os.path.join(exp_dir, "lab_data")
    fig_dir = os.path.join(exp_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)   # ne pas dependre d'un plot appele avant
    files = sorted(glob.glob(os.path.join(data_dir, f"chunk_*_{tag}.json")),
                   key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1)))
    if not files:
        print(f"No {tag} data to plot.")
        return
 
    P = [json.load(open(f)) for f in files]
    x = np.array([p["chunk"] for p in P])
 
    titles  = {"age": "Lifespan (steps)", "mean_rew": "Consumption /step",
               "mean_speed": "Movement /step", "energy_end": "Final energy",
               "wall_death": "Fraction wall deaths",
               "greediness": "Greediness  G = Cr/Tr",
               "adapt_gain": "Net gain / hungry step"}
    # Union sur TOUS les chunks, pas seulement le premier : une metrique ajoutee
    # en cours de run n'existe que dans les fichiers recents, et se caler sur
    # P[0] la ferait disparaitre a jamais. _get rend NaN sur les chunks qui ne
    # l'ont pas, donc la courbe demarre simplement plus tard.
    presentes = {k for p in P for k in p["metrics"]}
    metrics = [k for k in titles if k in presentes]

    n_cols = 3
    n_rows = -(-len(metrics) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows),
                             sharex=True, squeeze=False)
    axes = axes.ravel()

    for ax, k in zip(axes, metrics):
        # {} pour un chunk anterieur a cette metrique : _get y lira NaN
        Sk = [p["metrics"].get(k, {}) for p in P]
        _plot_band(ax, x, Sk, prefixes[0], color="C0", label=labels[0])
        _plot_band(ax, x, Sk, prefixes[1], color="C1", label=labels[1])
        ax.set_title(titles[k])
        ax.grid(alpha=0.3)
        ax.set_xlabel("chunk")
        if k == "adapt_gain":
            ax.axhline(0, color="black", lw=1)   # 0 = mange sans choisir

    for ax in axes[len(metrics):]:
        ax.axis("off")
    axes[0].legend(loc="best")
 
    fig.suptitle(titre, y=1.0)
    fig.tight_layout()
    out = os.path.join(fig_dir, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")
 
 
# =====================================================================
#  NOUVEAU — ENERGIE AU COURS DU TEMPS (les 3 labs)
#  Necessite en tete de plots.py :
#      from matplotlib.collections import LineCollection
#      from matplotlib.lines import Line2D
# =====================================================================
 
def _colored_line(ax, x, y, min_repr, e_die):
    """Trace y(x) en coloriant chaque segment selon le regime energetique."""
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ymid = 0.5 * (y[:-1] + y[1:])
    c = np.where(ymid >= min_repr, "#2e7d4f",
        np.where(ymid <  e_die,    "#c0392b", "#7f8c8d"))
    ax.add_collection(LineCollection(segs, colors=c, linewidths=1.5))
 
 
def _repro_events(e, min_repr, time_above):
    """c s'incremente tant que E>=min_repr, reset des que E repasse dessous.
       c atteint time_above -> evenement REPRO (et reset).
       E redescend avec c>0 -> evenement DROP (montee avortee)."""
    repro, drop, c = [], [], 0
    for t in range(len(e)):
        if e[t] >= min_repr:
            c += 1
            if c >= time_above:
                repro.append(t); c = 0
        else:
            if c > 0:
                drop.append(t)
            c = 0
    return np.array(repro, int), np.array(drop, int)
 
 
def _y_limits(y, y_min, y_max):
    """Cadrage : (y_min, y_max) est un PLANCHER, pas une borne dure. On voit
    toujours au moins cette fenetre (donc les deux seuils), et l'axe s'etend
    si la courbe sort."""
    pad = 0.05 * max(y.max() - y.min(), 1e-9)
    return min(y_min, y.min() - pad), max(y_max, y.max() + pad)
 
 
def plot_lab_energy(energy, alive, exp_dir, lab_dir, chunk,
                    min_energy_repr, time_above_repr, energy_to_die=0.0,
                    n_envs=10, env_title="", y_min=-1.0, y_max=4.5,
                    share_y=True):
    """Figure au format 6:4. L'axe y couvre au minimum (y_min, y_max) et
    s'agrandit si l'energie sort de cette fenetre.
    share_y=True : tous les sous-graphes d'une figure partagent le meme axe y
    (les clones sont alors comparables entre eux) ; False : un cadrage par
    agent, plus detaille mais non comparable."""
    energy = np.asarray(energy)
    alive  = np.asarray(alive)
    if energy.ndim == 4 and energy.shape[-1] == 1:
        energy = energy[..., 0]
 
    out_dir = os.path.join(exp_dir, "energy", lab_dir)
    os.makedirs(out_dir, exist_ok=True)
    B = min(n_envs, energy.shape[0])
 
    for b in range(B):
        E, A = energy[b], alive[b]
        slots = [n for n in range(A.shape[1]) if A[:, n].any()]
        if not slots:
            continue
 
        fig, axes = plt.subplots(len(slots), 1, figsize=(12, 8),   # rapport 6:4
                                 squeeze=False, sharex=True)
        axes = axes[:, 0]
 
        # trajectoires vivantes de cette figure, puis cadrage
        traj = []
        for n in slots:
            idx = np.where(A[:, n] == 1)[0]
            t0, t1 = idx[0], idx[-1]
            traj.append((n, np.arange(t0, t1 + 1), E[t0:t1 + 1, n]))
 
        if share_y:
            allv = np.concatenate([yy for _, _, yy in traj])
            lims = [_y_limits(allv, y_min, y_max)] * len(traj)
        else:
            lims = [_y_limits(yy, y_min, y_max) for _, _, yy in traj]
 
        for ax, (n, x, y), (y_lo, y_hi) in zip(axes, traj, lims):
            # bandes de fond : regime reproductif / intermediaire / letal
            ax.axhspan(min_energy_repr, y_hi,            facecolor="#e8f2ec", zorder=0)
            ax.axhspan(energy_to_die,   min_energy_repr, facecolor="#f7f7f7", zorder=0)
            ax.axhspan(y_lo,            energy_to_die,   facecolor="#fdeceb", zorder=0)
 
            _colored_line(ax, x, y, min_energy_repr, energy_to_die)
            ax.axhline(min_energy_repr, color="#2e7d4f", ls=":", lw=1, alpha=.8)
            ax.axhline(energy_to_die,   color="#c0392b", ls=":", lw=1, alpha=.8)
 
            repro, drop = _repro_events(y, min_energy_repr, time_above_repr)
            for t in repro:   # pointilles larges, vert fonce
                ax.axvline(x[t], color="#146b34", lw=2.0, alpha=.95, dashes=(6, 3))
            for t in drop:    # pointilles serres, orange
                ax.axvline(x[t], color="#c77d0a", lw=1.1, alpha=.85, dashes=(2, 2))
 
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel(f"energy (slot {n})")
            ax.grid(alpha=.25)
 
        handles = [
            Line2D([], [], color="#2e7d4f", lw=2, label=f"E $\\geq$ {min_energy_repr:g} (repr. threshold)"),
            Line2D([], [], color="#7f8c8d", lw=2, label="intermediate"),
            Line2D([], [], color="#c0392b", lw=2, label=f"E < {energy_to_die:g} (lethal)"),
            Line2D([], [], color="#146b34", lw=2.0, dashes=(6, 3), label=f"reproduction reached ({time_above_repr:g} steps above)"),
            Line2D([], [], color="#c77d0a", lw=1.1, dashes=(2, 2), label="dropped below before threshold"),
        ]
        axes[-1].set_xlabel("step")
        fig.suptitle(f"{env_title} — chunk {chunk}, agent {b}", y=0.995)
        frac = 0.75 / 8.0                         # ~0.75 pouce reserve pour la legende
        fig.tight_layout(rect=[0, frac, 1, 1])
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(os.path.join(out_dir, f"chunk_{chunk}_agent_{b:02d}.png"), dpi=130)
        plt.close(fig)
    print(f"Energy plots saved: {out_dir}")
 


GROUPS = ["encoder", "lstm_input", "lstm_recurrent", "controller"]


def _xaxis(steps, gen_depth, x_axis):
    if x_axis == "generation":
        if gen_depth is None:
            raise ValueError("x_axis='generation' requiert gen_depth")
        return np.asarray(gen_depth, float), "générations (profondeur généalogique)"
    return np.asarray(steps, float), "step"


def _connect_origin(ax, x, y, color, alpha=0.55):
    """Raccorde (0, 0) au premier point mesuré, en pointillé léger.

    L'origine est exacte : au step 0 chaque agent est son propre ancêtre, donc
    distance nulle. Mais la mesure ne démarre qu'après coalescence, souvent
    très loin dans le run. Le segment qui l'y relie ne traverse AUCUNE donnée —
    on le distingue visuellement pour ne pas le faire passer pour une mesure.
    (En abscisse 'generation' il reste une bonne approximation, la dérive
    neutre y étant linéaire ; en abscisse 'step' il ne l'est que si le temps de
    génération est stationnaire.)
    """
    ok = np.nonzero(np.isfinite(y))[0]
    if len(ok) == 0:
        return
    j = ok[0]
    ax.plot([0.0, x[j]], [0.0, y[j]], ls=":", lw=1.0, color=color,
            alpha=alpha, zorder=1)
    ax.plot([0.0], [0.0], marker="o", ms=3, color=color, alpha=alpha, zorder=1)


def plots_metrics_weight_distance(dist_list, exp_dir, steps,
                                  neutral=None, gen_depth=None,
                                  x_axis="step", show_origin=False,
                                  share_y=True, fname="weight_distance.png"):
    """Distance à l'ancêtre commun : un panneau par bloc, observé vs neutre.

    Un panneau par groupe plutôt que huit courbes superposées — avec l'observé
    et le neutre si proches, la version empilée ne permet pas de voir lequel
    passe au-dessus de l'autre.

    Args:
        dist_list   : [{groupe: (mean, std)}] observé, un élément par chunk
        neutral     : [{groupe: (mean, std)}] neutre, même structure
        share_y     : axe y commun aux quatre panneaux. Recommandé : c'est ce
                      qui rend les blocs comparables entre eux.
        show_origin : raccorder (0, 0) au premier point. Désactivé par défaut :
                      la mesure ne démarre qu'à la coalescence, donc ce segment
                      couvrirait des dizaines de milliers de steps sans donnée.
    """
    x, xlab = _xaxis(steps, gen_depth, x_axis)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             sharex=True, sharey=share_y)

    for i, (g, ax) in enumerate(zip(GROUPS, axes.ravel())):
        c = f"C{i}"
        m = np.array([d[g][0] for d in dist_list], dtype=float)
        s = np.array([d[g][1] for d in dist_list], dtype=float)

        ax.plot(x, m, marker="o", ms=3, color=c, lw=1.4,
                label="observé", zorder=3)
        ax.fill_between(x, m - s, m + s, alpha=0.18, color=c, lw=0)
        if show_origin:
            _connect_origin(ax, x, m, c)

        if neutral is not None:
            nm = np.array([n[g][0] for n in neutral], dtype=float)
            ns = np.array([n[g][1] for n in neutral], dtype=float)
            ax.plot(x, nm, ls="--", lw=1.4, color="0.25",
                    label=r"neutre  $p\sigma^2 g$", zorder=2)
            ax.fill_between(x, nm - ns, nm + ns, color="0.25",
                            alpha=0.12, lw=0)
            if show_origin:
                _connect_origin(ax, x, nm, "0.25", alpha=0.35)



        ax.set_title(g, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        # Axe démarré à 0 alors que la donnée démarre à la coalescence : le vide
        # à gauche est informatif, il montre quelle part du run s'est écoulée
        # avant que la mesure ne devienne définissable.
        ax.set_xlim(left=0)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel(xlab)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\langle (w - w_{\mathrm{anc}})^2 \rangle$")

    fig.suptitle("Distance des poids à l'ancêtre commun — observé vs dérive neutre",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(exp_dir, "fig", fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


def plots_weight_selection(dist_list, neutral, exp_dir, steps, gen_depth=None,
                           x_axis="step", fname="weight_selection.png",
                           shuffle_steps=()):
    """Écart à la neutralité — le graphe à regarder en premier.

    Panneau gauche, rapport observé/neutre :
        ~1 -> neutralité   <1 -> sélection stabilisante   >1 -> directionnelle
    Il élimine la croissance en p*sigma^2*g commune aux quatre blocs et rend
    les blocs comparables malgré leurs tailles de couches différentes.

    Panneau droit, écart normalisé  (obs - neutre) / std_neutre :
        combien d'écarts-types neutres séparent l'observé de la neutralité.
        C'est la version quantitative : un rapport de 0.9 n'a pas le même sens
        selon que la bande neutre est large ou serrée, et la largeur de cette
        bande dépend du bloc (petites couches = bande large).
    """
    x, xlab = _xaxis(steps, gen_depth, x_axis)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # En abscisse GENERATIONS, x n'est PAS monotone : la profondeur genealogique
    # moyenne recule quand une bouffee de naissances meurt, les survivants etant
    # les vieux agents, donc les moins profonds. Relier les points par un trait
    # ferait alors revenir la courbe sur elle-meme, ce qui se lit comme une
    # anomalie. On ne trace donc que les marqueurs, et un degrade indique le sens
    # du temps.
    recule = np.any(np.diff(np.asarray(x, float)) < 0)
    for i, g in enumerate(GROUPS):
        m  = np.array([d[g][0] for d in dist_list])
        nm = np.array([n[g][0] for n in neutral])
        ns = np.array([n[g][1] for n in neutral])

        style = dict(marker="o", ms=3.5, color=f"C{i}", label=g)
        if recule:
            style.update(ls="none")
        axes[0].plot(x, m / np.where(nm > 0, nm, np.nan), **style)
        axes[1].plot(x, (m - nm) / np.where(ns > 0, ns, np.nan), **style)

    # Permutations des ressources : converties dans l'unite de l'axe. `steps` est
    # croissant, donc l'interpolation est valide meme si gen_depth ne l'est pas.
    for j, s in enumerate(np.atleast_1d(np.asarray(shuffle_steps, float))):
        if s < np.min(steps) or s > np.max(steps):
            continue
        xv = s if x_axis != "generation" else float(np.interp(s, steps, gen_depth))
        for ax in axes:
            ax.axvline(xv, color="0.35", ls=":", lw=1.2, zorder=0,
                       label="permutation" if j == 0 else None)

    axes[0].axhline(1.0, color="k", ls="--", lw=1.5, label="neutre")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("distance observée / neutre")
    axes[0].set_title("Rapport à la neutralité")

    axes[1].axhline(0.0, color="k", ls="--", lw=1.5)
    axes[1].axhspan(-2, 2, color="k", alpha=0.07, lw=0)
    axes[1].set_ylabel(r"$(\mathrm{obs} - \mathrm{neutre})\,/\,\sigma_{\mathrm{neutre}}$")
    axes[1].set_title("Écart normalisé (bande grise : $\\pm 2\\sigma$)")

    for ax in axes:
        ax.set_xlabel(xlab)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(exp_dir, "fig", fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")



def _wilson(k, n, z=1.96):
    """Intervalle de Wilson 95% pour une proportion k/n. Mieux que l'intervalle
    normal quand p est proche de 0 ou 1 ou quand n est petit (ne sort jamais
    de [0, 1]). Retourne (p, lo, hi), NaN si n = 0."""
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half   = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = np.where(n > 0, center - half, np.nan)
    hi = np.where(n > 0, center + half, np.nan)
    return p, lo, hi

def plot_generation_rate(steps, gen_depth, gen_depth_std, exp_dir,
                         fname="generation_rate.png"):
    """Diagnostic : g(t) est-il linéaire ?

    Si oui, le temps de génération est stationnaire et 'step' est
    interchangeable avec 'generation'. Sinon la démographie dérive, et seule
    l'abscisse 'generation' isole proprement la dérive des poids.
    """
    x = np.asarray(steps, float)
    g = np.asarray(gen_depth, float)
    s = np.asarray(gen_depth_std, float)

    alpha, b = np.polyfit(x, g, 1)
    pred = alpha * x + b
    r2 = 1.0 - np.sum((g - pred) ** 2) / np.sum((g - g.mean()) ** 2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, g, marker="o", ms=3.5, color="C0", label="profondeur mesurée")
    ax.fill_between(x, g - s, g + s, alpha=0.15, color="C0", lw=0)
    ax.plot(x, pred, "k--", lw=1.4,
            label=fr"{alpha:.3g} gén/step, $R^2$={r2:.4f}")
    ax.set_xlabel("step")
    ax.set_ylabel("générations depuis l'ancêtre")
    ax.set_title("Temps de génération")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(exp_dir, "fig", fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")
    
# =====================================================================
#  NOUVEAU — COURBE DE RÉPONSE ÉNERGÉTIQUE  P(mange | voit, E)
#  Necessite en tete de plots.py :
#      from simulation.energy_response import _wilson
# =====================================================================
def plot_energy_response(exp_dir, chunk, curves, cfg=None, fname=None):
    """curves : dict {label: (bins, n, k)} — une entree par lab.
    Trace P(mange | ressource vue) par bin d'energie, avec bande de Wilson 95%.
    Sous-panneau : effectif n par bin (la ou n est faible, P est peu fiable)."""
    import matplotlib.pyplot as plt
 
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    colors = {}
    for i, (label, (bins, n, k)) in enumerate(curves.items()):
        c = f"C{i}"
        colors[label] = c
        centers = 0.5 * (np.asarray(bins[:-1], float) + np.asarray(bins[1:], float))
        # remplace les bords infinis par un centre lisible
        if not np.isfinite(centers[0]):
            centers[0] = bins[1] - 0.5 * (bins[2] - bins[1])
        if not np.isfinite(centers[-1]):
            centers[-1] = bins[-2] + 0.5 * (bins[-2] - bins[-3])
        p, lo, hi = _wilson(k, n)
        ok = n > 0
        ax.plot(centers[ok], p[ok], marker="o", color=c, label=label)
        ax.fill_between(centers[ok], lo[ok], hi[ok], alpha=0.20, color=c)
        axn.plot(centers[ok], n[ok], marker=".", color=c)
 
    # seuils physiques
    if cfg is not None:
        ax.axvline(cfg.energy_to_die,   color="#c0392b", ls=":", lw=1, alpha=.8)
        ax.axvline(cfg.min_energy_repr, color="#2e7d4f", ls=":", lw=1, alpha=.8)
 
    ax.set_ylabel("P(eats | resource in view)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Response to a visible resource vs energy — chunk {chunk}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
 
    axn.set_ylabel("# events")
    axn.set_xlabel("energy at the moment the resource is seen")
    axn.set_yscale("log")
    axn.grid(alpha=0.3)
 
    fig.tight_layout()
    data_dir = os.path.join(exp_dir, "lab_data")
    fig_dir = os.path.join(exp_dir, "fig")
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(fig_dir, fname or f"chunk_{chunk}_energy_response.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")
