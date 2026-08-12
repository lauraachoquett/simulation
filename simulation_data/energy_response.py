"""COURBE DE RÉPONSE ÉNERGÉTIQUE

Question : quand une ressource est dans le champ de vision, l'agent la mange-t-il ?
Et cette propension dépend-elle de son niveau d'énergie ?

Pour chaque pas t où l'agent VOIT une ressource (event conditionnant) :
  - x = energy[t]                          (son niveau d'énergie a cet instant)
  - y = 1 si l'agent consomme dans [t, t+W], sinon 0   (le temps de marcher
        jusqu'a la case ; W ~ rayon du champ de vision)
On agrege par bin d'energie :@@

  P(mange | voit, E in bin) = (# events avec y=1) / (# events du bin)@
                            = k_bin / n_bin@

C'est une courbe de reponse comportementale : E bas + P haut = l'agent se jette
sur la nourriture quand il a faim. P plat = comportement independant de la faim.

--------------------------------------------------------------------------
PREREQUIS : obs=state.obs dans StepLog (cf. patch greediness), sinon "voir" et
"manger" sont desalignes.
--------------------------------------------------------------------------
"""

import os
import json
import numpy as np


# W : pas apres l'observation ou la consommation compte encore.
# = 2 * cfg.agent_view : dans une vue de cote 2v+1, la case la plus eloignee est
# a une distance de MANHATTAN de 2v (les deplacements sont 4-connexes), pas v.
# A W = v, la moitie du champ de vision etait hors de portee et ces evenements
# comptaient quand meme au denominateur, ce qui ecrasait P vers le bas.
ENERGY_EAT_WINDOW = 10


def resource_in_view(obs, channels, n_channels):
    """(T, N) booleen : au moins un des `channels` est dans le champ de vision.

    Definition UNIQUE, importee par lab.py (l'inverse ferait un cycle, lab.py
    importe deja ce module).

    Le CANAL EST LE DERNIER AXE : get_obs_vector (agent_mov.py:61) termine par
    jnp.transpose(obs, (1, 2, 0)), donc obs vaut (T, N, side, side, C).
    Indexer l'axe 2, comme le faisaient les deux versions precedentes, decoupe
    en realite les premieres LIGNES de la vue tous canaux confondus : ca rate
    une ressource au centre du champ de vision et compte un mur (canal murs
    = 1 > 0) comme une ressource. Le padding hors grille vaut -1, ecarte par
    le test > 0."""
    o = np.asarray(obs)
    g = np.asarray(channels)
    if o.ndim == 5:                      # (T, N, side, side, C)
        return (o[..., g] > 0).any(axis=(2, 3, 4))
    if o.ndim == 4:                      # (T, N, k, C)
        return (o[..., g] > 0).any(axis=(2, 3))
    if o.ndim == 3:                      # (T, N, k*C) aplati (channel-minor)
        k = o.shape[2] // n_channels     # nb de cases par canal
        idx = np.concatenate([np.arange(c, k * n_channels, n_channels) for c in g])
        return (o[:, :, idx] > 0).any(axis=2)
    raise ValueError(f"forme d'obs inattendue : {o.shape}")


def default_energy_bins(cfg):
    """Bins fixes ancres aux seuils physiques (comparables entre chunks).
      e_die = energy_to_die      -> frontiere letale
      e_rep = min_energy_repr    -> seuil de reproduction
      e_max = energy_max         -> plafond dur (new_energy est clampe dessus)

    Resolution fine entre e_die et e_rep, la ou la decision est interessante.

    Le HAUT est ancre sur e_max, pas extrapole depuis e_rep. L'ancienne version
    posait ses bornes a e_rep + 0.5*span et e_rep + 1.5*span, soit 9 et 15 pour
    la config actuelle (e_rep=6) : au-dessus du plafond 8, donc deux bins
    toujours VIDES et un bin [6, 9) qui agregeait tout le haut de la gamme.

    Le dernier bin [e_max, inf) isole les agents colles au plafond. Ce n'est pas
    un residu : new_energy = min(..., energy_max) fait que ce sont exactement
    ceux qui mangent en continu, une population a part qui tirait vers le haut
    le bin large precedent."""
    e_die = float(cfg.energy_to_die)
    e_rep = float(cfg.min_energy_repr)
    span  = e_rep - e_die
    e_max = float(getattr(cfg, "energy_max", e_rep + span))

    edges = [-np.inf, e_die,
             e_die + 0.25 * span,
             e_die + 0.50 * span,
             e_die + 0.75 * span,
             e_rep]
    if e_max > e_rep:                       # garde-fou si le plafond est sous le seuil
        edges += [0.5 * (e_rep + e_max), e_max]
    edges.append(np.inf)
    return np.array(edges)


def _ate_within_window(ate_col, window):
    """ate_col (L,) booleen -> ateW (L,) : y a-t-il consommation dans [t, t+W] ?
    OU glissant sur les W pas suivants, par decalages successifs (W petit)."""
    ateW = ate_col.copy()
    for d in range(1, window + 1):
        ateW[:-d] |= ate_col[d:]
    return ateW


def energy_response_accumulate(saw, ate_step, energy, slot, birth_row, death_row,
                               bins, window=ENERGY_EAT_WINDOW):
    """Accumule (n, k) par bin d'energie sur tous les events "ressource vue".
      n[b] = nb de pas ou une ressource est vue, energie dans le bin b
      k[b] = parmi ceux-la, nb ou l'agent mange dans [t, t+W]
    Retourne (n, k), tableaux (n_bins,)."""
    n_bins = len(bins) - 1
    n = np.zeros(n_bins, dtype=np.int64)
    k = np.zeros(n_bins, dtype=np.int64)

    for e in range(slot.size):
        s, lo, hi = slot[e], int(birth_row[e]), int(death_row[e])
        if hi < lo:
            continue
        saw_col = saw[lo:hi + 1, s]                      # (L,)
        if not saw_col.any():
            continue
        ateW = _ate_within_window(ate_step[lo:hi + 1, s], window)
        e_col = energy[lo:hi + 1, s]

        idx    = np.where(saw_col)[0]                    # pas ou une ressource est vue
        e_seen = e_col[idx]
        y_seen = ateW[idx]                               # a mange dans la fenetre

        b = np.digitize(e_seen, bins) - 1                # bin de chaque event
        b = np.clip(b, 0, n_bins - 1)
        np.add.at(n, b, 1)
        np.add.at(k, b, y_seen.astype(np.int64))
    return n, k


def energy_response_over_envs(outputs_lab, cfg, bins=None, window=ENERGY_EAT_WINDOW,
                              rew_lag=1):
    """Accumule (bins, n, k) sur TOUS les agents de TOUS les environnements d'un
    lab. Fonction libre : passe-lui outputs_lab (B, T, N, ...) et cfg.
      n[b] = pas ou une ressource est vue, energie dans le bin b
      k[b] = parmi eux, l'agent mange dans [t, t+window]
    Pool sur toute la population -> une courbe P = k/n par lab."""
    bins  = default_energy_bins(cfg) if bins is None else np.asarray(bins, float)
    n_tot = np.zeros(len(bins) - 1, dtype=np.int64)
    k_tot = np.zeros(len(bins) - 1, dtype=np.int64)

    alive_all  = np.asarray(outputs_lab.alive)      # (B, T, N)
    energy_all = np.asarray(outputs_lab.energy)     # (B, T, N)
    obs_all    = np.asarray(outputs_lab.obs)
    rew_all    = np.asarray(outputs_lab.rewards)    # (B, T, N) ou (B, T, N, 1)
    if rew_all.ndim == 4 and rew_all.shape[-1] == 1:
        rew_all = rew_all[..., 0]

    # canaux conditionnants : les ressources BENEFIQUES (l'ancienne version codait
    # "canal 0" en dur, soit une ressource sur trois, et sur le mauvais axe)
    delta_e    = np.array([r.delta_energy for r in cfg.resources])
    channels   = np.where(delta_e > 0)[0]
    n_channels = len(cfg.resources) + 2          # ressources + agents + murs

    B = alive_all.shape[0]
    for b in range(B):
        saw = resource_in_view(obs_all[b], channels, n_channels)   # (T, N)
        ate = np.zeros_like(saw, dtype=bool)                # consommation alignee
        if rew_lag > 0:
            ate[:-rew_lag] = rew_all[b, rew_lag:] > 0
        else:
            ate = rew_all[b] > 0

        alive, energy = alive_all[b], energy_all[b]
        for s in range(alive.shape[1]):
            idx = np.where(alive[:, s] == 1)[0]             # intervalle vivant du slot
            if idx.size == 0:
                continue
            nb, kb = energy_response_accumulate(
                saw, ate, energy,
                np.array([s]), np.array([idx[0]]), np.array([idx[-1]]),
                bins, window)
            n_tot += nb
            k_tot += kb
    return bins, n_tot, k_tot


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