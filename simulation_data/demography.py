"""Démographie / écologie.

Accumule, chunk par chunk, les séries temporelles population / ressources /
mouvement / durée de vie, les sauve sur disque, et fournit la condition d'arrêt
de la simulation.
"""

import glob
import os
import re

import numpy as np

from simulation.utils.plots import compute_mean_movement_chunk, compute_lifetime_chunk
from simulation.utils.utils_sim import classify_outcome

# W : pas apres l'observation ou la consommation compte encore. Doit rester egal
# a ENERGY_EAT_WINDOW (energy_response.py), sinon la sim principale et le lab ne
# mesurent plus la meme chose. = 2 * cfg.agent_view, cf. le commentaire la-bas.
EAT_WINDOW = 10


def compute_seen_eaten_chunk(outputs, window=EAT_WINDOW):
    """(T, n_types) x2 : n = agents voyant le type k, k_ = ceux qui le mangent.

    On conditionne sur "type k dans le champ de vision au pas t" et on regarde
    s'il y a consommation dans [t, t+window] : voir une ressource et l'atteindre
    prend le temps d'y marcher, donc exiger la consommation au pas t sous-estime
    massivement la propension (meme fenetre que energy_response).

    Reduit des le chunk : garder (T, N, n_types) pour tout le run ferait
    2000 agents x 1000 pas x 3 types et par chunk."""
    saw = np.asarray(outputs.saw_res).astype(bool)      # (T, N, n_types)
    ate = np.asarray(outputs.ate_res).astype(bool)      # (T, N, n_types)

    ate_w = ate.copy()                                  # consommation dans [t, t+W]
    for d in range(1, window + 1):
        ate_w[:-d] |= ate[d:]

    n_seen  = saw.sum(axis=1)                           # (T, n_types)
    n_eaten = (saw & ate_w).sum(axis=1)                 # (T, n_types)
    return n_seen, n_eaten


def compute_action_fractions_chunk(outputs):
    """(T, n_actions) : part des agents VIVANTS ayant choisi chaque action.

    Reduit des le chunk, comme les autres series : garder (T, N, n_actions) sur
    tout le run coûterait 1000 pas x 2000 agents x 4 par chunk pour un trace qui
    n'a besoin que des fractions.

    Le slot 0 est exclu (index de gestion JAX), et un agent dont le one-hot est
    entierement nul -- il n'a pas encore agi -- est retire du denominateur au
    lieu d'etre compte comme "rester", ce qu'un argmax naif ferait.
    """
    acts  = np.asarray(outputs.actions)          # (T, N, n_actions), one-hot
    alive = np.asarray(outputs.alive).astype(bool).copy()   # (T, N)
    alive[:, 0] = False
    pris = acts.sum(axis=-1) > 0                 # (T, N) a agi
    m = (alive & pris)[..., None]                # (T, N, 1)
    compte = (acts * m).sum(axis=1)              # (T, n_actions)
    total = compte.sum(axis=1, keepdims=True)
    return np.divide(compte, total, out=np.zeros_like(compte, dtype=float),
                     where=total > 0)


class DemographyMixin:

    def _init_demography(self):
        self.pop_history = []
        self.res_history = []
        self.oracle_history = []
        self.consumed_history = []
        self.seen_history = []
        self.eaten_seen_history = []
        self.mov_history = []
        self.life_history = []
        self.action_history = []

    # Les series sauvees par chunk, et le nom de l'historique qui les recoit.
    # L'axe de concatenation differe : mean_life est (2, n_morts), donc empilee
    # sur l'axe 1, toutes les autres sur l'axe 0.
    SERIES = (("population",    "pop_history",        0),
              ("resources",     "res_history",        0),
              ("oracles",       "oracle_history",     0),
              ("consumed",      "consumed_history",   0),
              ("n_seen",        "seen_history",       0),
              ("n_eaten_seen",  "eaten_seen_history", 0),
              ("mean_movement", "mov_history",        0),
              ("mean_life",     "life_history",       1),
              ("action_frac",   "action_history",     0))

    def charger_historique(self, source_dir, jusqu_au_chunk):
        """Reprend les series d'un run precedent, pour les chunks < jusqu_au_chunk.

        Sans ca, une reprise repart avec des historiques VIDES : les figures ne
        couvrent que la fenetre depuis la reprise, et life_expectancy est en plus
        biaisee -- ses casiers de naissance anciens ne retiennent que les agents
        morts APRES la reprise, c'est-a-dire les plus vieux, ce qui tire la
        mediane vers le haut.

        Rend le pas de depart a poser dans start_step : les figures indexent
        l'historique depuis cette valeur, la laisser au chunk de reprise
        decalerait tout l'axe des abscisses de la partie rechargee.
        """
        data_dir = os.path.join(source_dir, "data")
        fichiers = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        repris = []
        for f in fichiers:
            m = re.search(r"chunk_(\d+)\.npz$", f)
            if not m or int(m.group(1)) >= jusqu_au_chunk:
                continue
            d = np.load(f)
            manquantes = [k for k, _, _ in self.SERIES if k not in d.files]
            if manquantes:
                # un run anterieur a l'ajout d'une serie : on ne recharge rien
                # plutot que de decaler les series entre elles
                print(f"[historique] {os.path.basename(f)} sans {manquantes} : "
                      "chargement abandonne, les figures repartiront de la reprise")
                return None
            for cle, nom, _ in self.SERIES:
                getattr(self, nom).append(d[cle])
            repris.append(int(m.group(1)))
            d.close()

        if not repris:
            print(f"[historique] rien a reprendre dans {data_dir}")
            return None
        print(f"[historique] {len(repris)} chunk(s) repris "
              f"({min(repris)} a {max(repris)})")
        return (min(repris) - 1) * self.cfg.chunk_size

    def update_data_with_chunk(self, outputs, data_dir,chunk_idx):
        self.chunk_idx = chunk_idx
        pop_chunk  = np.array(outputs.alive.sum(axis=1))          # (T,)
        # envahisseurs vivants, pour la courbe d'invasion
        oracle_chunk = np.array((outputs.alive * outputs.is_oracle).sum(axis=1))
        n_types = len(self.cfg.resources)
        res_chunk = np.array(outputs.grid[:, :n_types, :, :].sum(axis=(2, 3)))
        consumed_chunk = np.array(outputs.consumed_res)           # (T, n_types)
        seen_chunk, eaten_seen_chunk = compute_seen_eaten_chunk(outputs)
        action_chunk = compute_action_fractions_chunk(outputs)    # (T, n_actions)
        mov_chunk  = compute_mean_movement_chunk(outputs, self.cfg.grid_length)
        life_chunk = compute_lifetime_chunk(outputs, self.cfg)

        self.pop_history.append(pop_chunk)
        self.res_history.append(res_chunk)
        self.oracle_history.append(oracle_chunk)
        self.consumed_history.append(consumed_chunk)
        self.seen_history.append(seen_chunk)
        self.eaten_seen_history.append(eaten_seen_chunk)
        self.mov_history.append(mov_chunk)
        self.life_history.append(life_chunk)
        self.action_history.append(action_chunk)

        np.savez(
            os.path.join(data_dir, f"chunk_{self.chunk_idx:05d}.npz"),
            population    = pop_chunk,
            resources     = res_chunk,
            oracles       = oracle_chunk,
            consumed      = consumed_chunk,
            n_seen        = seen_chunk,
            n_eaten_seen  = eaten_seen_chunk,
            mean_movement = mov_chunk,
            mean_life     = life_chunk,
            action_frac   = action_chunk,
        )

    def check_end_condition(self):
        pop_full = np.concatenate(self.pop_history)
        res_full = np.concatenate(self.res_history)
        current_sim_state = classify_outcome(pop_full, res_full, self.cfg)
        return current_sim_state