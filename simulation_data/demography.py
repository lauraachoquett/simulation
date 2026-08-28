"""Démographie / écologie.

Accumule, chunk par chunk, les séries temporelles population / ressources /
mouvement / durée de vie, les sauve sur disque, et fournit la condition d'arrêt
de la simulation.
"""

import os
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
        self.perte_history = []

    def update_data_with_chunk(self, outputs, data_dir,chunk_idx):
        self.chunk_idx = chunk_idx
        pop_chunk  = np.array(outputs.alive.sum(axis=1))          # (T,)
        # envahisseurs vivants, pour la courbe d'invasion
        oracle_chunk = np.array((outputs.alive * outputs.is_oracle).sum(axis=1))
        n_types = len(self.cfg.resources)
        res_chunk = np.array(outputs.grid[:, :n_types, :, :].sum(axis=(2, 3)))
        consumed_chunk = np.array(outputs.consumed_res)           # (T, n_types)
        seen_chunk, eaten_seen_chunk = compute_seen_eaten_chunk(outputs)
        # Erreur de prediction, moyennee sur les agents vivants qui ont mange
        # dans la fenetre. Les autres portent un NaN (cf. one_simulation) et sont
        # ecartes : les compter comme zero ferait croire a une prediction parfaite.
        if self.cfg.inner_loop:
            perte = np.asarray(outputs.perte_pred)             # (T, N)
            vivants = np.asarray(outputs.alive) > 0
            valides = np.where(vivants & np.isfinite(perte), perte, np.nan)
            n_valides = np.isfinite(valides).sum(axis=1)
            perte_chunk = np.divide(np.nansum(valides, axis=1), n_valides,
                                    out=np.full(len(pop_chunk), np.nan),
                                    where=n_valides > 0)
        else:
            perte_chunk = np.zeros(len(pop_chunk))
        mov_chunk  = compute_mean_movement_chunk(outputs, self.cfg.grid_length)
        life_chunk = compute_lifetime_chunk(outputs, self.cfg)

        self.pop_history.append(pop_chunk)
        self.res_history.append(res_chunk)
        self.oracle_history.append(oracle_chunk)
        self.consumed_history.append(consumed_chunk)
        self.seen_history.append(seen_chunk)
        self.eaten_seen_history.append(eaten_seen_chunk)
        self.perte_history.append(perte_chunk)
        self.mov_history.append(mov_chunk)
        self.life_history.append(life_chunk)

        np.savez(
            os.path.join(data_dir, f"chunk_{self.chunk_idx:05d}.npz"),
            population    = pop_chunk,
            resources     = res_chunk,
            oracles       = oracle_chunk,
            consumed      = consumed_chunk,
            n_seen        = seen_chunk,
            n_eaten_seen  = eaten_seen_chunk,
            perte_pred    = perte_chunk,
            mean_movement = mov_chunk,
            mean_life     = life_chunk,
        )

    def check_end_condition(self):
        pop_full = np.concatenate(self.pop_history)
        res_full = np.concatenate(self.res_history)
        current_sim_state = classify_outcome(pop_full, res_full, self.cfg)
        return current_sim_state