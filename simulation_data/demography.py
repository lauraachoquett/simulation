"""Démographie / écologie.

Accumule, chunk par chunk, les séries temporelles population / ressources /
mouvement / durée de vie, les sauve sur disque, et fournit la condition d'arrêt
de la simulation.
"""

import os
import numpy as np

from simulation.utils.plots import compute_mean_movement_chunk, compute_lifetime_chunk
from simulation.utils.utils_sim import classify_outcome


class DemographyMixin:

    def _init_demography(self):
        self.pop_history = []
        self.res_history = []
        self.mov_history = []
        self.life_history = []

    def update_data_with_chunk(self, outputs, data_dir):
        self.chunk_idx += 1
        pop_chunk  = np.array(outputs.alive.sum(axis=1))          # (T,)
        n_types = len(self.cfg.resources)
        res_chunk = np.array(outputs.grid[:, :n_types, :, :].sum(axis=(2, 3))) 
        mov_chunk  = compute_mean_movement_chunk(outputs, self.cfg.grid_length)
        life_chunk = compute_lifetime_chunk(outputs, self.cfg)

        self.pop_history.append(pop_chunk)
        self.res_history.append(res_chunk)
        self.mov_history.append(mov_chunk)
        self.life_history.append(life_chunk)

        np.savez(
            os.path.join(data_dir, f"chunk_{self.chunk_idx+1:05d}.npz"),
            population    = pop_chunk,
            resources     = res_chunk,
            mean_movement = mov_chunk,
            mean_life     = life_chunk,
        )

    def check_end_condition(self):
        pop_full = np.concatenate(self.pop_history)
        res_full = np.concatenate(self.res_history)
        current_sim_state = classify_outcome(pop_full, res_full, self.cfg)
        return current_sim_state