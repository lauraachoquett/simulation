"""Assemblage de simulation_data.

Recombine les quatre mixins thématiques en un seul objet partageant un unique
`self`. Ne contient que les deux vrais points de coordination transverse :
- __init__  : construit l'état en déléguant à chaque _init_<thème>() ;
- plot      : orchestrateur qui mêle graphes démographiques et, si la
              population est coalescée, les métriques de poids.
"""

import numpy as np
import json
from simulation.data_class import BASE_RESOURCES
import os 

from simulation.utils.plots import (
    plot_evolution,
    plot_consumption,
    plot_phase_portrait_png,
    plot_mean_movement,
    plot_current_config,
    plot_lifetime_vs_step,
    plot_life_expectancy,
)

from simulation.utils.utils_sim import load_shuffle_log
from .demography import DemographyMixin
from .genealogy import GenealogyMixin
from .weights import WeightsMixin
from .lab import LabMixin


class simulation_data(DemographyMixin, GenealogyMixin, WeightsMixin, LabMixin):

    def __init__(self, cfg, start_step,start_chunk):
        self.cfg = cfg
        self.start_step = start_step
        self.chunk_idx = start_chunk

        self._init_demography()   # chaque mixin pose SON propre état
        self._init_genealogy()
        self._init_weights()
        # LabMixin n'a pas d'état persistant -> pas de _init_lab()

    def plot(self, state, exp_dir):
        
        shuffle_log       = load_shuffle_log(exp_dir)
        initial_order_ids = [r.id for r in BASE_RESOURCES]# l'ordre au step 0
        plot_evolution(
            np.concatenate(self.pop_history, axis=0),
            np.concatenate(self.res_history, axis=0),
            exp_dir,
            shuffle_log,
            initial_order_ids,
            self.start_step,
        )
        pop_full      = np.concatenate(self.pop_history, axis=0)
        consumed_full = np.concatenate(self.consumed_history, axis=0)

        plot_consumption(
            pop_full, consumed_full, exp_dir, shuffle_log, initial_order_ids,
            self.start_step, window=100, name_fig='plot_conso_window_mean',
        )
        plot_phase_portrait_png(
            np.concatenate(self.pop_history, axis=0),
            np.concatenate(self.res_history, axis=0),
            exp_dir,
            self.cfg,
            self.start_step,
        )
        life_data = np.concatenate(self.life_history, axis=1)
        plot_mean_movement(
            np.concatenate(self.mov_history)[1500:],
            exp_dir, self.start_step,
        )
        n_types = len(self.cfg.resources)
        grid_res = state.grid[:n_types, :, :]       
        plot_current_config(
            grid_res, state.grid[-1, :, :],
            state.agents.position, state.agents.alive,
            exp_dir, self.cfg.resources,name_fig=f'{self.chunk_idx}',
        )
        plot_lifetime_vs_step((life_data[1, :]), (life_data[0, :]), exp_dir, self.cfg)
        plot_life_expectancy((life_data[1, :]), (life_data[0, :]), exp_dir, bin_width=1000)
        if self.coalesced:                    # -> WeightsMixin
            print("PLOT metrics")
            self.plot_weight_metrics(exp_dir)