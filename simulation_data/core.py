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
    block_edges, block_apply, block_steps,
    plot_evolution,
    plot_consumption,
    plot_prob_eat_given_seen,
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

        # Toutes les series d'historique sont reduites par blocs AVANT tracage.
        # Sans ca, chaque figure coute lineairement en longueur de run pour un
        # rendu identique (une figure de 1200 px ne resout pas 1,6 M de points),
        # et plot() etant appele periodiquement, le total est quadratique.
        # Les donnees pleine resolution restent dans les .npz : tools/replot.py
        # les rejoue a la demande.
        pop_full      = np.concatenate(self.pop_history, axis=0)
        res_full      = np.concatenate(self.res_history, axis=0)
        consumed_full = np.concatenate(self.consumed_history, axis=0)
        seen_full     = np.concatenate(self.seen_history, axis=0)
        eaten_full    = np.concatenate(self.eaten_seen_history, axis=0)

        edges = block_edges(len(pop_full), self.start_step,
                            cut_steps=[e["step"] for e in shuffle_log])
        steps_r = block_steps(edges, self.start_step)
        pop_r      = block_apply(pop_full,      edges, "mean")
        res_r      = block_apply(res_full,      edges, "mean")
        consumed_r = block_apply(consumed_full, edges, "mean")
        # comptes -> SOMME, le ratio se fait ensuite (cf. block_apply)
        seen_r     = block_apply(seen_full,  edges, "sum")
        eaten_r    = block_apply(eaten_full, edges, "sum")

        plot_evolution(
            pop_r, res_r, exp_dir, shuffle_log, initial_order_ids,
            self.start_step, steps=steps_r,
        )
        # window=1 : le bloc a deja lisse, un second lissage ferait double emploi
        plot_consumption(
            pop_r, consumed_r, exp_dir, shuffle_log, initial_order_ids,
            self.start_step, window=1, name_fig='plot_conso_window_mean',
            steps=steps_r,
        )
        plot_prob_eat_given_seen(
            pop_r, seen_r, eaten_r, exp_dir, shuffle_log, initial_order_ids,
            self.start_step, window=1, steps=steps_r,
        )
        plot_phase_portrait_png(
            pop_r, res_r, exp_dir, self.cfg, self.start_step, steps=steps_r,
        )
        life_data = np.concatenate(self.life_history, axis=1)
        # les 1500 premiers pas sont ecartes : apres une reprise il peut ne
        # rester aucun point avant plusieurs chunks, il n'y a alors rien a tracer
        mov_full = np.concatenate(self.mov_history)[1500:]
        if mov_full.size:
            mov_edges = block_edges(len(mov_full), self.start_step + 1500,
                                    cut_steps=[e["step"] for e in shuffle_log])
            plot_mean_movement(
                block_apply(mov_full, mov_edges, "mean"),
                exp_dir, self.start_step,
                steps=block_steps(mov_edges, self.start_step + 1500),
            )
        n_types = len(self.cfg.resources)
        grid_res = state.grid[:n_types, :, :]       
        plot_current_config(
            grid_res, state.grid[-1, :, :],
            state.agents.position, state.agents.alive,
            exp_dir, self.cfg.resources,name_fig=f'{self.chunk_idx}',
        )
        # plot_lifetime_vs_step((life_data[1, :]), (life_data[0, :]), exp_dir, self.cfg)
        plot_life_expectancy((life_data[1, :]), (life_data[0, :]), exp_dir, bin_width=1000)
        if self.cfg.track_weights and self.coalesced:   # -> WeightsMixin
            print("PLOT metrics")
            self.plot_weight_metrics(exp_dir,x_axis='generation')