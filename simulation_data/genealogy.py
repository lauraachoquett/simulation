
"""Généalogie et ancêtre commun (MRCA).

Construit incrémentalement l'arbre (node_parent / node_children), détecte la
coalescence de la population (tous les vivants descendent d'un même ancêtre)
et déclenche les tracés associés (TMRCA, PCA de clade).

Couplage : update_genealogy appelle self.update_weight_metrics (WeightsMixin)
quand la population est coalescée.
"""

import os
import numpy as np


from simulation.genealogy.genealogy import update_genealogy
from simulation.genealogy.lod import (segment_fixe, params_du_segment,
                                      enregistre, capture_vivants,
                                      elague_cache, feuilles_vivantes)
from simulation.genealogy.pca import save_alive_snapshot
from simulation.genealogy.r0 import plot_r0, r0_by_birth_window
from simulation.genealogy.mrca import coalescence_point, plot_tmrca_gen


class GenealogyMixin:

    def _init_genealogy(self):
        self.node_parent = {}
        self.node_children = {}
        self.tmrca_gen = []
        self.tmrca_pas = []        # age du MRCA en pas de simulation
        self.prev_born = None
        self.prev_parent = None
        self.coalesced = False
        self.mrca_prev = None      # dernier MRCA enregistre dans la lignee
        self.lod_cache = {}        # (slot, born) -> genome, candidats a la lignee

    def update_genealogy(self, outputs,state, exp_dir):
        self.prev_born, self.prev_parent = update_genealogy(
            outputs, self.node_parent, self.node_children,
            self.prev_born, self.prev_parent)   # arbre complet, pas cher
        
        if self.cfg.track_weights:
            if self.coalesced:
                self.update_weight_metrics(state)      # -> WeightsMixin
            save_alive_snapshot(state, self.chunk_idx, os.path.join(exp_dir, 'params'))

        if self.cfg.track_lod:
            # en memoire, et seulement les candidats a la lignee
            capture_vivants(self.lod_cache, state)
            elague_cache(self.lod_cache, feuilles_vivantes(state),
                         self.node_parent, self.mrca_prev)

    def update_mrca_and_plot(self, outputs, exp_dir):
        outputs_mcra = coalescence_point(outputs, self.node_parent)
        self.coalesced = outputs_mcra['coalesced']
        self.tmrca_gen.append(outputs_mcra['tmrca_generations'])
        self.tmrca_pas.append(outputs_mcra.get('tmrca_pas_de_simu'))
        if self.coalesced:
            self.sauve_lignee(outputs_mcra['mrca'], exp_dir)
            plot_tmrca_gen(np.concatenate(self.pop_history, axis=0), self.tmrca_gen,
                           exp_dir, tmrca_pas=self.tmrca_pas)

    def sauve_lignee(self, mrca, exp_dir):
        """Enregistre les ancetres fixes depuis le dernier changement de MRCA."""
        if mrca is None or mrca == self.mrca_prev:
            return
        segment, rupture = segment_fixe(mrca, self.mrca_prev, self.node_parent)
        if not segment:
            self.mrca_prev = mrca
            return

        params = {n: self.lod_cache[n] for n in segment if n in self.lod_cache}
        manquants = [n for n in segment if n not in params]
        if manquants and os.path.isdir(os.path.join(exp_dir, "params")):
            # filet quand --weights ecrit params/
            deb = max(1, int(segment[0][1]) // self.cfg.chunk_size)
            params.update(params_du_segment(manquants,
                                            os.path.join(exp_dir, "params"),
                                            range(deb, self.chunk_idx + 1)))
        n = enregistre(exp_dir, self.chunk_idx,
                       self.chunk_idx * self.cfg.chunk_size,
                       mrca, self.mrca_prev, segment, params, rupture)
        print(f"[lod] chunk {self.chunk_idx} : {len(segment)} ancetre(s) fixe(s), "
              f"{n} genome(s) retrouve(s)"
              + ("  [rupture : changement de fondateur]" if rupture else ""),
              flush=True)
        self.mrca_prev = mrca

    def compute_survivors(self, state):
        alive_last = np.array(state.agents.alive)
        born_last  = np.array(state.agents.born_step)
        survivors  = [(i, int(born_last[i])) for i in range(1, len(alive_last))
                        if alive_last[i] == 1]
        return survivors

    def plot_pca(self, outputs, data_dir, exp_dir):
        return
        # survivors= self.compute_survivors(state)
        # if survivors:
        #     root  = find_root(survivors[0], self.node_parent)
        #     clade = collect_clade(root, self.node_children)
        #     name_save=list(np.arange(self.chunk_idx-(self.cfg.pca)//2,self.chunk_idx))
            #node_params = load_clade_snapshots(clade, os.path.join(exp_dir,'params'),name_save)
            #plot_clade_pca_html(node_params, os.path.join(exp_dir),name_fig=f'{self.chunk_idx}')

    def save_mrca_sim(self, data_dir):
        np.savez(
            os.path.join(data_dir, f"tmrca.npz"),
            tmrca=np.array(self.tmrca_gen, dtype=float),
            tmrca_pas=np.array(self.tmrca_pas, dtype=float),
        )
        
    def compute_R0_and_plot(self,state,current_step,exp_dir):
        survivors = self.compute_survivors(state)
        oldest_alive_birth = min(b for (_, b) in survivors) if survivors else current_step
        window = 2000
        
        r0_by_window=r0_by_birth_window(self.node_children, oldest_alive_birth, window=window)
        plot_r0(r0_by_window, window, exp_dir=exp_dir, smooth_w=5, fname="r0_evolution.png")