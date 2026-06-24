from simulation.pca import update_genealogy,load_clade_snapshots,find_root,collect_clade,plot_clade_pca_html,save_alive_snapshot,plot_clade_pca_html_res
from simulation.mrca import coalescence_point,plot_tmrca_gen
from simulation.plots import plot_evolution,plot_current_config, compute_mean_movement_chunk,plot_mean_movement,compute_lifetime_chunk,plot_lifetime_vs_step,plot_life_expectancy,plot_phase_portrait_png
from simulation.utils import classify_outcome
from simulation.lab_env import launch_one_agent_one_env,vmap_over_agents_env_lab
from simulation.utils import _video_worker,outputs_to_numpy

import os
import numpy as np
import jax
class simulation_data :
    def __init__(self,cfg,start_step):
        self.pop_history = []
        self.res_history = []
        self.mov_history = []
        self.life_history = []
        self.node_parent = {}
        self.node_children = {}
        self.tmrca_gen = []
        self.prev_born = None
        self.prev_parent = None
        self.cfg = cfg
        self.chunk_idx=0
        self.start_step=start_step

    def update_data_with_chunk(self,outputs,data_dir):
        self.chunk_idx+=1

        pop_chunk      = np.array(outputs.agents.alive.sum(axis=1))
        res_chunk      = np.array(outputs.grid[:, 0, :, :].sum(axis=(1, 2)))
        mov_chunk      = compute_mean_movement_chunk(outputs, self.cfg.n)
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
            mean_life = life_chunk
        )

    def plot(self,state,exp_dir):
        plot_evolution(
            np.concatenate(self.pop_history, axis=0),
            np.concatenate(self.res_history, axis=0),
            exp_dir,
            self.start_step
        )
        plot_phase_portrait_png(
            np.concatenate(self.pop_history, axis=0),
            np.concatenate(self.res_history, axis=0),
            exp_dir,
            self.start_step
        )
        life_data = np.concatenate(self.life_history, axis=1)
        plot_mean_movement(np.concatenate(self.mov_history)[1500:],np.concatenate(self.res_history, axis=0)[1500:], exp_dir, self.start_step)
        plot_current_config(state.grid[0, :, :], state.grid[-1, :, :],state.agents.position, state.agents.alive, exp_dir,name_fig=f'{self.chunk_idx}')
        plot_lifetime_vs_step((life_data[1,:]), (life_data[0,:]), exp_dir, self.cfg)
        plot_life_expectancy((life_data[1,:]), (life_data[0,:]), exp_dir, bin_width=10)

    def check_end_condition(self):
        pop_full = np.concatenate(self.pop_history)
        res_full = np.concatenate(self.res_history)
        current_sim_state = classify_outcome(pop_full, res_full, self.cfg)
        return current_sim_state


    def update_mrca(self,outputs,exp_dir):
        outputs_mcra = coalescence_point(outputs, self.node_parent)
        tmrca_generations = outputs_mcra['tmrca_generations']
        self.tmrca_gen.append(tmrca_generations)
        plot_tmrca_gen(np.concatenate(self.pop_history, axis=0),self.tmrca_gen,exp_dir)

    def update_genealogy(self,outputs,exp_dir):
        self.prev_born, self.prev_parent = update_genealogy(
        outputs, self.node_parent, self.node_children, self.prev_born, self.prev_parent)   # arbre complet, pas cher
        save_alive_snapshot(outputs, self.chunk_idx, os.path.join(exp_dir,'params'))

    def compute_survivors(self,outputs):
        alive_last = np.array(outputs.agents.alive)[-1]
        born_last  = np.array(outputs.agents.born_step)[-1]
        survivors  = [(i, int(born_last[i])) for i in range(1, len(alive_last))
                        if alive_last[i] == 1]
        return survivors

    def plot_pca_and_mrca_and_lab_env(self,outputs,data_dir,exp_dir):
        np.savez(
            os.path.join(data_dir, f"tmrca.npz"),
            tmrca    = np.array(self.tmrca_gen),
        )
        survivors= self.compute_survivors(outputs)
        if survivors:
            root  = find_root(survivors[0], self.node_parent)
            clade = collect_clade(root, self.node_children)
            name_save=list(np.arange(self.chunk_idx-(self.cfg.pca)//2,self.chunk_idx))
            node_params = load_clade_snapshots(clade, os.path.join(exp_dir,'params'),name_save)
            plot_clade_pca_html(node_params, os.path.join(exp_dir),name_fig=f'{self.chunk_idx}')

    def launch_env(self,outputs,key,model,exp_dir,n):
        survivors= self.compute_survivors(outputs)
        agent_params = outputs.agents.params[survivors[0][0]][-n:]
        print(np.array(agent_params).shape)
        final_state,outputs= vmap_over_agents_env_lab(agent_params,key,self.cfg,model)
        outputs_np = outputs_to_numpy(outputs)

        agg, summary = self.data_lab_env(outputs_lab=outputs)

        #Save only two videos
        vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{self.chunk_idx+1}_lab_0.mp4")
        _video_worker(outputs_np[0], vid_path, 20, 10)
        vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{self.chunk_idx+1}_lab_1.mp4")
        _video_worker(outputs_np[1], vid_path, 20, 10)


    def _per_agent_metrics(self, outputs):
        """Métriques par agent MORT, alignées par événement de mort.
        Toutes les sorties sont des tableaux 1D de longueur D (nb de morts)."""
        n = self.cfg.n

        alive  = np.asarray(outputs.agents.alive)        # (T, N)
        born   = np.asarray(outputs.agents.born_step)     # (T, N) step de naissance
        step   = np.asarray(outputs.step)                 # (T,)   step global de la ligne
        pos    = np.asarray(outputs.agents.position)      # (T, N, 2)
        rew    = np.asarray(outputs.rewards)              # (T, N) consommation / pas
        energy = np.asarray(outputs.agents.energy)        # (T, N)  <-- VÉRIFIE le nom du champ

        # 1) Détection des morts sur la transition t -> t+1
        a_t, a_tp1 = alive[:-1], alive[1:]
        b_t, b_tp1 = born[:-1],  born[1:]
        death_event = (a_t == 1) & ((a_tp1 == 0) | (b_tp1 != b_t))   # (T-1, N)

        t_row, slot = np.where(death_event)        # indices (locaux) ligne + slot, (D,)
        if t_row.size == 0:
            return None

        # 2) Naissance -> indice de ligne locale + durée de vie
        b_dead    = b_t[t_row, slot]                       # step GLOBAL de naissance
        birth_row = (b_dead - step[0]).astype(int)         # step global -> indice de ligne
        death_row = t_row                                  # dernière ligne vivant
        age       = step[t_row] - b_dead                   # durée de vie en pas, (D,)

        # somme inclusive sur [lo, hi] par différence de cumsum (capture slot)
        def window_sum(cum, lo, hi):
            hi_v = cum[hi, slot]
            lo_v = np.where(lo >= 0, cum[np.clip(lo, 0, None), slot], 0.0)
            return hi_v - lo_v

        # 3) Consommation (grandeur "par état") -> fenêtre [birth_row, death_row]
        cum_rew   = np.cumsum(rew, axis=0)                          # (T, N)
        total_rew = window_sum(cum_rew, birth_row - 1, death_row)
        mean_rew  = total_rew / (age + 1)          # age+1 = nb de lignes vécues

        # 4) Mouvement (grandeur "par transition") -> fenêtre [birth_row, death_row-1]
        delta = pos[1:] - pos[:-1]                                  # (T-1, N, 2)
        delta = np.where(delta >  n // 2, delta - n, delta)         # tore : image minimale
        delta = np.where(delta < -(n // 2), delta + n, delta)
        mag   = np.sqrt((delta ** 2).sum(axis=-1))                 # (T-1, N)

        cum_mag    = np.cumsum(mag, axis=0)                         # (T-1, N)
        total_move = window_sum(cum_mag, birth_row - 1, np.clip(death_row - 1, 0, None))
        total_move = np.where(age > 0, total_move, 0.0)
        mean_speed = np.where(age > 0, total_move / age, 0.0)

        # 5) Mort "par mur" : énergie encore franchement au-dessus du critique au dernier pas vivant
        energy_at_death = energy[death_row, slot]
        seuil = self.cfg.energy_critical + self.cfg.energy_step_cost   # <-- marge (cf. note)
        wall_death = energy_at_death > seuil                        # bool (D,)

        return {
            "slot":        slot,            # pour recroiser avec la généalogie (slot, born)
            "born":        b_dead,
            "age":         age,             # durée de vie
            "total_rew":   total_rew,       # consommation totale
            "mean_rew":    mean_rew,        # consommation moyenne / pas
            "total_move":  total_move,      # distance parcourue
            "mean_speed":  mean_speed,      # mouvement moyen / pas
            "energy_end":  energy_at_death,
            "wall_death":  wall_death,      # True = mort non liée à la famine
        }


    def data_lab_env(self, outputs_lab):
        keys = ["age", "total_rew", "mean_rew", "total_move",
                "mean_speed", "energy_end", "wall_death"]
        agg = {k: [] for k in keys}

        for outputs in outputs_lab:
            m = self._per_agent_metrics(outputs)
            if m is None:
                continue
            for k in keys:
                agg[k].append(m[k])

        agg = {k: (np.concatenate(v) if v else np.array([])) for k, v in agg.items()}

        summary = {
            "n_morts":          int(agg["age"].size),
            "duree_vie_moy":    agg["age"].mean(),
            "consommation_moy": agg["mean_rew"].mean(),
            "mouvement_moy":    agg["mean_speed"].mean(),
            "frac_mort_mur":    agg["wall_death"].mean(),   # proportion de morts non-famine
        }
        return agg, summary
            
            
            
            



    def save_mrca_end_sim(self,data_dir):
        np.savez(
            os.path.join(data_dir, f"tmrca.npz"),
            tmrca    = np.array(self.tmrca_gen),
        )