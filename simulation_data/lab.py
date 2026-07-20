"""Environnement « lab » (test contrôlé).

Rejoue les poids des survivants dans un environnement standardisé, mesure des
métriques par agent mort (durée de vie, consommation, mouvement, cause de mort)
et agrège / sauve / trace le résumé.

Trois environnements de test :
- high_res  : agent SEUL, ressources renouvelées  -> métriques de base
- low_res   : agent seul, ressources NON renouvelées -> EXPLORATION
            (temps avant la 1re ressource mangée)
- clones    : même env que high_res mais 4 clones du même génome -> EFFET
            DES PAIRS (comportement moyen des clones vs agent seul, apparié)

Bloc le plus autonome : aucun état persistant, ne lit que self.cfg et
self.chunk_idx. Appelle self.compute_survivors (GenealogyMixin).
"""

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from simulation.lab_env import vmap_over_agents_env_lab_high_res,vmap_over_agents_env_lab_low_res,vmap_over_agents_env_lab_high_res_with_clones
from simulation.utils.plots import (plot_lab_metrics, plot_lab_exploration,
                            plot_alone_vs_clones, plot_lab_energy)
from simulation.utils.utils_sim import _video_worker, outputs_to_numpy



class LabMixin:

    def launch_env(self, state, key_env, subkey_sim, model, exp_dir, n):
        survivors = self.compute_survivors(state)
        ids = np.array([agent_id for agent_id, _ in survivors[:n]])
        agent_params = state.agents.params[ids]
        key_sim = random.split(subkey_sim, len(ids))

        def agent_slice(state, b):
            return jax.tree_util.tree_map(lambda x: x[b], state)

        # ============ 1) HIGH_RES (agent seul) ============
        final_state, outputs_high = vmap_over_agents_env_lab_high_res(
            agent_params, key_env, key_sim, model, self.cfg)
        agg, summary = self.data_lab_env(outputs_lab=outputs_high)
        self._save_lab_data(agg, summary, exp_dir)
        self._print_and_plot_lab_summary(summary, exp_dir)
        self._plot_energy(outputs_high, exp_dir, "lab_1",                     # <== NOUVEAU
                          "lab_1 — high_res (agent alone)")

        vid = os.path.join(exp_dir, "videos", f"high_res_video_chunk_{self.chunk_idx+1}_lab_0.mp4")
        _video_worker(outputs_to_numpy(agent_slice(outputs_high, 0)), vid, 20, 10)
        vid = os.path.join(exp_dir, "videos", f"high_res_video_chunk_{self.chunk_idx+1}_lab_1.mp4")
        _video_worker(outputs_to_numpy(agent_slice(outputs_high, 1)), vid, 20, 10)

        # ============ 2) LOW_RES (exploration) ============
        final_state, outputs_low = vmap_over_agents_env_lab_low_res(
            agent_params, key_env, key_sim, model, self.cfg)
        agg_low, summary_low = self.data_lab_env_low_res(outputs_low)                # <== NOUVEAU
        self._save_lab_data(agg_low, summary_low, exp_dir, suffix="lowres")          # fichiers séparés
        self._print_and_plot_low_res(summary_low, exp_dir)                          # <== NOUVEAU
        self._plot_energy(outputs_low, exp_dir, "lab_2",                            # <== NOUVEAU
                          "lab_2 — low_res (exploration)")

        for b in range(3):
            vid = os.path.join(exp_dir, "videos", f"low_res_video_chunk_{self.chunk_idx+1}_lab_{b}.mp4")
            _video_worker(outputs_to_numpy(agent_slice(outputs_low, b)), vid, 20, 10)

        # ============ 3) CLONES (effet des pairs) ============
        final_state, outputs_clones = vmap_over_agents_env_lab_high_res_with_clones(
            agent_params, key_env, key_sim, model, self.cfg)
        # comparaison APPARIÉE avec l'env high_res (mêmes génomes, même ordre) :
        self.compare_alone_vs_clones(outputs_high, outputs_clones, exp_dir)          # <== NOUVEAU
        self._plot_energy(outputs_clones, exp_dir, "lab_3",                          # <== NOUVEAU
                          "lab_3 — high_res with clones")

        for b in range(3):
            vid = os.path.join(exp_dir, "videos", f"high_res_clones_video_chunk_{self.chunk_idx+1}_lab_{b}.mp4")
            _video_worker(outputs_to_numpy(agent_slice(outputs_clones, b)), vid, 20, 10)



    def _print_and_plot_lab_summary(self, summary, exp_dir):
        print(f"\n--- Lab chunk {self.chunk_idx+1} | {summary['n_morts']} morts sur  {summary['n_agents']} agents ---")
        if summary["n_morts"] == 0:
            print("  (aucune mort sur ce rollout)")
            return
        print(f"  durée de vie moy.   : {summary['duree_vie_moy']:8.2f} pas")
        print(f"  consommation moy.   : {summary['consommation_moy']:8.4f} /pas")
        print(f"  mouvement moy.      : {summary['mouvement_moy']:8.4f} /pas")
        print(f"  morts par mur       : {summary['frac_mort_mur']:8.2%}")
        plot_lab_metrics(exp_dir=exp_dir)

    # ---------- NOUVEAU : résumé + plot exploration (low_res) ----------
    def _print_and_plot_low_res(self, summary, exp_dir):
        print(f"\n--- Lab low_res chunk {summary['chunk']} | exploration ---")
        print(f"  found food          : {summary['frac_found_food']:8.2%} des agents")
        if not np.isnan(summary["explore_time_moy"]):
            print(f"  time to 1st resource: {summary['explore_time_moy']:8.2f} "
                  f"± {summary['explore_time_std']:.2f} pas  "
                  f"(médiane {summary['explore_time_med']:.1f})")
        plot_lab_exploration(exp_dir=exp_dir)

    # ---------- NOUVEAU : énergie(t) pour 10 agents d'un lab ----------
    def _plot_energy(self, outputs, exp_dir, lab_dir, env_title, n_envs=10):
        """Sauve exp_dir/energy/<lab_dir>/chunk_<N>_agent_<b>.png pour les
        n_envs premiers génomes (les survivants sont déjà triés). Un sous-graphe
        par agent vivant : 1 pour lab_1/lab_2, 4 pour lab_3 (les clones)."""
        plot_lab_energy(
            energy          = np.asarray(outputs.energy),      # (B, T, N)
            alive           = np.asarray(outputs.alive),       # (B, T, N)
            exp_dir         = exp_dir,
            lab_dir         = lab_dir,
            chunk           = self.chunk_idx + 1,
            min_energy_repr = self.cfg.min_energy_repr,
            time_above_repr = self.cfg.time_above_repr,
            energy_to_die   = self.cfg.energy_to_die,
            n_envs          = n_envs,
            env_title       = env_title,
        )

    def _save_lab_data(self, agg, summary, exp_dir, suffix=""):
        data_dir = os.path.join(exp_dir, "lab_data")
        os.makedirs(data_dir, exist_ok=True)
        tag = f"chunk_{self.chunk_idx+1}" + (f"_{suffix}" if suffix else "")

        np.savez_compressed(os.path.join(data_dir, f"{tag}.npz"), **agg)

        with open(os.path.join(data_dir, f"{tag}_summary.json"), "w") as f:
            json.dump({k: float(v) for k, v in summary.items()}, f, indent=2)

    def _per_agent_metrics(self, outputs):
            """Métriques par agent, alignées par événement de fin de vie
            (mort OU survie jusqu'au dernier pas = censure à droite).
            Toutes les sorties sont des tableaux 1D de longueur D (nb d'événements)."""
            alive  = np.asarray(outputs.alive)         # (T, N)
            born   = np.asarray(outputs.born_step)     # (T, N) step de naissance
            step   = np.asarray(outputs.step)                 # (T,)   step global de la ligne
            pos    = np.asarray(outputs.position)      # (T, N, 2)
            rew    = np.asarray(outputs.rewards)              # (T, N) ou (T, N, 1)
            if rew.ndim == 3 and rew.shape[-1] == 1:          # rewards est (T, N, 1) -> (T, N)
                rew = rew[..., 0]
            time_under_min_energy = np.asarray(outputs.time_under_min_energy)  # (T, N)
            energy = np.asarray(outputs.energy)        # (T, N)

            T = alive.shape[0]

            # 1) Morts : transition t -> t+1
            a_t, a_tp1 = alive[:-1], alive[1:]
            b_t, b_tp1 = born[:-1],  born[1:]
            death_event = (a_t == 1) & ((a_tp1 == 0) | (b_tp1 != b_t))   # (T-1, N)
            d_row, d_slot = np.where(death_event)                        # dernière ligne vivant

            # 2) Survivants : vivants à la dernière ligne (censurés à droite)
            s_slot = np.where(alive[-1] == 1)[0]
            s_row  = np.full(s_slot.shape, T - 1, dtype=int)

            # 3) Fusion morts + survivants + drapeau
            t_row = np.concatenate([d_row,  s_row])
            slot  = np.concatenate([d_slot, s_slot])
            died  = np.concatenate([np.ones(d_row.shape, bool),
                                    np.zeros(s_slot.shape, bool)])       # True = mort

            if t_row.size == 0:
                return None

            # 4) Naissance, durée de vie (marche pour les deux cas)
            b_dead    = born[t_row, slot]
            birth_row = (b_dead - step[0]).astype(int)
            death_row = t_row
            age       = step[t_row] - b_dead        # survivant : step[T-1] - born = durée totale

            # somme inclusive sur [lo, hi] par différence de cumsum (capture slot)
            def window_sum(cum, lo, hi):
                hi_v = cum[hi, slot]
                lo_v = np.where(lo >= 0, cum[np.clip(lo, 0, None), slot], 0.0)
                return hi_v - lo_v

            # 5) Consommation (grandeur "par état") -> fenêtre [birth_row, death_row]
            cum_rew   = np.cumsum(rew, axis=0)                          # (T, N)
            total_rew = window_sum(cum_rew, birth_row - 1, death_row)
            mean_rew  = total_rew / (age + 1)          # age+1 = nb de lignes vécues

            # 6) Mouvement (grandeur "par transition") -> fenêtre [birth_row, death_row-1]
            delta = pos[1:] - pos[:-1]                                  # (T-1, N, 2)
            mag   = np.sqrt((delta ** 2).sum(axis=-1))                 # (T-1, N)
            cum_mag    = np.cumsum(mag, axis=0)                         # (T-1, N)
            total_move = window_sum(cum_mag, birth_row - 1, np.clip(death_row - 1, 0, None))
            total_move = np.where(age > 0, total_move, 0.0)
            mean_speed = np.where(age > 0, total_move / age, 0.0)

            # 7) Cause de mort "par mur" (famine exclue) : n'a de sens que pour les morts
            seuil          = self.cfg.time_to_die
            wall_death_raw = (time_under_min_energy[death_row, slot] < seuil - 1)
            wall_death     = wall_death_raw & died                     # False pour un survivant
            energy_end     = energy[death_row, slot]                   # (D,) aligné sur les événements

            # 8) Exploration : temps jusqu'à la 1re ressource mangée
            #    t_explore = min{ t in [birth_row, death_row] : rew[t, slot] > 0 } - birth_row
            #    NaN si l'agent n'a jamais mangé (censuré : pas trouvé de ressource).
            row_idx   = np.arange(T)[:, None]                          # (T, 1)
            series    = rew[:, slot]                                   # (T, E)
            in_win    = (row_idx >= birth_row[None, :]) & (row_idx <= death_row[None, :])
            ate       = (series > 0) & in_win                         # (T, E)
            ever_ate  = ate.any(axis=0)                               # (E,)
            first_r   = np.where(ever_ate, ate.argmax(axis=0), -1)    # 1re ligne positive
            t_explore = np.where(ever_ate, first_r - birth_row, np.nan)  # (E,)

            return {
                "slot":       slot,        # (slot, born) pour recroiser avec la généalogie
                "born":       b_dead,
                "age":        age,         # durée de vie (survivant = durée totale)
                "total_rew":  total_rew,   # consommation totale
                "mean_rew":   mean_rew,    # consommation moyenne / pas
                "total_move": total_move,  # distance parcourue
                "mean_speed": mean_speed,  # mouvement moyen / pas
                "wall_death": wall_death,  # True = mort par mur ; False si survivant
                "energy_end": energy_end,  # énergie au dernier pas vivant
                "died":       died,        # True = mort, False = survivant (censuré)
                "t_explore":  t_explore,   # délai avant 1re ressource (NaN si jamais)  <== NOUVEAU
                "ever_ate":   ever_ate,    # True si l'agent a mangé au moins une fois  <== NOUVEAU
            }

    def data_lab_env(self, outputs_lab):
        keys = ["age", "total_rew", "mean_rew", "total_move",
                "mean_speed", "energy_end", "wall_death", "died"]
        agg = {k: [] for k in keys}

        B = outputs_lab.alive.shape[0]          # nb d'environnements = nb d'agents testés
        for b in range(B):
            single = jax.tree_util.tree_map(lambda x: x[b], outputs_lab)
            m = self._per_agent_metrics(single)
            if m is None:
                continue
            for k in keys:
                agg[k].append(m[k])

        agg = {k: (np.concatenate(v) if v else np.array([])) for k, v in agg.items()}

        died    = agg["died"].astype(bool)
        n_morts = int(died.sum())
        n_surv  = int((~died).sum())
        n_mur   = int(agg["wall_death"].sum())             # déjà masqué par died
        n_faim  = n_morts - n_mur

        def _stat(arr):
            return (float(arr.mean()), float(arr.std())) if arr.size else (0.0, 0.0)

        vie_m,   vie_s   = _stat(agg["age"])               # tous (survivants comptés à l'âge plein)
        viem_m,  viem_s  = _stat(agg["age"][died])         # conditionnel : morts seulement
        conso_m, conso_s = _stat(agg["mean_rew"])
        mouv_m,  mouv_s  = _stat(agg["mean_speed"])

        summary = {
            "chunk":              self.chunk_idx + 1,
            "n_agents":           B,
            "n_morts":            n_morts,
            "n_survivants":       n_surv,
            "duree_vie_moy":      vie_m,   "duree_vie_std":      vie_s,    # inclut les survivants
            "duree_vie_mort_moy": viem_m,  "duree_vie_mort_std": viem_s,   # morts uniquement
            "consommation_moy":   conso_m, "consommation_std":   conso_s,
            "mouvement_moy":      mouv_m,  "mouvement_std":      mouv_s,
            "frac_mort_mur":      n_mur  / B,
            "frac_mort_faim":     n_faim / B,
            "frac_survie":        n_surv / B,              # mur + faim + survie = 1
        }
        return agg, summary

    # =================================================================
    #  NOUVEAU — A) EXPLORATION (env low_res)
    # =================================================================
    def data_lab_env_low_res(self, outputs_lab):
        """Résumé de l'env 'low_res' (ressources fixes, non renouvelées).
        Métrique d'exploration = temps avant la 1re ressource mangée.
        On sépare deux grandeurs qu'il ne faut PAS mélanger :
          - frac_found_food : P(l'agent trouve au moins une ressource)
          - explore_time    : conditionnel, calculé UNIQUEMENT sur ceux qui ont
                              mangé (sinon les NaN des censurés fausseraient tout).
        """
        B = outputs_lab.alive.shape[0]
        t_explore, ever_ate = [], []
        for b in range(B):
            single = jax.tree_util.tree_map(lambda x: x[b], outputs_lab)
            m = self._per_agent_metrics(single)
            if m is None:
                continue
            t_explore.append(m["t_explore"])
            ever_ate.append(m["ever_ate"])

        t_explore = np.concatenate(t_explore) if t_explore else np.array([])
        ever_ate  = np.concatenate(ever_ate).astype(bool) if ever_ate else np.array([], bool)

        found = t_explore[ever_ate]                       # temps de ceux qui ont mangé
        summary = {
            "chunk":            self.chunk_idx + 1,
            "n_agents":         int(ever_ate.size),
            "frac_found_food":  float(ever_ate.mean())     if ever_ate.size else 0.0,
            "explore_time_moy": float(found.mean())        if found.size    else float("nan"),
            "explore_time_std": float(found.std())         if found.size    else float("nan"),
            "explore_time_med": float(np.median(found))    if found.size    else float("nan"),
        }
        agg = {"t_explore": t_explore, "ever_ate": ever_ate}
        return agg, summary

    # =================================================================
    #  NOUVEAU — B) EFFET DES PAIRS (env clones)
    # =================================================================
    def data_lab_env_grouped(self, outputs_lab):
        """Réduit le comportement À L'INTÉRIEUR de chaque environnement.
        Dans clones, les agents vivants partagent le même génome ; on les
        moyenne -> un seul profil comportemental par génome. Pour high_res
        (1 agent/env) la moyenne est triviale. Utiliser la MÊME fonction pour
        les deux garantit des définitions de métriques identiques.

        Retourne per_genome : dict de tableaux (B,), alignés par index de
        génome (même ordre que agent_params / key_sim).
        """
        keys = ["age", "mean_rew", "mean_speed", "energy_end", "wall_death", "died"]
        B = outputs_lab.alive.shape[0]
        per_genome = {k: np.full(B, np.nan) for k in keys}
        n_peers    = np.zeros(B, dtype=int)

        for b in range(B):
            single = jax.tree_util.tree_map(lambda x: x[b], outputs_lab)
            m = self._per_agent_metrics(single)
            if m is None:
                continue
            n_peers[b] = m["age"].size                     # nb de clones vivants
            for k in keys:
                per_genome[k][b] = float(np.mean(m[k]))    # bool -> fraction

        per_genome["n_peers"] = n_peers
        return per_genome

    def compare_alone_vs_clones(self, outputs_alone, outputs_clones, exp_dir):
        """Compare, PAR GÉNOME, le comportement SEUL (high_res) vs EN GROUPE
        (clones). Les deux rollouts partagent agent_params/key_env/key_sim dans
        le même ordre -> tableaux alignés par génome -> comparaison APPARIÉE :

            delta[g] = comportement_clones[g] - comportement_seul[g]

        delta.mean() isole l'effet moyen des pairs à génome fixé (élimine la
        variance inter-génomes)."""
        a = self.data_lab_env_grouped(outputs_alone)
        c = self.data_lab_env_grouped(outputs_clones)

        metrics = ["age", "mean_rew", "mean_speed", "energy_end", "wall_death"]
        labels  = {"age": "lifespan (steps)", "mean_rew": "consumption /step",
                   "mean_speed": "movement /step", "energy_end": "final energy",
                   "wall_death": "fraction wall deaths"}

        table = {}
        for k in metrics:
            mask  = ~(np.isnan(a[k]) | np.isnan(c[k]))     # génomes valides des 2 côtés
            delta = c[k][mask] - a[k][mask]
            table[k] = {
                "alone_moy":  float(np.nanmean(a[k])) if np.isfinite(a[k]).any() else float("nan"),
                "clones_moy": float(np.nanmean(c[k])) if np.isfinite(c[k]).any() else float("nan"),
                "delta_moy":  float(delta.mean()) if delta.size else float("nan"),
                "delta_std":  float(delta.std())  if delta.size else float("nan"),
                "n":          int(mask.sum()),
            }

        # affichage
        print(f"\n--- Lab chunk {self.chunk_idx+1} | ALONE vs CLONES (peers effect) ---")
        print(f"  {'metric':<22}{'alone':>10}{'clones':>10}{'Δ (peers)':>14}")
        for k in metrics:
            r = table[k]
            print(f"  {labels[k]:<22}{r['alone_moy']:>10.3f}{r['clones_moy']:>10.3f}"
                  f"{r['delta_moy']:>10.3f}±{r['delta_std']:.2f}")

        # sauvegarde json (une entrée par chunk -> suivi de l'évolution)
        data_dir = os.path.join(exp_dir, "lab_data")
        os.makedirs(data_dir, exist_ok=True)
        payload = {"chunk": self.chunk_idx + 1, "metrics": table}
        with open(os.path.join(data_dir, f"chunk_{self.chunk_idx+1}_alone_vs_clones.json"), "w") as f:
            json.dump(payload, f, indent=2)

        plot_alone_vs_clones(exp_dir=exp_dir)
        return table