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

from simulation.lab_env import vmap_over_agents_env_lab_high_res,vmap_over_agents_env_lab_low_res,vmap_over_agents_env_lab_high_res_with_clones, rotate_resources, vmap_over_agents_env_lab_adapt,ROTATIONS
from simulation.utils.plots import (plot_lab_metrics, plot_lab_exploration,
                            plot_alone_vs_clones, plot_lab_energy,plot_energy_response,
                            plot_eaten_by_type_boxplot, plot_prob_eat_vs_eaten,
                            plot_prob_eat_over_life)
from simulation.utils.utils_sim import _video_worker, outputs_to_numpy, load_shuffle_log
from simulation.simulation_data.energy_response import (default_energy_bins,
                                        energy_response_over_envs,
                                        resource_in_view, _wilson,
                                        ENERGY_EAT_WINDOW)
from simulation.data_class import LABELS
 
GREED_WINDOW = 10     # W : fenetres non chevauchantes pour la greediness
REWARD_LAG   = 1      # log.rewards[t] = recompense gagnee au pas t-1
#   Une fois `obs=state.obs` applique dans StepLog :
#     log.obs[t]     = observation sur laquelle l'agent DECIDE au pas t
#     log.rewards[t] = recompense gagnee au pas t-1
#   donc la recompense qui SUIT l'observation du pas t est log.rewards[t+1].

# definition unique dans energy_response (evite un cycle d'import)
_resource_in_view = resource_in_view
 
 
def _greediness(saw, ate, slot, birth_row, death_row, window=GREED_WINDOW):
    """G = Cr / Tr sur des fenetres non chevauchantes de `window` pas.
      Tr = fenetres vecues ou l'agent voit >= 1 ressource
      Cr = fenetres PARMI CELLES-LA ou il en consomme >= 1
    Cr etant compte parmi Tr, G est dans [0, 1] par construction.
    Fenetres alignees sur le rollout (t // window) : en lab tous les agents
    naissent au pas 0, donc elles sont comparables entre agents.
    G = NaN si Tr = 0 (jamais rien vu -> la question ne se pose pas)."""
    T, E  = saw.shape[0], slot.size
    n_win = int(np.ceil(T / window))
    G  = np.full(E, np.nan)
    Tr = np.zeros(E, dtype=int)
    Cr = np.zeros(E, dtype=int)
    for e in range(E):
        s, lo, hi = slot[e], int(birth_row[e]), int(death_row[e])
        if hi < lo:
            continue
        rows = np.arange(lo, hi + 1)
        w    = rows // window
        seen_w = np.zeros(n_win, dtype=bool)
        ate_w  = np.zeros(n_win, dtype=bool)
        seen_w[w[saw[rows, s]]] = True
        ate_w[w[ate[rows, s]]]  = True
        tr = int(seen_w.sum())
        cr = int((seen_w & ate_w).sum())
        Tr[e], Cr[e] = tr, cr
        if tr > 0:
            G[e] = cr / tr
    return G, Tr, Cr
 
 
def _clean(a):
    """Retire les NaN avant _dispersion (agents pour qui la mesure est indefinie)."""
    a = np.asarray(a, dtype=float)
    return a[~np.isnan(a)]

# =====================================================================
#  À PLACER AU NIVEAU MODULE — avant "class LabMixin:"
#  (fonction libre, pas une méthode : elle ne prend pas self)
# =====================================================================
def _dispersion(arr, prefix, empty=0.0):
    """Statistiques de position ET de dispersion pour une mesure.
    On stocke a la fois std, min/max et quartiles : les quantiles sont des
    valeurs OBSERVEES, donc toujours dans le support de la mesure (jamais de
    duree de vie negative), contrairement a moy ± std."""
    keys = ("moy", "std", "min", "max", "p25", "p50", "p75")
    if arr.size == 0:
        return {f"{prefix}_{k}": empty for k in keys}
    return {
        f"{prefix}_moy": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
    }


def rotation_name(resources, rot):
    """Nom de la CONDITION testee : "<ce qui est remplace>_to_poison".

    Le canal qui portait X se retrouve a porter le poison -> "X_to_poison".
    C'est ce que l'agent subit : la case qu'il avait appris a manger devient
    toxique.

    C'est ce nom, et pas l'indice de rotation, qui sert de cle partout (donnees,
    figures, videos) : `rot` n'est qu'un decalage de canaux, dont le SENS depend
    de la config courante que shuffle_resources repermute en cours de run. Le
    meme rot designe donc des conditions differentes au fil du temps, alors que
    le nom designe toujours la meme -> a recalculer a chaque appel, jamais en dur.
    """
    if rot == 0:
        return "baseline"
    rotated = rotate_resources(resources, rot)
    poison_ch = next(k for k, r in enumerate(rotated) if LABELS[r.id] == "poison")
    return f"{LABELS[resources[poison_ch].id]}_to_poison"


def config_caption(resources, rotated, steps_in_place=None):
    """Deux lignes alignees : config apprise vs config testee, + anciennete."""
    fmt = lambda rs: " | ".join(f"ch{k} {LABELS[r.id]:<6}" for k, r in enumerate(rs))
    since = ("" if steps_in_place is None
             else f"   (in place for {steps_in_place:,} steps)".replace(",", " "))
    return f"learned: {fmt(resources)}{since}\ntested : {fmt(rotated)}"


def steps_since_last_shuffle(shuffle_log, step_now):
    """Depuis combien de steps la config courante est-elle en place ?

    Pas de shuffle encore -> la config initiale tient depuis le step 0."""
    past = [e["step"] for e in shuffle_log if e["step"] <= step_now]
    return int(step_now) - (max(past) if past else 0)

class LabMixin:

    def launch_env(self, state, key_env, subkey_sim, model, exp_dir, n, submit_video):
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

            plot_lab_metrics(exp_dir=exp_dir)
            self._plot_energy(outputs_high, exp_dir, "high_res",
                            "lab_1 — high_res (agent alone)")

            for b in range(2):
                vid = os.path.join(exp_dir, "videos", "high",
                                f"high_res_video_chunk_{self.chunk_idx}_lab_{b}.mp4")
                submit_video(outputs_to_numpy(agent_slice(outputs_high, b)), vid, 20, 10,
                            self.cfg.resources,
                            label=f"high_res_chunk_{self.chunk_idx}_lab_{b}")

            # ============ 2) LOW_RES (exploration) ============
            final_state, outputs_low = vmap_over_agents_env_lab_low_res(
                agent_params, key_env, key_sim, model, self.cfg)
            agg_low, summary_low = self.data_lab_env_low_res(outputs_low)
            self._save_lab_data(agg_low, summary_low, exp_dir, suffix="lowres")

            plot_lab_exploration(exp_dir=exp_dir)
            self._plot_energy(outputs_low, exp_dir, "low_res",
                            "lab_2 — low_res (exploration)")

            for b in range(3):
                vid = os.path.join(exp_dir, "videos", "low",
                                f"low_res_video_chunk_{self.chunk_idx}_lab_{b}.mp4")
                submit_video(outputs_to_numpy(agent_slice(outputs_low, b)), vid, 20, 10,
                            self.cfg.resources,
                            label=f"low_res_chunk_{self.chunk_idx}_lab_{b}")

            # ============ 3) CLONES (effet des pairs) ============
            final_state, outputs_clones = vmap_over_agents_env_lab_high_res_with_clones(
                agent_params, key_env, key_sim, model, self.cfg)
            # comparaison APPARIÉE avec l'env high_res (mêmes génomes, même ordre) :
            self.compare_alone_vs_clones(outputs_high, outputs_clones, exp_dir)
            self._plot_energy(outputs_clones, exp_dir, "high_res_clones",
                            "lab_3 — high_res with clones")

            # self.plot_energy_response_labs(outputs_high, outputs_low, outputs_clones, exp_dir)
            for b in range(3):
                vid = os.path.join(exp_dir, "videos", "high_res_clones",
                                f"high_res_clones_video_chunk_{self.chunk_idx}_lab_{b}.mp4")
                submit_video(outputs_to_numpy(agent_slice(outputs_clones, b)), vid, 20, 10,
                            self.cfg.resources,
                            label=f"clones_chunk_{self.chunk_idx}_lab_{b}")

            # ============ 4) ADAPTATION (rotations des canaux) ============
            final_state, outputs_adapt = vmap_over_agents_env_lab_adapt(
                agent_params, key_env, key_sim, model, self.cfg)
            # outputs_adapt : axe 0 = agent (B), axe 1 = rotation (2)

            # Controle apparie : lab_1 partage agent_params / key_env / key_sim et
            # le meme in_axes que l'env adapt -> l'index b designe le MEME genome
            # dans les deux, seule la permutation des canaux differe.
            eaten_baseline = self.eaten_by_type(outputs_high)
            baseline_ids   = [r.id for r in self.cfg.resources]
            # Plafond lu sur la grille de CHAQUE env : la permutation change le canal
            # de chaque ressource, donc les tirages de croissance a l'init different
            # legerement d'une rotation a l'autre -> un plafond par cote.
            n_types = len(self.cfg.resources)
            av_base = self.available_by_type(outputs_high, n_types)
            baseline_available = {r.id: av_base[k] for k, r in enumerate(self.cfg.resources)}

            # Depuis quand l'agent vit-il avec la config qu'on s'apprete a casser ?
            # C'est ce qui dit s'il a eu le temps de l'apprendre.
            steps_in_place = steps_since_last_shuffle(load_shuffle_log(exp_dir),
                                                      int(state.step))

            # Reference : la meme propension dans l'env NON permute, memes genomes.
            # Poolee sur tout le rollout -> une seule valeur, tracee en horizontale.
            _, n_base_poison, k_base_poison = self.prob_eat_vs_eaten(
                outputs_high, self.cfg.resources)
            _, life_bn, life_bk = self.prob_eat_over_life(outputs_high, self.cfg.resources)
            life_baseline = (life_bn, life_bk)

            for j, rot in enumerate(ROTATIONS):          # j = position sur l'axe, rot = vraie rotation
                out_rot = jax.tree_util.tree_map(lambda x: x[:, j], outputs_adapt)   # slice par j

                # On indexe TOUT par la condition experimentale, pas par l'indice de
                # rotation. `rot` n'est qu'un decalage de canaux : selon la config
                # courante, rot1 est tantot good_to_poison tantot medium_to_poison.
                # Grouper par rot melangerait donc deux conditions differentes dans
                # la meme serie ; grouper par nom rassemble les memes.
                name = rotation_name(self.cfg.resources, rot)
                resources_rot = rotate_resources(self.cfg.resources, rot)
                caption = config_caption(self.cfg.resources, resources_rot, steps_in_place)
                title = f"lab_4 — adapt {name}  (config MODIFIÉE)\n{caption}"

                agg_r, summary_r = self.data_lab_env(outputs_lab=out_rot)
                self._save_lab_data(agg_r, summary_r, exp_dir, suffix=f"adapt_{name}")
                plot_lab_metrics(exp_dir=exp_dir, suffix=f"adapt_{name}")
                self._plot_energy(out_rot, exp_dir, f"adapt/{name}", title)

                av_rot = self.available_by_type(out_rot, n_types)

                # Combien de chaque type l'agent a-t-il mange sous cette permutation,
                # compare a lui-meme dans l'env non permute ?
                plot_eaten_by_type_boxplot(
                    eaten          = self.eaten_by_type(out_rot),
                    ids_by_channel = [r.id for r in resources_rot],
                    exp_dir        = exp_dir,
                    chunk          = self.chunk_idx,
                    tag            = name,
                    mapping        = caption,
                    available      = {r.id: av_rot[k] for k, r in enumerate(resources_rot)},
                    baseline       = eaten_baseline,
                    baseline_ids   = baseline_ids,
                    baseline_available = baseline_available,
                )

                # L'agent apprend-il a eviter le poison au fil de ses erreurs ?
                x_p, n_p, k_p = self.prob_eat_vs_eaten(out_rot, resources_rot)
                if x_p.size:
                    plot_prob_eat_vs_eaten(
                        x_p, n_p, k_p, _wilson, exp_dir,
                        chunk=self.chunk_idx, tag=name, mapping=caption,
                        baseline=(n_base_poison, k_base_poison),
                    )

                # Meme question, mais chaque agent compare a LUI-MEME : c'est
                # celle-ci qui est interpretable (cf. prob_eat_over_life).
                life, life_n, life_k = self.prob_eat_over_life(out_rot, resources_rot)
                if life.size:
                    plot_prob_eat_over_life(
                        life, life_n, life_k, _wilson, exp_dir,
                        chunk=self.chunk_idx, tag=name,
                        mapping=caption, baseline=life_baseline,
                    )

                for b in range(2):
                    vid = os.path.join(exp_dir, "videos", "adapt", name,
                                    f"adapt_{name}_chunk_{self.chunk_idx}_lab_{b}.mp4")
                    submit_video(outputs_to_numpy(agent_slice(out_rot, b)), vid, 20, 10,
                                resources_rot,
                                label=f"adapt_{name}_chunk_{self.chunk_idx}_lab_{b}")

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

    @staticmethod
    def available_by_type(outputs_lab, n_types):
        """(n_types,) : ressources PRESENTES sur la grille au debut du rollout, par CANAL.

        A ne PAS confondre avec init_number_of_resources : init_state_lab lance
        pre_growth_step iterations de resources_growth quoi qu'il arrive (le flag
        cfg.resources_growth ne coupe que la repousse PENDANT l'episode), donc la
        grille de depart contient bien plus que les graines initiales. On lit donc
        le vrai contenu de la grille, ce qui suit automatiquement tout changement
        de pre_growth_step. L'episode ne faisant pas repousser, ce total est aussi
        le disponible sur toute la duree."""
        grid0 = np.asarray(outputs_lab.grid[:, 0, :n_types])   # (B, n_types, L, L)
        return grid0.sum(axis=(2, 3)).mean(axis=0)             # key_env partage -> grilles identiques

    @staticmethod
    def _poison_events(outputs_lab, resources_cfg, label="poison",
                       window=ENERGY_EAT_WINDOW):
        """Par agent : la suite ORDONNEE de ses rencontres avec `label`.

        Rend une liste de B tuples (t, cum, y) ou, pour chaque pas t ou le type
        etait dans le champ de vision et l'agent vivant :
          t[i]   = le pas lui-meme
          cum[i] = combien il en avait deja mange AVANT ce pas
          y[i]   = 1 s'il en consomme dans [t, t+window]

        `resources_cfg` doit etre la config REELLEMENT jouee (permutee pour un
        env adapt) : c'est elle qui dit quel canal porte le poison."""
        ch = [k for k, r in enumerate(resources_cfg) if LABELS[r.id] == label]
        if not ch:
            return []
        ch = ch[0]
        n_channels = len(resources_cfg) + 2          # ressources + agents + murs

        consumed = np.asarray(outputs_lab.consumed_res)[..., ch]   # (B, T)
        alive    = np.asarray(outputs_lab.alive)                   # (B, T, N)
        B, T = consumed.shape

        # obs est (B, T, N, side, side, C) ; resource_in_view attend un seul axe
        # temps devant, on replie donc B et T. reshape sur un tableau contigu est
        # une vue -> pas de copie du (B, T, N, 11, 11, C).
        obs = np.asarray(outputs_lab.obs)
        saw_all = resource_in_view(obs.reshape((-1,) + obs.shape[2:]),
                                   np.array([ch]), n_channels)     # (B*T, N)
        saw_all = saw_all.reshape(B, T, -1)                        # (B, T, N)

        events = []
        for b in range(B):
            ate = consumed[b] > 0                                  # (T,)
            ate_w = ate.copy()                                     # dans [t, t+W]
            for d in range(1, window + 1):
                ate_w[:-d] |= ate[d:]
            # cumul AVANT le pas t -> decalage de 1
            cum = np.concatenate([[0], np.cumsum(consumed[b])[:-1]]).astype(int)
            live = alive[b].any(axis=1)                            # (T,) un agent vivant
            saw  = saw_all[b].any(axis=1) & live                   # (T,)
            idx  = np.where(saw)[0]                                # deja trie
            events.append((idx, cum[idx], ate_w[idx].astype(int)))
        return events, T

    @staticmethod
    def prob_eat_over_life(outputs_lab, resources_cfg, label="poison",
                           n_bins=4, window=ENERGY_EAT_WINDOW, min_events=8):
        """P(manger | en vue) le long de la VIE PROPRE de chaque agent.

        Rend (courbes, n, k) : courbes (B', n_bins) par agent, et n/k (n_bins,)
        les comptes pooles sur tous les agents.

        L'intervalle [naissance, mort] de CHAQUE agent est decoupe en `n_bins`
        tranches de duree egale. Deux raisons de normaliser par la vie plutot
        que d'utiliser le temps absolu du rollout :

          - les agents ne vivent pas tous aussi longtemps (mort de faim). Sur un
            axe en temps absolu, un agent mort tot ne peuple que les premieres
            tranches, donc les tranches tardives ne contiennent que des
            survivants -- et si ce sont les mangeurs de poison qui meurent, on
            lirait une fausse decroissance. Ici chaque agent couvre TOUT l'axe.
          - "poison deja mange" serait pire encore : c'est un cumul de la
            variable mesuree, donc la courbe monte meme SANS apprentissage.

        L'axe est le temps et non le nombre de poisons manges, car un agent
        apprend aussi en mangeant les BONNES ressources : son experience ne se
        resume pas a ses erreurs.

        Les courbes par agent sont bruitees (peu de rencontres par tranche, donc
        des valeurs multiples de 1/n) : c'est n et k, pooles, qui donnent la
        courbe agregee stable. NaN quand un agent n'a rien rencontre dans une
        tranche. Les agents avec moins de `min_events` rencontres sont ecartes."""
        events, _T = LabMixin._poison_events(outputs_lab, resources_cfg, label, window)
        alive = np.asarray(outputs_lab.alive)                  # (B, T, N)

        out = []
        n_tot = np.zeros(n_bins, dtype=int)
        k_tot = np.zeros(n_bins, dtype=int)
        for b, (t, _cum, y) in enumerate(events):
            if len(y) < min_events:
                continue
            live = np.where(alive[b].any(axis=1))[0]           # pas ou il est vivant
            if live.size == 0:
                continue
            lo, hi = int(live[0]), int(live[-1]) + 1           # sa vie a lui
            duree = max(hi - lo, n_bins)
            ligne = []
            for q in range(n_bins):
                a = lo + q * duree // n_bins
                z = lo + (q + 1) * duree // n_bins
                m = (t >= a) & (t < z)
                n_tot[q] += int(m.sum())
                k_tot[q] += int(y[m].sum())
                ligne.append(float(np.mean(y[m])) if m.any() else np.nan)
            out.append(ligne)
        return (np.array(out) if out else np.zeros((0, n_bins))), n_tot, k_tot

    @staticmethod
    def prob_eat_vs_eaten(outputs_lab, resources_cfg, label="poison",
                          window=ENERGY_EAT_WINDOW):
        """(x, n, k) : P(manger `label` | `label` en vue) selon le nombre deja mange.

        ATTENTION : l'axe x est un cumul de la variable mesuree, donc la courbe
        monte a droite meme sans apprentissage (seuls les gros mangeurs
        atteignent les x eleves). Pour juger l'adaptation, utiliser plutot
        prob_eat_over_life, qui compare chaque agent a lui-meme."""
        counts = {}
        events, _T = LabMixin._poison_events(outputs_lab, resources_cfg, label, window)
        for _t, cum, y in events:
            for c, yi in zip(cum, y):
                n_, k_ = counts.get(int(c), (0, 0))
                counts[int(c)] = (n_ + 1, k_ + int(yi))

        if not counts:
            return np.array([]), np.array([]), np.array([])
        x = np.array(sorted(counts))
        n = np.array([counts[i][0] for i in x], dtype=float)
        k = np.array([counts[i][1] for i in x], dtype=float)
        return x, n, k

    @staticmethod
    def eaten_by_type(outputs_lab):
        """(B, n_types) : total mange par chaque agent, par CANAL.

        Les env de lab high_res n'ont qu'UN agent vivant (n_agents_max=2, l'index
        0 est toujours mort), donc consumed_res — global a l'env — est exactement
        la consommation de cet agent. Un agent mort ne consomme plus
        (survives_int=0), la somme sur tout le rollout couvre donc sa vie entiere
        sans avoir a fenetrer sur [birth, death]."""
        return np.asarray(outputs_lab.consumed_res).sum(axis=1)   # (B, T, n_types) -> (B, n_types)

    def _save_lab_data(self, agg, summary, exp_dir, suffix=""):
        data_dir = os.path.join(exp_dir, "lab_data")
        os.makedirs(data_dir, exist_ok=True)
        tag = f"chunk_{self.chunk_idx}" + (f"_{suffix}" if suffix else "")

        np.savez_compressed(os.path.join(data_dir, f"{tag}.npz"), **agg)

        with open(os.path.join(data_dir, f"{tag}_summary.json"), "w") as f:
            json.dump({k: float(v) for k, v in summary.items()}, f, indent=2)


   # =================================================================
    #  INCHANGÉ
    # =================================================================
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
            rew_pos   = np.maximum(rew, 0.0)                 # <-- on jette les gains négatifs
            cum_rew   = np.cumsum(rew_pos, axis=0)
            total_rew = window_sum(cum_rew, birth_row - 1, death_row)
            mean_rew  = total_rew / (age + 1)
 
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
 
            # 9) Greediness : G = Cr / Tr sur des fenetres de GREED_WINDOW pas
            delta_e       = np.array([r.delta_energy for r in self.cfg.resources])
            n_channels    = len(self.cfg.resources) + 2       # ressources + agents + murs
            good_channels = np.where(delta_e > 0)[0]
            saw = _resource_in_view(outputs.obs, good_channels, n_channels)
            ate_step = np.zeros_like(saw, dtype=bool)          # consommation alignee
            if REWARD_LAG > 0:
                ate_step[:-REWARD_LAG] = rew[REWARD_LAG:] > 0
            else:
                ate_step = rew > 0
            greediness, greed_Tr, greed_Cr = _greediness(
                saw, ate_step, slot, birth_row, death_row)
            
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
                "t_explore":  t_explore,   # délai avant 1re ressource (NaN si jamais)
                "ever_ate":   ever_ate,    # True si l'agent a mangé au moins une fois
                "greediness": greediness,  # G = Cr / Tr (NaN si Tr = 0)      <== NOUVEAU
                "greed_Tr":   greed_Tr,    # fenêtres avec ressource visible  <== NOUVEAU
                "greed_Cr":   greed_Cr,    # fenêtres avec consommation       <== NOUVEAU
            }
 
    # =================================================================
    #  MODIFIÉ — _stat supprimé, remplacé par **_dispersion
    # =================================================================
    def data_lab_env(self, outputs_lab):
        keys = ["age", "total_rew", "mean_rew", "total_move",
                "mean_speed", "energy_end", "wall_death", "died",
                "greediness"]                                  # <== NOUVEAU
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
 
        summary = {
            "chunk":              self.chunk_idx + 1,
            "n_agents":           B,
            "n_morts":            n_morts,
            "n_survivants":       n_surv,
            "frac_mort_mur":      n_mur  / B,
            "frac_mort_faim":     n_faim / B,
            "frac_survie":        n_surv / B,              # mur + faim + survie = 1
            # chaque appel étale 7 clés : _moy _std _min _max _p25 _p50 _p75
            **_dispersion(agg["age"],        "duree_vie"),       # inclut les survivants
            **_dispersion(agg["age"][died],  "duree_vie_mort"),  # morts uniquement
            **_dispersion(agg["mean_rew"],   "consommation"),
            **_dispersion(agg["mean_speed"], "mouvement"),
            # G indéfini (NaN) pour les agents n'ayant jamais vu de ressource
            **_dispersion(_clean(agg["greediness"]), "greediness", empty=float("nan")),
        }
        return agg, summary
 
    # =================================================================
    #  MODIFIÉ — A) EXPLORATION (env low_res)
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
        t_explore, ever_ate, greed = [], [], []
        for b in range(B):
            single = jax.tree_util.tree_map(lambda x: x[b], outputs_lab)
            m = self._per_agent_metrics(single)
            if m is None:
                continue
            t_explore.append(m["t_explore"])
            ever_ate.append(m["ever_ate"])
            greed.append(m["greediness"])
 
        t_explore = np.concatenate(t_explore) if t_explore else np.array([])
        ever_ate  = np.concatenate(ever_ate).astype(bool) if ever_ate else np.array([], bool)
        greed     = np.concatenate(greed) if greed else np.array([])
 
        found = t_explore[ever_ate]                       # temps de ceux qui ont mangé
        summary = {
            "chunk":            self.chunk_idx + 1,
            "n_agents":         int(ever_ate.size),        # tous les agents testés
            "n_found_food":     int(found.size),           # ceux retenus dans explore_time
            "frac_found_food":  float(ever_ate.mean()) if ever_ate.size else 0.0,
            **_dispersion(found, "explore_time", empty=float("nan")),
            **_dispersion(_clean(greed), "greediness", empty=float("nan")),  # <== NOUVEAU
        }
        agg = {"t_explore": t_explore, "ever_ate": ever_ate, "greediness": greed}
        return agg, summary
 
    # =================================================================
    #  INCHANGÉ
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
        keys = ["age", "mean_rew", "mean_speed", "energy_end", "wall_death",
                "died", "greediness"]                          # <== NOUVEAU
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
                v = np.asarray(m[k], dtype=float)
                # nanmean : greediness est indéfinie pour un clone n'ayant rien vu
                per_genome[k][b] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
 
        per_genome["n_peers"] = n_peers
        return per_genome
 
    # =================================================================
    #  MODIFIÉ — B) EFFET DES PAIRS (env clones)
    # =================================================================
    def compare_alone_vs_clones(self, outputs_alone, outputs_clones, exp_dir):
        """Compare, PAR GÉNOME, le comportement SEUL (high_res) vs EN GROUPE
        (clones). Les deux rollouts partagent agent_params/key_env/key_sim dans
        le même ordre -> tableaux alignés par génome -> comparaison APPARIÉE :
 
            delta[g] = comportement_clones[g] - comportement_seul[g]
 
        La médiane de delta isole l'effet des pairs à génome fixé (élimine la
        variance inter-génomes)."""
        a = self.data_lab_env_grouped(outputs_alone)
        c = self.data_lab_env_grouped(outputs_clones)
 
        metrics = ["age", "mean_rew", "mean_speed", "energy_end", "wall_death",
                   "greediness"]                               # <== NOUVEAU
        labels  = {"age": "lifespan (steps)", "mean_rew": "consumption /step",
                   "mean_speed": "movement /step", "energy_end": "final energy",
                   "wall_death": "fraction wall deaths",
                   "greediness": "greediness G = Cr/Tr"}
 
        table = {}
        for k in metrics:
            mask  = ~(np.isnan(a[k]) | np.isnan(c[k]))     # génomes valides des 2 côtés
            delta = c[k][mask] - a[k][mask]
            row = {"n": int(mask.sum())}
            row.update(_dispersion(a[k][mask], "alone",  empty=float("nan")))
            row.update(_dispersion(c[k][mask], "clones", empty=float("nan")))
            row.update(_dispersion(delta,      "delta",  empty=float("nan")))
            table[k] = row
 
        # affichage : médianes + IQR de l'effet apparié
        print(f"\n--- Lab chunk {self.chunk_idx} | ALONE vs CLONES (peers effect) ---")
        print(f"  {'metric':<22}{'alone':>10}{'clones':>10}{'Δ median':>11}{'Δ IQR':>20}")
        for k in metrics:
            r = table[k]
            print(f"  {labels[k]:<22}{r['alone_p50']:>10.3f}{r['clones_p50']:>10.3f}"
                  f"{r['delta_p50']:>11.3f}"
                  f"{'[' + format(r['delta_p25'], '.3f') + ', ' + format(r['delta_p75'], '.3f') + ']':>20}")
 
        # sauvegarde json (une entrée par chunk -> suivi de l'évolution)
        data_dir = os.path.join(exp_dir, "lab_data")
        os.makedirs(data_dir, exist_ok=True)
        payload = {"chunk": self.chunk_idx + 1, "metrics": table}
        with open(os.path.join(data_dir, f"chunk_{self.chunk_idx}_alone_vs_clones.json"), "w") as f:
            json.dump(payload, f, indent=2)
 
        plot_alone_vs_clones(exp_dir=exp_dir)
        return table
    
    def plot_energy_response_labs(self, out_high, out_low, out_clones, exp_dir):
        curves = {
            "lab_1 high_res": energy_response_over_envs(out_high,   self.cfg),
            "lab_2 low_res":  energy_response_over_envs(out_low,    self.cfg),
            "lab_3 clones":   energy_response_over_envs(out_clones, self.cfg),
        }
        plot_energy_response(exp_dir, self.chunk_idx + 1, curves, cfg=self.cfg)