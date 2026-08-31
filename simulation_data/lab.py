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
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap

from simulation.lab_env import launch_env_high_res, vmap_over_agents_env_lab_high_res,vmap_over_agents_env_lab_low_res,vmap_over_agents_env_lab_high_res_with_clones, rotate_resources, vmap_over_agents_env_lab_adapt, rotations_for, vmap_mutate
from simulation.utils.plots import (plot_memory_gain_hist, plot_metric_pairs, plot_food_simplex, plot_replay_top_gain, plot_evolvability, EVO_METRIQUES, plot_lab_metrics, plot_lab_exploration,
                            plot_alone_vs_clones, plot_lab_energy,plot_energy_response,
                            plot_eaten_by_type_boxplot, plot_prob_eat_over_life_by_type)
# plot_prob_eat_ratio : desactive, voir les appels commentes plus bas
from simulation.utils.utils_sim import _video_worker, outputs_to_numpy, load_shuffle_log
from simulation.simulation_data.energy_response import (default_energy_bins,
                                        energy_response_over_envs,
                                        resource_in_view, _wilson,
                                        ENERGY_EAT_WINDOW)
from simulation.data_class import LABELS
 
N_FILM       = 3      # genomes rejoues avec la grille, pour les videos
def _vmap_rot(params, key_env, cles, cfg, model, rot):
    """vmap sur (params, key_sim) ; le reste est ferme dans la closure.

    `rot` NE DOIT PAS etre un argument de vmap : il indexe un tuple Python dans
    rotate_resources, donc il doit rester statique. Meme patron que
    launch_adaptation_env, ou rot est une variable de boucle.
    """
    return vmap(lambda p, k: launch_env_high_res(p, key_env, k, cfg, model, rot))(
        params, cles)

EVO_BATCH    = 25     # enfants evalues par vmap : borne la memoire GPU
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
    # On nomme la condition par la ressource la plus NEFASTE, pas par "le
    # poison" : une config qui n'en contient pas plantait ici.
    pire = min(range(len(rotated)), key=lambda k: rotated[k].delta_energy)
    return f"{LABELS[resources[pire].id]}_to_{LABELS[rotated[pire].id]}"


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

    def launch_evolvability(self, state, key_env, subkey_sim, model, exp_dir):
        """Evaluabilite : les N meilleurs genomes, chacun re-echantillonne en
        M enfants mutes, plus le parent lui-meme, evalues dans l'env high_res.

        "Meilleur" = le plus vieux. compute_survivors ne trie pas, et sous
        critere de viabilite survivre longtemps EST le critere.
        """
        survivants = self.compute_survivors(state)
        if not survivants:
            print("[evolvability] aucun survivant, saute")
            return
        classes = sorted(survivants, key=lambda t: t[1])[:self.cfg.evolvability_agents]
        n_enf   = self.cfg.evolvability_children
        cfg_m   = self.cfg._replace(log_grid=False)
        t0 = time.time()

        enfants_m = {k: [] for k in EVO_METRIQUES}
        parent_m  = {k: [] for k in EVO_METRIQUES}
        etiquettes = []
        n_types = len(self.cfg.resources)
        ids_base = [r.id for r in self.cfg.resources]

        # disponible sur la grille : un seul rollout journalise, la grille de
        # depart ne depend que de key_env et de cfg, donc elle est la meme partout
        dispo = None
        if n_types == 3:
            _, out_g = vmap_over_agents_env_lab_high_res(
                state.agents.params[classes[0][0]][None], key_env,
                random.split(subkey_sim, 1), model,
                self.cfg._replace(log_grid=True))
            dispo = self.available_by_type(out_g, n_types)

        for rang, (slot, born) in enumerate(classes, start=1):
            subkey_sim, k_mut, k_sim, k_par = random.split(subkey_sim, 4)
            parent  = state.agents.params[slot]
            enfants = vmap_mutate(parent, random.split(k_mut, n_enf), self.cfg)
            cles    = random.split(k_sim, n_enf)

            # le parent, dans le meme env et avec le meme protocole
            _, out_p = vmap_over_agents_env_lab_high_res(
                parent[None], key_env, random.split(k_par, 1), model, cfg_m)
            agg_p = self._agg_lab(out_p)

            agg, mange = None, []
            for deb in range(0, n_enf, EVO_BATCH):
                tr = slice(deb, deb + EVO_BATCH)
                _, out = vmap_over_agents_env_lab_high_res(
                    enfants[tr], key_env, cles[tr], model, cfg_m)
                a = self._agg_lab(out)
                agg = a if agg is None else {k: np.concatenate([agg[k], a[k]])
                                             for k in agg}
                mange.append(self.eaten_by_type(out))

            # un agent vivant par env, donc agg[k] suit l'ordre des env et
            # s'aligne ligne a ligne avec eaten_by_type
            if n_types == 3:
                plot_food_simplex(
                    np.concatenate(mange), ids_base, agg["age"], dispo,
                    exp_dir, self.chunk_idx,
                    suffix=f"_parent_{rang}",
                    titre=f"offspring of parent {rang} (slot {slot})",
                    fig_dir=os.path.join(exp_dir, "fig", "evolvability"),
                    parent=self.eaten_by_type(out_p)[0],
                    age_max=self.cfg.lab_time_steps)

            for k in EVO_METRIQUES:
                enfants_m[k].append(np.asarray(agg[k], dtype=float))
                v = np.asarray(agg_p[k], dtype=float)
                parent_m[k].append(float(np.nanmean(v)) if v.size else np.nan)
            etiquettes.append(str(rang))   # le slot reste dans parent_slot

        # les longueurs peuvent differer d'un parent a l'autre -> NaN de bourrage
        def _rect(listes):
            L = max((x.size for x in listes), default=0)
            out = np.full((len(listes), L), np.nan)
            for i, x in enumerate(listes):
                out[i, :x.size] = x
            return out

        d_dir = os.path.join(exp_dir, "evolvability")
        os.makedirs(d_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(d_dir, f"chunk_{self.chunk_idx}.npz"),
            etiquettes=np.array(etiquettes),
            parent_slot=np.array([s for s, _ in classes]),
            parent_born=np.array([b for _, b in classes]),
            **{f"enfants_{k}": _rect(enfants_m[k]) for k in EVO_METRIQUES},
            **{f"parent_{k}":  np.array(parent_m[k]) for k in EVO_METRIQUES})

        plot_evolvability(d_dir, self.chunk_idx)
        print(f"[evolvability] chunk {self.chunk_idx} : "
              f"{len(classes)} genomes x {n_enf} enfants en {time.time()-t0:.0f} s")

    def replay_top_gain(self, agent_params, gains, key_env, subkey_sim, model,
                        exp_dir, submit_video=None, rot=0, name="", resources=None):
        """Rejoue les genomes au plus fort gain avec replay_keys graines.

        Un gros ecart sur UN rollout peut venir d'une seule action qui bascule :
        les deux bras partagent la meme key_sim, donc ils restent identiques
        jusqu'a la premiere divergence, apres quoi ils se decorrelent. Rejouer
        le meme genome sur beaucoup de graines separe les deux lectures --
        distribution centree sur du positif = apprentissage, centree sur zero
        avec des queues = point de bascule.

        `rot` selectionne l'environnement PERMUTE : selection et rejeu doivent
        s'y faire tous les deux, sinon on teste la memoire la ou le genome n'en
        a pas besoin.
        """
        g = np.asarray(gains, dtype=float)
        ordre = np.argsort(np.where(np.isfinite(g), g, -np.inf))[::-1]
        ordre = [int(b) for b in ordre[:self.cfg.replay_top_n] if np.isfinite(g[int(b)])]
        if not ordre:
            return
        n_cles = self.cfg.replay_keys
        cfg_f  = self.cfg._replace(log_grid=False)
        cfg_a  = cfg_f._replace(ablate_recurrence=True)
        t0 = time.time()

        gains_rejoues, observes, cles_par_genome = [], [], []
        for b in ordre:
            subkey_sim, k = random.split(subkey_sim)
            cles   = random.split(k, n_cles)
            params = jnp.broadcast_to(agent_params[b], (n_cles, agent_params.shape[1]))
            gf, ga = [], []
            for d in range(0, n_cles, EVO_BATCH):
                tr = slice(d, d + EVO_BATCH)
                _, of = _vmap_rot(params[tr], key_env, cles[tr], cfg_f, model, rot)
                _, oa = _vmap_rot(params[tr], key_env, cles[tr], cfg_a, model, rot)
                gf.append(self.data_lab_env_grouped(of, resources)["age"])
                ga.append(self.data_lab_env_grouped(oa, resources)["age"])
            gains_rejoues.append(np.concatenate(gf) - np.concatenate(ga))
            observes.append(float(g[b]))
            cles_par_genome.append(cles)

        d_dir = os.path.join(exp_dir, "lab_data")
        os.makedirs(d_dir, exist_ok=True)
        sfx = f"_{name}" if name else ""
        gains_rejoues = np.stack(gains_rejoues)
        np.savez_compressed(
            os.path.join(d_dir, f"chunk_{self.chunk_idx}_replay{sfx}.npz"),
            gains=gains_rejoues, observe=np.array(observes),
            genome=np.array(ordre))

        # on ne filme que les genomes dont l'effet est net : en dessous du seuil
        # les deux bras ne different que par du bruit, il n'y a rien a voir.
        frac = np.array([(x > 0).mean() for x in gains_rejoues])
        seuil = self.cfg.replay_video_min_frac
        if submit_video is not None:
            for j in np.where(frac > seuil)[0]:
                gj = gains_rejoues[j]
                # graine mediane : un comportement representatif, pas l'extreme
                i_cle = int(np.argmin(np.abs(gj - np.median(gj))))
                cfg_v = cfg_f._replace(log_grid=True)
                un = jnp.broadcast_to(agent_params[ordre[j]], (1, agent_params.shape[1]))
                cle = cles_par_genome[j][i_cle:i_cle + 1]
                for nom, c in (("memory", cfg_v),
                               ("ablated", cfg_v._replace(ablate_recurrence=True))):
                    _, out = _vmap_rot(un, key_env, cle, c, model, rot)
                    chemin = os.path.join(exp_dir, "videos", "replay",
                                          f"genome_{ordre[j]}_chunk_{self.chunk_idx}{sfx}_{nom}.mp4")
                    submit_video(outputs_to_numpy(jax.tree_util.tree_map(lambda x: x[0], out)),
                                 chemin, 20, 10, self.cfg.resources,
                                 label=f"replay_{ordre[j]}{sfx}_{nom}")
                print(f"[replay] video du genome {ordre[j]} : "
                      f"{100*frac[j]:.0f}% > 0, graine {i_cle}")
            if not (frac > seuil).any():
                print(f"[replay] aucun genome au-dessus de {100*seuil:.0f}% > 0, "
                      f"pas de video (max {100*frac.max():.0f}%)")
        plot_replay_top_gain(d_dir, self.chunk_idx, suffix=sfx,
                             fig_dir=os.path.join(exp_dir, "fig"))
        print(f"[replay] chunk {self.chunk_idx}{sfx} : {len(ordre)} genomes x "
              f"{n_cles} graines en {time.time()-t0:.0f} s")

    def launch_env(self, state, key_env, subkey_sim, model, exp_dir, n, submit_video):
            survivors = self.compute_survivors(state)
            ids = np.array([agent_id for agent_id, _ in survivors[:n]])
            agent_params = state.agents.params[ids]
            key_sim = random.split(subkey_sim, len(ids))

            def agent_slice(state, b):
                return jax.tree_util.tree_map(lambda x: x[b], state)

            # `grid` n'est lu que par les videos et par available_by_type (pas 0).
            # Le journaliser sur les n genomes coute ~2,7 Go par env a
            # lab_time_steps=3000, pour 3 genomes filmes -- l'env adapt, qui
            # empile ses rotations, a fait tomber le GPU en OOM sur 5,03 Gio.
            # On le coupe donc sur le rollout de MESURE et on rejoue les N_FILM
            # premiers genomes avec la grille. vmap est element par element :
            # agent_params[:k] et key_sim[:k] redonnent trait pour trait les
            # trajectoires des k premiers genomes du rollout de mesure.
            cfg_m = self.cfg._replace(log_grid=False)
            cfg_v = self.cfg._replace(log_grid=True)

            def rollout_video(lanceur, cfg=None):
                _, out = lanceur(agent_params[:N_FILM], key_env,
                                 key_sim[:N_FILM], model, cfg or cfg_v)
                return out

            # ============ 1) HIGH_RES (agent seul) ============
            final_state, outputs_high = vmap_over_agents_env_lab_high_res(
                agent_params, key_env, key_sim, model, cfg_m)
            agg, summary = self.data_lab_env(outputs_lab=outputs_high)
            self._save_lab_data(agg, summary, exp_dir)

            plot_lab_metrics(exp_dir=exp_dir)
            self._plot_energy(outputs_high, exp_dir, "high_res",
                            "lab_1 — high_res (agent alone)")

            vid_high = rollout_video(vmap_over_agents_env_lab_high_res)
            for b in range(min(2, N_FILM)):
                vid = os.path.join(exp_dir, "videos", "high",
                                f"high_res_video_chunk_{self.chunk_idx}_lab_{b}.mp4")
                submit_video(outputs_to_numpy(agent_slice(vid_high, b)), vid, 20, 10,
                            self.cfg.resources,
                            label=f"high_res_chunk_{self.chunk_idx}_lab_{b}")

            # ============ 2) LOW_RES (exploration) ============
            final_state, outputs_low = vmap_over_agents_env_lab_low_res(
                agent_params, key_env, key_sim, model, cfg_m)
            agg_low, summary_low = self.data_lab_env_low_res(outputs_low)
            self._save_lab_data(agg_low, summary_low, exp_dir, suffix="lowres")

            plot_lab_exploration(exp_dir=exp_dir)
            self._plot_energy(outputs_low, exp_dir, "low_res",
                            "lab_2 — low_res (exploration)")

            vid_low = rollout_video(vmap_over_agents_env_lab_low_res)
            for b in range(N_FILM):
                vid = os.path.join(exp_dir, "videos", "low",
                                f"low_res_video_chunk_{self.chunk_idx}_lab_{b}.mp4")
                submit_video(outputs_to_numpy(agent_slice(vid_low, b)), vid, 20, 10,
                            self.cfg.resources,
                            label=f"low_res_chunk_{self.chunk_idx}_lab_{b}")

            # ============ 3) CLONES (effet des pairs) ============
            final_state, outputs_clones = vmap_over_agents_env_lab_high_res_with_clones(
                agent_params, key_env, key_sim, model, cfg_m)
            # comparaison APPARIÉE avec l'env high_res (mêmes génomes, même ordre) :
            self.compare_alone_vs_clones(outputs_high, outputs_clones, exp_dir)
            self._plot_energy(outputs_clones, exp_dir, "high_res_clones",
                            "lab_3 — high_res with clones")

            # self.plot_energy_response_labs(outputs_high, outputs_low, outputs_clones, exp_dir)
            vid_clones = rollout_video(vmap_over_agents_env_lab_high_res_with_clones)
            for b in range(N_FILM):
                vid = os.path.join(exp_dir, "videos", "high_res_clones",
                                f"high_res_clones_video_chunk_{self.chunk_idx}_lab_{b}.mp4")
                submit_video(outputs_to_numpy(agent_slice(vid_clones, b)), vid, 20, 10,
                            self.cfg.resources,
                            label=f"clones_chunk_{self.chunk_idx}_lab_{b}")

            # ============ 4) ADAPTATION (rotations des canaux) ============
            # A une ressource il n'y a aucune permutation a tester : tous les
            # rollouts adapt sont sautes, et la boucle sur rotations_for plus bas
            # ne tourne pas non plus.
            rotations = rotations_for(self.cfg.resources)
            final_state = outputs_adapt = None
            if rotations:
                final_state, outputs_adapt = vmap_over_agents_env_lab_adapt(
                    agent_params, key_env, key_sim, model, cfg_m)
            # outputs_adapt : axe 0 = agent (B), axe 1 = rotation (2)

            # Le MEME rollout, memes genomes, memes cles, memoire coupee. C'est
            # le controle de l'adaptation intra-vie : si la baisse persiste sans
            # memoire, elle ne vient pas d'un apprentissage. Ablation au moment
            # du TEST et non a l'evolution, pour que la comparaison reste appariee.
            outputs_adapt_abl = outputs_high_abl = None
            if self.cfg.lab_memory_ablation:
                cfg_abl = cfg_m._replace(ablate_recurrence=True)   # grille non journalisee aussi
                if rotations:
                    _, outputs_adapt_abl = vmap_over_agents_env_lab_adapt(
                        agent_params, key_env, key_sim, model, cfg_abl)
                _, outputs_high_abl = vmap_over_agents_env_lab_high_res(
                    agent_params, key_env, key_sim, model, cfg_abl)
                self.compare_memory(outputs_high, outputs_high_abl, exp_dir)

            # Bras propre a la boucle interne : MEME reseau, MEME genome, memes
            # cles -- seul le gradient est coupe, donc v_pred reste a sa valeur
            # evoluee. C'est ce que le gradient rapporte, et rien d'autre :
            # l'ablation de la memoire ci-dessus ne repond pas a cette
            # question-la, elle coupe aussi le carry.
            outputs_adapt_gele = outputs_high_gele = None
            if self.cfg.inner_loop:
                cfg_gele = cfg_m._replace(inner_loop=False)
                if rotations:
                    _, outputs_adapt_gele = vmap_over_agents_env_lab_adapt(
                        agent_params, key_env, key_sim, model, cfg_gele)
                _, outputs_high_gele = vmap_over_agents_env_lab_high_res(
                    agent_params, key_env, key_sim, model, cfg_gele)
                self.compare_memory(outputs_high, outputs_high_gele, exp_dir,
                                    bras=("inner", "gele"),
                                    entete="GRADIENT INTRA-VIE vs POIDS GELES",
                                    label_intact="gradient on",
                                    label_bras2="weights frozen for life",
                                    titre_fig="Same genomes, with and without the inner loop",
                                    fname_fig="lab_inner_loop_evolution",
                                    titre_gain="Gain de la boucle interne, par genome")

            # Controle apparie : lab_1 partage agent_params / key_env / key_sim et
            # le meme in_axes que l'env adapt -> l'index b designe le MEME genome
            # dans les deux, seule la permutation des canaux differe.
            vid_adapt = rollout_video(vmap_over_agents_env_lab_adapt) if rotations else None

            eaten_baseline = self.eaten_by_type(outputs_high)
            baseline_ids   = [r.id for r in self.cfg.resources]
            # Plafond lu sur la grille de CHAQUE env : la permutation change le canal
            # de chaque ressource, donc les tirages de croissance a l'init different
            # legerement d'une rotation a l'autre -> un plafond par cote.
            n_types = len(self.cfg.resources)
            av_base = self.available_by_type(vid_high, n_types)
            baseline_available = {r.id: av_base[k] for k, r in enumerate(self.cfg.resources)}

            # composition du regime, un point par agent. Le simplex n'a de sens
            # qu'a trois ressources ; BASE_RESOURCES est deja passe a deux.
            par_genome = self.data_lab_env_grouped(outputs_high)
            if n_types == 3:
                plot_food_simplex(
                    eaten_baseline, baseline_ids, par_genome["age"],
                    av_base, exp_dir, self.chunk_idx, titre="lab_1 — high_res",
                    age_max=self.cfg.lab_time_steps)
            else:
                print(f"Simplex : {n_types} ressources, il en faut 3.")

            plot_metric_pairs(par_genome, exp_dir, self.chunk_idx,
                                    titre="lab_1 — high_res")

            # Depuis quand l'agent vit-il avec la config qu'on s'apprete a casser ?
            # C'est ce qui dit s'il a eu le temps de l'apprendre.
            steps_in_place = steps_since_last_shuffle(load_shuffle_log(exp_dir),
                                                      int(state.step))

            # Reference : la meme propension dans l'env NON permute, memes genomes.

            # Idem pour CHAQUE identite. Calcule une seule fois : la baseline ne
            # depend pas de la rotation, la recalculer par rotation serait deux
            # fois le meme balayage de `obs`.
            base_par_type = {}
            for r in self.cfg.resources:
                _, bn_i, bk_i = self.prob_eat_over_life(
                    outputs_high, self.cfg.resources, label=LABELS[r.id])
                base_par_type[r.id] = (bn_i, bk_i)

            for j, rot in enumerate(rotations):          # j = position sur l'axe, rot = vraie rotation
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

                agg_r, summary_r = self.data_lab_env(outputs_lab=out_rot,
                                                     resources=resources_rot)
                self._save_lab_data(agg_r, summary_r, exp_dir, suffix=f"adapt_{name}")
                plot_lab_metrics(exp_dir=exp_dir, suffix=f"adapt_{name}")
                self._plot_energy(out_rot, exp_dir, f"adapt/{name}", title)

                # Duree de vie avec et sans memoire SOUS CETTE PERMUTATION. Le
                # meme genome des deux cotes (memes cles, seul ablate_memory
                # differe), donc un simple appariement. C'est ici que l'ecart
                # doit se creuser : dans l'env non permute le genome peut suffire.
                out_abl = None
                if outputs_adapt_abl is not None:
                    out_abl = jax.tree_util.tree_map(lambda x: x[:, j],
                                                     outputs_adapt_abl)
                    _, gain_rot = self.compare_memory(out_rot, out_abl, exp_dir,
                                        suffix=f"_adapt_{name}",
                                        env_titre=f"adapt {name}",
                                        resources=resources_rot)
                    # Rejeu des meilleurs genomes dans l'env PERMUTE. Depend de
                    # gain_rot, donc doit rester DANS ce bloc : l'en sortir le
                    # laisserait non defini quand lab_memory_ablation est faux.
                    if self.cfg.replay_top_n > 0:
                        subkey_sim, k_rej = random.split(subkey_sim)
                        self.replay_top_gain(agent_params, gain_rot, key_env, k_rej,
                                             model, exp_dir, submit_video=submit_video,
                                             rot=rot, name=f"adapt_{name}",
                                             resources=resources_rot)
                # C'est ICI que la boucle interne doit payer : sous permutation,
                # la valeur des canaux a change et seul un apprentissage pendant
                # la vie peut la retrouver.
                if outputs_adapt_gele is not None:
                    out_gele = jax.tree_util.tree_map(lambda x: x[:, j],
                                                      outputs_adapt_gele)
                    self.compare_memory(out_rot, out_gele, exp_dir,
                                        suffix=f"_adapt_{name}",
                                        env_titre=f"adapt {name}",
                                        resources=resources_rot,
                                        bras=("inner", "gele"),
                                        entete="GRADIENT INTRA-VIE vs POIDS GELES",
                                        label_intact="gradient on",
                                        label_bras2="weights frozen for life",
                                        titre_fig="Same genomes, with and without "
                                                  "the inner loop",
                                        fname_fig="lab_inner_loop_evolution",
                                        titre_gain="Gain de la boucle interne, "
                                                   "par genome")

                av_rot = self.available_by_type(
                    jax.tree_util.tree_map(lambda x: x[:, j], vid_adapt), n_types)

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

                # Les TROIS identites sur les memes tranches. Repond a la fois a
                # "la ressource qui prend la place du poison est-elle plus
                # consommee ?" et a "la baisse du poison est-elle de la
                # selectivite ou un appetit general qui retombe ?".
                par_type = {}
                for r in resources_rot:
                    _, n_i, k_i = self.prob_eat_over_life(
                        out_rot, resources_rot, label=LABELS[r.id])
                    par_type[r.id] = (n_i, k_i)
                plot_prob_eat_over_life_by_type(
                    par_type, base_par_type, _wilson, exp_dir,
                    chunk=self.chunk_idx, tag=name, mapping=caption,
                )
                # Le meme contenu, mais l'ecart au non-permute trace directement :
                # la baisse d'appetit liee a l'age s'annule dans la soustraction.
                # DESACTIVE -- figure jugee peu informative en pratique. Le
                # rapport reste immune a un effet d'echelle multiplicatif la ou
                # l'exces ne l'est pas ; plot_prob_eat_ratio est conserve dans
                # plots.py, il suffit de decommenter pour le reactiver.
                # plot_prob_eat_ratio(
                #     par_type, base_par_type, exp_dir,
                #     chunk=self.chunk_idx, tag=name, mapping=caption,
                # )

                for b in range(min(2, N_FILM)):
                    vid = os.path.join(exp_dir, "videos", "adapt", name,
                                    f"adapt_{name}_chunk_{self.chunk_idx}_lab_{b}.mp4")
                    submit_video(outputs_to_numpy(agent_slice(
                                     jax.tree_util.tree_map(lambda x: x[:, j], vid_adapt), b)),
                                 vid, 20, 10,
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
    def _per_agent_metrics(self, outputs, resources=None):
            """Métriques par agent, alignées par événement de fin de vie
            (mort OU survie jusqu'au dernier pas = censure à droite).
            Toutes les sorties sont des tableaux 1D de longueur D (nb d'événements).

            `resources` donne le contenu de chaque CANAL pour ce rollout. L'env
            adapt permute les canaux : sans ce paramètre, delta_energy serait lu
            dans l'ordre de base et greediness comme adapt_score porteraient sur
            les mauvais canaux."""
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
            res_ch        = resources if resources is not None else self.cfg.resources
            delta_e       = np.array([r.delta_energy for r in res_ch])
            n_channels    = len(res_ch) + 2                   # ressources + agents + murs
            good_channels = np.where(delta_e > 0)[0]
            saw = _resource_in_view(outputs.obs, good_channels, n_channels)
            ate_step = np.zeros_like(saw, dtype=bool)          # consommation alignee
            if REWARD_LAG > 0:
                ate_step[:-REWARD_LAG] = rew[REWARD_LAG:] > 0
            else:
                ate_step = rew > 0
            greediness, greed_Tr, greed_Cr = _greediness(
                saw, ate_step, slot, birth_row, death_row)

            # 10) Score d'adaptation = différentiel de sélection
            #
            #     Δe moyen de ce qu'il a MANGÉ  −  Δe moyen de ce qu'il a VU
            #
            # 0 = mange indifféremment ce qu'il croise, >0 = choisit mieux que
            # l'offre. Normalisé par la disponibilité, donc insensible à un env
            # plus riche ou plus pauvre, et sans dimension de durée.
            #
            # À préférer à l'énergie récoltée Σ mangé·Δe, qui vaut par
            # conservation (E_fin − E_début) + décroissance, soit essentiellement
            # la durée de vie : elle ne dit rien de la qualité des choix.
            saw_t = np.asarray(outputs.saw_res).astype(float)   # (T, N, n_types)
            ate_t = np.asarray(outputs.ate_res).astype(float)
            n_vu   = np.stack([window_sum(np.cumsum(saw_t[:, :, i], axis=0),
                                          birth_row - 1, death_row)
                               for i in range(len(delta_e))])   # (n_types, D)
            n_mange = np.stack([window_sum(np.cumsum(ate_t[:, :, i], axis=0),
                                           birth_row - 1, death_row)
                                for i in range(len(delta_e))])
            def _moyenne_ponderee(w):
                tot = w.sum(axis=0)
                return np.divide((w * delta_e[:, None]).sum(axis=0), tot,
                                 out=np.full(tot.shape, np.nan), where=tot > 0)
            # NaN si l'agent n'a rien vu ou rien mangé : indéfini, pas nul
            adapt_score = _moyenne_ponderee(n_mange) - _moyenne_ponderee(n_vu)

            # 11) Gain net par pas de vie, gains filtres par la faim.
            #
            # adapt_score compare des COMPOSITIONS : il ignore les quantites, punit
            # de refuser du good a satiete et rend NaN l'agent qui s'abstient de
            # tout. On somme donc l'energie reellement acquise.
            #
            # Le masque de faim ne porte que sur les GAINS : le good est clippe
            # par energy_max, donc sans valeur a satiete -- le refuser est
            # rationnel et ne doit pas compter. Le poison n'est jamais clippe, il
            # coute son delta quel que soit le niveau d'energie : le masquer
            # reviendrait a ne pas voir la bouchee de poison d'un agent rassasie,
            # qui est justement ce qu'on cherche.
            seuil_faim = self.cfg.energy_max - max(float(delta_e.max()), 0.0)
            faim = energy < seuil_faim                          # (T, N)
            # energy[t] precede l'action du pas t, ate_res[t] en resulte : alignes.
            gain_pos = (ate_t * np.maximum(delta_e, 0.0)).sum(axis=2)   # (T, N)
            gain_neg = (ate_t * np.minimum(delta_e, 0.0)).sum(axis=2)   # (T, N)
            gain_t = np.where(faim, gain_pos, 0.0) + gain_neg
            num = window_sum(np.cumsum(gain_t, axis=0), birth_row - 1, death_row)
            # Denominateur = la fenetre de VIE, pas les seuls pas de faim : sinon
            # un agent jamais affame qui s'empoisonne donnerait NaN au lieu de
            # compter son degat. Dans le lab les durees de vie sont quasi
            # constantes, donc ca ne reintroduit pas la confusion avec l'age --
            # sauf dans compare_memory, ou c'est precisement l'effet mesure.
            adapt_gain = num / (age + 1)

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
                "adapt_score": adapt_score,  # Δe mangé − Δe vu (NaN si rien)
                "adapt_gain": adapt_gain,    # (gains si faim + pertes) / pas de vie
            }
 
    # =================================================================
    #  MODIFIÉ — _stat supprimé, remplacé par **_dispersion
    # =================================================================
    def data_lab_env(self, outputs_lab, resources=None):
        agg = self._agg_lab(outputs_lab, resources)
        return agg, self._summary_lab(agg, outputs_lab.alive.shape[0])

    def _agg_lab(self, outputs_lab, resources=None):
        keys = ["age", "total_rew", "mean_rew", "total_move",
                "mean_speed", "energy_end", "wall_death", "died",
                "greediness", "adapt_score", "adapt_gain"]
        agg = {k: [] for k in keys}
 
        B = outputs_lab.alive.shape[0]          # nb d'environnements = nb d'agents testés
        for b in range(B):
            single = jax.tree_util.tree_map(lambda x: x[b], outputs_lab)
            m = self._per_agent_metrics(single, resources)
            if m is None:
                continue
            for k in keys:
                agg[k].append(m[k])
 
        return {k: (np.concatenate(v) if v else np.array([])) for k, v in agg.items()}

    def _summary_lab(self, agg, B):
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
            # NaN pour un agent n'ayant rien vu ou rien mange
            **_dispersion(_clean(agg["adapt_score"]), "adapt_score", empty=float("nan")),
            **_dispersion(_clean(agg["adapt_gain"]), "adapt_gain", empty=float("nan")),
        }
        return summary
 
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
    def data_lab_env_grouped(self, outputs_lab, resources=None):
        """Réduit le comportement À L'INTÉRIEUR de chaque environnement.
        Dans clones, les agents vivants partagent le même génome ; on les
        moyenne -> un seul profil comportemental par génome. Pour high_res
        (1 agent/env) la moyenne est triviale. Utiliser la MÊME fonction pour
        les deux garantit des définitions de métriques identiques.
 
        Retourne per_genome : dict de tableaux (B,), alignés par index de
        génome (même ordre que agent_params / key_sim).
        """
        keys = ["age", "mean_rew", "mean_speed", "energy_end", "wall_death",
                "died", "greediness", "adapt_score", "adapt_gain"]
        B = outputs_lab.alive.shape[0]
        per_genome = {k: np.full(B, np.nan) for k in keys}
        n_peers    = np.zeros(B, dtype=int)
 
        for b in range(B):
            single = jax.tree_util.tree_map(lambda x: x[b], outputs_lab)
            m = self._per_agent_metrics(single, resources)
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
 
        # sans mur letal, wall_death vaut 0 partout : panneau vide et trompeur
        metrics = ["age", "mean_rew", "mean_speed", "energy_end",
                   "greediness", "adapt_score", "adapt_gain"]
        if self.cfg.letal_wall:
            metrics.insert(4, "wall_death")
        labels  = {"age": "lifespan (steps)", "mean_rew": "consumption /step",
                   "mean_speed": "movement /step", "energy_end": "final energy",
                   "wall_death": "fraction wall deaths",
                   "greediness": "greediness G = Cr/Tr",
                   "adapt_score": "adaptation (de eaten - seen)",
                   "adapt_gain": "net gain /hungry step"}
 
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

    def compare_memory(self, outputs_full, outputs_abl, exp_dir, suffix="",
                       env_titre="high_res (unpermuted)", resources=None,
                       bras=("memory", "ablated"), entete="MEMOIRE INTACTE vs COUPEE",
                       label_intact=None, label_bras2=None,
                       titre_fig="Same genomes, with and without within-life memory",
                       fname_fig="lab_memory_ablation_evolution",
                       titre_gain="Gain de la memoire, par genome"):
        """Compare, PAR GENOME, deux bras qui ne different que par un facteur.

        `bras` nomme les deux cotes : ("memory", "ablated") pour la memoire,
        ("inner", "gele") pour la boucle interne. Il sert de prefixe aux cles du
        npz et de tag de fichier, donc chaque comparaison a sa propre serie.

        `suffix` indexe la famille de fichiers, donc une comparaison par
        environnement : "" pour l'env non permute, "_adapt_<condition>" pour
        chaque permutation. C'est sous permutation que la memoire est censee
        servir le plus -- l'env non permute est celui pour lequel le genome a
        deja ete selectionne, l'agent peut y survivre en aveugle.

        Meme patron apparie que compare_alone_vs_clones : memes agent_params,
        memes key_env / key_sim, seul ablate_memory differe -> l'index b designe
        le meme genome des deux cotes.

        C'est la comparaison la plus simple et la plus robuste dont on dispose :
        la DUREE DE VIE est un scalaire par agent, sans conditionnement, sans
        fenetrage, sans biais de composition. Si la memoire rallonge la vie, elle
        sert fonctionnellement a quelque chose, et ce constat ne depend d'aucune
        des hypotheses qui fragilisent les mesures "par tranche de vie".

        Elle a d'ailleurs une consequence sur celles-ci : des durees de vie
        differentes rendent les tranches "0-25% de sa vie" non comparables entre
        les deux conditions, puisqu'elles couvrent des pas absolus differents,
        donc des etats d'environnement differents."""
        f = self.data_lab_env_grouped(outputs_full, resources)
        a = self.data_lab_env_grouped(outputs_abl, resources)

        # sans mur letal, wall_death vaut 0 partout : panneau vide et trompeur
        metrics = ["age", "mean_rew", "mean_speed", "energy_end",
                   "greediness", "adapt_score", "adapt_gain"]
        if self.cfg.letal_wall:
            metrics.insert(4, "wall_death")
        labels  = {"age": "lifespan (steps)", "mean_rew": "consumption /step",
                   "mean_speed": "movement /step", "energy_end": "final energy",
                   "wall_death": "fraction wall deaths",
                   "greediness": "greediness G = Cr/Tr",
                   "adapt_score": "adaptation (de eaten - seen)",
                   "adapt_gain": "net gain /hungry step"}

        table = {}
        par_genome = {}
        for k in metrics:
            mask  = ~(np.isnan(f[k]) | np.isnan(a[k]))
            delta = a[k][mask] - f[k][mask]          # ablate - intact
            row = {"n": int(mask.sum())}
            row.update(_dispersion(f[k][mask], bras[0], empty=float("nan")))
            row.update(_dispersion(a[k][mask], bras[1], empty=float("nan")))
            row.update(_dispersion(delta,      "delta",    empty=float("nan")))
            table[k] = row
            # signe inverse du tableau : positif = la memoire AIDE
            par_genome[f"gain_{k}"] = f[k][mask] - a[k][mask]
            if k == "age":
                gain_brut = f[k] - a[k]        # non masque : indices = agent_params
            par_genome[f"{bras[0]}_{k}"] = f[k][mask]
            par_genome[f"{bras[1]}_{k}"] = a[k][mask]

        print(f"\n--- Lab chunk {self.chunk_idx} | {entete} | {env_titre} ---")
        print(f"  {'metric':<22}{bras[0]:>10}{bras[1]:>10}{'Δ median':>11}{'Δ IQR':>20}")
        for k in metrics:
            r = table[k]
            print(f"  {labels[k]:<22}{r[bras[0] + '_p50']:>10.3f}{r[bras[1] + '_p50']:>10.3f}"
                  f"{r['delta_p50']:>11.3f}"
                  f"{'[' + format(r['delta_p25'], '.3f') + ', ' + format(r['delta_p75'], '.3f') + ']':>20}")

        data_dir = os.path.join(exp_dir, "lab_data")
        os.makedirs(data_dir, exist_ok=True)
        payload = {"chunk": self.chunk_idx + 1, "metrics": table}
        tag = f"{bras[0]}{suffix}"
        with open(os.path.join(data_dir,
                               f"chunk_{self.chunk_idx}_{tag}.json"), "w") as fh:
            json.dump(payload, fh, indent=2)

        # les deltas par genome, que la mediane et l'IQR effacent
        np.savez_compressed(
            os.path.join(data_dir, f"chunk_{self.chunk_idx}_{tag}_pergenome.npz"),
            **par_genome)

        # Le bras "intact" ne l'est que si la sim elle-meme n'ablate rien. Quand
        # elle tourne deja avec un canal coupe, on trace quand meme -- la
        # comparaison reste appariee et informative -- mais l'etiquette doit le
        # dire, sinon la figure ment. Les prefixes de cles, eux, ne bougent pas :
        # ce sont eux qui font la continuite de la serie.
        coupes = [n for n, on in (("recurrence", self.cfg.ablate_recurrence),
                                  ("interoception", self.cfg.ablate_interoception),
                                  ("feedback", self.cfg.ablate_feedback))
                  if on or self.cfg.ablate_memory]
        ref = label_intact or ("memory intact" if not coupes
                               else f"as evolved ({'+'.join(coupes)} cut)")
        # Le bras ablate ne coupe que la RECURRENCE : l'interoception et le
        # feedback restent branches, et sous vpred_oracle l'information de
        # valeur aussi. "all channels ablated" laissait croire le contraire.
        if label_bras2 is None:
            label_bras2 = "recurrence cut"
            if self.cfg.vpred_oracle:
                label_bras2 += " (v_pred intact)"

        plot_alone_vs_clones(
            exp_dir=exp_dir, tag=tag,
            prefixes=bras,
            labels=(ref, label_bras2),
            titre=f"{titre_fig} — {env_titre}",
            fname=f"{fname_fig}{suffix}.png")
        plot_memory_gain_hist(exp_dir=exp_dir, tag=tag, suffix=suffix,
                              env_titre=env_titre,
                              axe_x=f"{ref} − {label_bras2}", titre=titre_gain)
        return table, gain_brut
    
    def plot_energy_response_labs(self, out_high, out_low, out_clones, exp_dir):
        curves = {
            "lab_1 high_res": energy_response_over_envs(out_high,   self.cfg),
            "lab_2 low_res":  energy_response_over_envs(out_low,    self.cfg),
            "lab_3 clones":   energy_response_over_envs(out_clones, self.cfg),
        }
        plot_energy_response(exp_dir, self.chunk_idx + 1, curves, cfg=self.cfg)