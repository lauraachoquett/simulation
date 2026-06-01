
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"
from jax import random
import jax.numpy as jnp
import jax
import os
import numpy as np
from simulation.one_simulation import run_simulation_chunk
from simulation.utils import plot_evolution,plot_current_config,save_checkpoint,_video_worker,save_config,create_exp_file,load_config,load_checkpoint,outputs_to_numpy,sec_to_minutes
from simulation.data_class import Config
from EcoEvoJax.source.agent import MetaRnnPolicy_bcppr
from simulation.utils import init_state,load_checkpoint,save_checkpoint
import multiprocessing as mp
import time
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna
import logging
import sys

import os

def get_next_dir(base_path=".", prefix="try_"):
    """
    Trouve le prochain dossier disponible avec le préfixe donné et le crée.
    """
    i = 1
    while True:
        dir_path = os.path.join(base_path, f"{prefix}{i}")
        if not os.path.exists(dir_path):
            # Création immédiate pour réserver l'emplacement
            os.makedirs(dir_path)
            return dir_path
        i += 1
        
def classify_outcome(pop_full, res_full, cfg):
    """
    Returns 'extinction', 'overpopulation', 'depletion', or 'interesting'.
    """
    n_chunks = 20
    if pop_full[-1] == 0:
        return 'extinction'
    
    last = pop_full[-n_chunks*cfg.chunk_size:]
    if len(last) == n_chunks*cfg.chunk_size and (last > 0.9 * cfg.n_agents_max).all():
        return 'overpopulation'
    
    if res_full[-1]==0:
        return 'depletion'
    
    last_res = res_full[-n_chunks*cfg.chunk_size:]
    if len(last_res) == n_chunks*cfg.chunk_size and (last_res > 0.95 * cfg.n ** 2).all():
        return 'easy'
    
    return 'interesting'

def objective(trial, base_cfg, base_key,current_dir):
    # 1. Définition de l'espace de recherche à 5 dimensions
    prob_factor = trial.suggest_float("prob_factor", 0.01, 1.0, log=True)
    energy_decay = trial.suggest_float("energy_decay", 0.001, 0.1, log=True)
    
    # Nouveaux paramètres (les bornes sont à ajuster selon ton modèle physique)
    time_to_die = trial.suggest_int("time_to_die", 10, 100)
    time_above_repr = trial.suggest_int("time_above_repr", 5, 50)
    min_energy_repr = trial.suggest_float("min_energy_repr", 0.0, 2.0)
    
    trial_cfg = base_cfg._replace(
        prob_factor=prob_factor,
        energy_decay=energy_decay,
        time_to_die=time_to_die,
        time_above_repr=time_above_repr,
        min_energy_repr=min_energy_repr
    )
    
    # 2. Vérification des contraintes d'intégrité (Sanity Check enrichi)
    a = trial_cfg.starting_energy - trial_cfg.energy_decay * trial_cfg.time_above_repr
    b = trial_cfg.min_energy_repr
    
    # On ajoute une vérification logique : le temps nécessaire avant reproduction 
    # doit être strictement inférieur au temps de vie total sans manger.
    if b <= a or trial_cfg.time_to_die <= trial_cfg.time_above_repr:
        raise optuna.TrialPruned("Contraintes physiques ou temporelles non respectées.")

    # 3. Lancement de la simulation
    trial_key, _ = jax.random.split(base_key)
    
    
    try:
        state, outputs, exp_dir, outcome = launch_simulation_chunked(
            trial_key, trial_cfg, resume_exp=None, n_video_workers=1, dir=current_dir
        )
    except Exception as e:
        raise optuna.TrialPruned(f"Erreur d'exécution: {e}")

    # 4. Calcul du score continu
    pop_full = outputs_to_numpy(outputs.agents.alive).sum(axis=1)
    chunks_survived = len(pop_full) // trial_cfg.chunk_size
    
    final_pop = pop_full[-1] if len(pop_full) > 0 else 0
    pop_ratio = final_pop / trial_cfg.n_agents_max
    
    score = chunks_survived + pop_ratio
    
    if outcome == 'overpopulation':
        score -= 5.0 
        
    return score

def run_optuna_search(cfg, key, n_trials=50):
    """
    Configure et lance l'étude d'optimisation.
    """
    # Pour réduire la verbosité d'Optuna dans la console
    optuna.logging.get_logger("optuna").setLevel(logging.INFO)
    
    study = optuna.create_study(
        study_name="ecoevo_hyperopt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42) # Arbre de Parzen pour l'optimisation bayésienne
    )
    
    # Passage des arguments fixes via une fonction lambda
    current_dir = get_next_dir(base_path="exp/", prefix="try_")
    study.optimize(lambda trial: objective(trial, cfg, key,current_dir), n_trials=n_trials)
    
    print("\n=== Recherche Optuna terminée ===")
    print(f"Meilleur score : {study.best_value}")
    print(f"Meilleurs paramètres : {study.best_params}")
    
    # Retourne la configuration optimale
    best_cfg = cfg._replace(**study.best_params)
    return best_cfg, study
    
    
def launch_simulation_chunked(key, cfg, resume_exp=None, n_video_workers=2, chunk_id=0,dir=''):

    start_time_sim = time.time()
    if dir == '':
        now = datetime.now()
        dir = os.path.join("exp", now.strftime("%Y-%m-%d"))
    exp_dir = create_exp_file(dir)
    num_chunks_exp = cfg.num_chunks + chunk_id

    if resume_exp is not None and os.path.exists(resume_exp):
        print(f"Reprise depuis {resume_exp} au chunk {chunk_id}")
        state = load_checkpoint(resume_exp, chunk_id)
        cfg, subkeys = load_config(resume_exp)
        if len(subkeys) < num_chunks_exp:
            new_keys = random.split(key, num_chunks_exp - len(subkeys))
            subkeys.extend(new_keys)

        start_chunk = chunk_id
        model = MetaRnnPolicy_bcppr(
            input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 2),
            hidden_dim=4, output_dim=4, encoder_layers=[], hidden_layers=[8]
        )
    else:
        model = MetaRnnPolicy_bcppr(
            input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 2),
            hidden_dim=4, output_dim=4, encoder_layers=[], hidden_layers=[8]
        )
        key, *subkeys = random.split(key, num_chunks_exp + 1)
        key, subkey = jax.random.split(key)
        state = init_state(subkey, cfg, model)
        start_chunk = 0
        
    save_config(cfg, subkeys, exp_dir)
    start_step = start_chunk * cfg.chunk_size
    

    pop_history = []
    res_history = []
    initial_grid_res = state.grid[0, :, :]
    
    plot_current_config(initial_grid_res, state.agents.position, state.agents.alive, exp_dir,name_fig=f'{start_chunk}')

    pending_futures = {}  # future -> chunk_idx
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=n_video_workers,mp_context=ctx) as executor:
        for chunk_idx in range(start_chunk, num_chunks_exp):
            
            # --- Collecte non-bloquante des vidéos déjà terminées ---
            done = [f for f in pending_futures if f.done()]
            for f in done:
                cidx = pending_futures.pop(f)
                try:
                    path = f.result()
                    print(f"  [video] chunk {cidx} sauvegardé : {path}")
                except Exception as e:
                    print(f"  [video] ERREUR chunk {cidx} : {e}")

            # --- Simulation GPU ---
            subkey = subkeys[chunk_idx]
            keys_chunk = jax.random.split(subkey, cfg.chunk_size)
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx+1} START  @ {time.strftime('%H:%M:%S')}")
            state, outputs = run_simulation_chunk(state, model, keys_chunk, cfg)
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx+1} DONE   @ {time.strftime('%H:%M:%S')}")
            
            # --- Plots (CPU léger, synchrone) ---
            pop_history.append(np.array(outputs.agents.alive.sum(axis=1)))
            res_history.append(np.array(outputs.grid[:, 0, :, :].sum(axis=(1, 2))))
            plot_evolution(
                np.concatenate(pop_history, axis=0),
                np.concatenate(res_history, axis=0),
                exp_dir,
                start_step
            )
            pop_full = np.concatenate(pop_history)
            res_full = np.concatenate(res_history)
            
            plot_current_config(state.grid[0, :, :], state.agents.position, state.agents.alive, exp_dir,name_fig=f'{chunk_idx}')
            
            current_sim_state = classify_outcome(pop_full, res_full, cfg)
            if current_sim_state != 'interesting':
                print(f"Stopping criterion : {current_sim_state}")
                return state, outputs, exp_dir, current_sim_state

            # --- Checkpoint (synchrone) ---
            if (chunk_idx + 1) % cfg.checkpoint_freq == 0:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"state_chunk_{chunk_idx+1}.pkl")
                save_checkpoint(state, ckpt_path)

            # --- Vidéo (asynchrone) ---
            if (chunk_idx + 1) % cfg.video_freq == 0:
                vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{chunk_idx+1}.mp4")
                #    Conversion numpy AVANT envoi — bloque le GPU le temps du transfert H→D,
                #     mais libère ensuite le GPU pour le chunk suivant pendant l'encodage vidéo
                outputs_np = outputs_to_numpy(outputs)
                future = executor.submit(_video_worker, outputs_np, vid_path, 20, 5)
                pending_futures[future] = chunk_idx + 1

        # --- Attente finale de toutes les vidéos restantes ---
        print("Simulation terminée. Attente des vidéos en cours...")
        for f in as_completed(pending_futures):
            cidx = pending_futures[f]
            try:
                print(f"  [video] chunk {cidx} finalisé : {f.result()}")
            except Exception as e:
                print(f"  [video] ERREUR chunk {cidx} : {e}")

    delta_sim = time.time()-start_time_sim
    delta_min,delta_sec = sec_to_minutes(delta_sim)
    print(f"Time to compute the simulation : {delta_min} min and {delta_sec:.1f} s")
    return state, outputs, exp_dir,current_sim_state

if __name__ =='__main__':
    
    cfg = Config(
        n=200,

        chunk_size = 1000,
        num_chunks = 400,
        checkpoint_freq = 50,
        video_freq = 400,

        energy_decay=0.03,
        n_agents_max=(770),
        n_agents_init=200,
        time_to_die=30,
        time_above_repr = 15,
        min_energy_repr = 1.5,
        prob_factor = 0.045,
        pre_growth_step = 5000,
        mutation_var = 0.01,
        starting_energy= 1,
        agent_view = 5,
        prob_init_resources=0.05,
    )
    
    

    # Sanity check : 
    a = cfg.starting_energy - cfg.energy_decay * cfg.time_above_repr
    b = cfg.min_energy_repr
    assert(  b > a)
    


    seed = 5
    key = random.PRNGKey(seed)
    
    print(jax.devices())

    #Recherche des paramètres
    print("Démarrage de la recherche de paramètres avec Optuna...")
    best_cfg, study = run_optuna_search(cfg, key, n_trials=40)
    
    # Lancement de la vraie simulation avec la meilleure config trouvée
    print("\nLancement de la simulation finale avec les meilleurs paramètres...")
    best_cfg._replace(video_freq = 150)
    state_final, output, exp_dir, _ = launch_simulation_chunked(
        key, best_cfg, n_video_workers=1
    )
    
    
    # Classic simulation : 
    # resume_exp = 'exp/2026-05-29_10-30-59'
    # chunk_id=50
    state_final, output, exp_dir,_ = launch_simulation_chunked(key,cfg,n_video_workers = 1)
