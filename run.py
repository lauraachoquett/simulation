
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"
from jax import random
import jax.numpy as jnp
import jax
import numpy as np
from simulation.one_simulation import run_simulation_chunk
from simulation.utils import save_checkpoint,_video_worker,save_config,create_exp_file,load_config,load_checkpoint,outputs_to_numpy,sec_to_minutes
from simulation.plots import plot_evolution,plot_current_config, compute_mean_movement_chunk, compute_resources_consumed_chunk,plot_mean_movement,plot_resources_consumed
from simulation.data_class import Config
from EcoEvoJax.source.agent import MetaRnnPolicy_bcppr
from simulation.utils import init_state,load_checkpoint,save_checkpoint
import multiprocessing as mp
import time
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import optuna
import logging
import json
import os


def save_script(exp_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Liste des fichiers à sauvegarder
    files_to_save = ["run.py", "update_env.py", "one_simulation.py",'agent_mov.py']

    for file_name in files_to_save:
        source_path = os.path.join(current_dir, file_name)
        destination_path = os.path.join(exp_dir, file_name)

        # Vérifie si le fichier existe avant de le copier
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
        else:
            print(
                f"Attention : Le fichier {source_path} n'a pas pu être trouvé."
            )
            
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
    n_chunks = 30
    if pop_full[-1] == 0:
        return 'extinction'
    
    last = pop_full[-n_chunks*cfg.chunk_size:]
    if len(last) == n_chunks*cfg.chunk_size and (last > 0.95 * cfg.n_agents_max).all():
        return 'overpopulation'
    
    if res_full[-1]==0:
        return 'depletion'
    
    last_res = res_full[-n_chunks*cfg.chunk_size:]
    if len(last_res) == n_chunks*cfg.chunk_size and (last_res > 0.8 * cfg.n ** 2).all():
        return 'easy'
    
    return 'interesting'

def objective(trial, base_cfg, base_key,current_dir):
    prob_factor = trial.suggest_float("prob_factor", 0.01, 1.0, log=True)
    energy_decay = trial.suggest_float("energy_decay", 0.001, 0.1, log=True)
    
    trial_cfg = base_cfg._replace(
        prob_factor=prob_factor,
        energy_decay=energy_decay,
    )
    
    a = trial_cfg.starting_energy - trial_cfg.energy_decay * trial_cfg.time_above_repr
    b = trial_cfg.min_energy_repr
    
    if b <= a or trial_cfg.time_to_die <= trial_cfg.time_above_repr:
        raise optuna.TrialPruned("Contraintes physiques ou temporelles non respectées.")

    trial_key, _ = jax.random.split(base_key)
    
    
    try:
        state, outputs, exp_dir, outcome,chunks_survived = launch_simulation_chunked(
            trial_key, trial_cfg, resume_exp=None, n_video_workers=1, dir=current_dir
        )
    except Exception as e:
        raise optuna.TrialPruned(f"Erreur d'exécution: {e}")

    
    pop_full = outputs_to_numpy(outputs.agents.alive).sum(axis=1)
    
    final_pop = pop_full[-1] if len(pop_full) > 0 else 0
    pop_ratio = final_pop / trial_cfg.n_agents_max
    
    score = chunks_survived + pop_ratio
    
    if outcome == 'overpopulation':
        score -= 1.0
    if outcome == 'easy':
        score -= 2.0
        
    trial.set_user_attr('exp_dir', exp_dir)
    trial.set_user_attr('chunks_survived', int(chunks_survived))
    trial.set_user_attr('final_pop', int(final_pop))

    return score

def run_optuna_search(cfg, key, n_trials=50):
    """
    Configure et lance l'étude d'optimisation.
    """
    optuna.logging.get_logger("optuna").setLevel(logging.INFO)
    
    study = optuna.create_study(
        study_name="ecoevo_hyperopt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42) 
    )
    
    # Passage des arguments fixes via une fonction lambda
    current_dir = get_next_dir(base_path="exp/", prefix="try_")
    study.optimize(lambda trial: objective(trial, cfg, key,current_dir), n_trials=n_trials)
    
    print("\n=== Recherche Optuna terminée ===")
    print(f"Meilleur score : {study.best_value}")
    print(f"Meilleurs paramètres : {study.best_params}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: t.value, reverse=True)
    results = [
        {
            'rank': i + 1,
            'score': t.value,
            'exp_dir': t.user_attrs.get('exp_dir'),
            'chunks_survived': t.user_attrs.get('chunks_survived'),
            'final_pop': t.user_attrs.get('final_pop'),
            'params': t.params,
        }
        for i, t in enumerate(ranked)
    ]
    results_path = os.path.join(current_dir, "trials_ranked.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Résultats triés sauvegardés dans : {results_path}")

    # Retourne la configuration optimale
    best_cfg = cfg._replace(**study.best_params)
    return best_cfg, study
    
    
def launch_simulation_chunked(key, cfg, resume_exp=None, n_video_workers=2, chunk_id=0,dir=''):
    
    start_time_sim = time.time()
    if dir == '':
        now = datetime.now()
        dir = os.path.join("exp", now.strftime("%Y-%m-%d"))
    exp_dir,data_dir = create_exp_file(dir)
    save_script(exp_dir)
    
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
            hidden_dim=2, output_dim=4, encoder_layers=[], hidden_layers=[4]
        )
    else:
        model = MetaRnnPolicy_bcppr(
            input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 2),
            hidden_dim=2, output_dim=4, encoder_layers=[], hidden_layers=[4]
        )
        key, *subkeys = random.split(key, num_chunks_exp + 1)
        key, subkey = jax.random.split(key)
        state = init_state(subkey, cfg, model)
        start_chunk = 0
        
    save_config(cfg, subkeys, exp_dir)
    start_step = start_chunk * cfg.chunk_size
    

    pop_history = []
    res_history = []
    mov_history = []
    consumed_history = []

    initial_grid_res = state.grid[0, :, :]
    chunks_survived = 0
    
    plot_current_config(initial_grid_res, state.agents.position, state.agents.alive, exp_dir,name_fig=f'init')

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
            pop_chunk      = np.array(outputs.agents.alive.sum(axis=1))
            res_chunk      = np.array(outputs.grid[:, 0, :, :].sum(axis=(1, 2)))
            mov_chunk      = compute_mean_movement_chunk(outputs, cfg.n)
            consumed_chunk = compute_resources_consumed_chunk(outputs)

            pop_history.append(pop_chunk)
            res_history.append(res_chunk)
            mov_history.append(mov_chunk)
            consumed_history.append(consumed_chunk)

            # --- Sauvegarde des données du chunk ---
            np.savez(
                os.path.join(data_dir, f"chunk_{chunk_idx+1:05d}.npz"),
                population    = pop_chunk,
                resources     = res_chunk,
                mean_movement = mov_chunk,
                consumed      = consumed_chunk,
            )

            plot_evolution(
                np.concatenate(pop_history, axis=0),
                np.concatenate(res_history, axis=0),
                exp_dir,
                start_step
            )
            plot_mean_movement(np.concatenate(mov_history), exp_dir, start_step)
            plot_resources_consumed(np.concatenate(consumed_history), exp_dir, start_step)
            pop_full = np.concatenate(pop_history)
            res_full = np.concatenate(res_history)
            
            plot_current_config(state.grid[0, :, :], state.agents.position, state.agents.alive, exp_dir,name_fig=f'{chunk_idx}')
            
            current_sim_state = classify_outcome(pop_full, res_full, cfg)
            if current_sim_state != 'interesting':
                print(f"Stopping criterion : {current_sim_state}")
                return state, outputs, exp_dir, current_sim_state,chunks_survived

            chunks_survived+=1
            # --- Checkpoint (synchrone) ---
            if (chunk_idx + 1) % cfg.checkpoint_freq == 0:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"state_chunk_{chunk_idx+1}.pkl")
                save_checkpoint(state, ckpt_path)

            # --- Vidéo (asynchrone) ---
            if (chunk_idx + 1) % cfg.video_freq == 0 or chunk_idx==start_chunk:
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
    return state, outputs, exp_dir,current_sim_state,chunks_survived

if __name__ =='__main__':
    
    cfg = Config(
        n=50,

        ### Simulation computation :
        chunk_size = 1000,
        num_chunks = 300,
        checkpoint_freq = 50,
        video_freq = 25,
        pca=50,

        ### AGENTS : 
        # Agents number : 
        n_agents_max=1000,
        n_agents_init=50,
        
        agent_view = 5,
        
        #Physiologie
        energy_decay=0.08,
        factor_energy_decay_not_moving = 0.5,
        
        time_to_die=50,
        time_above_repr = 40,
        min_energy_repr = 1.,
        starting_energy= 1,
        
        # Mutation parameters
        mutation_var = 0.01,
        param_mutate = 0.2,
        
        # RESOURCES : 
        prob_factor = 0.095,
        
        # INIT RESOURCES MAP
        pre_growth_step = 1000,
        prob_init_resources=0.01,
    )
    
    
    

    # Sanity check : 
    a = cfg.starting_energy - cfg.energy_decay * cfg.time_above_repr
    b = cfg.min_energy_repr
    assert(  b > a)
    


    seed = 4
    key = random.PRNGKey(seed)
    
    print(jax.devices())

    state_final, output, exp_dir,_,_ = launch_simulation_chunked(key,cfg,n_video_workers = 2)

    # #Recherche des paramètres
    # print("Démarrage de la recherche de paramètres avec Optuna...")
    # best_cfg, study = run_optuna_search(cfg, key, n_trials=50)
    
    # # Lancement de la vraie simulation avec la meilleure config trouvée
    # print("\nLancement de la simulation finale avec les meilleurs paramètres...")
    # best_cfg._replace(video_freq = 150)
    # state_final, output, exp_dir, _,_ = launch_simulation_chunked(
    #     key, best_cfg, n_video_workers=1
    # )
    
    
    # Classic simulation : 
    # resume_exp = 'exp/try_7/2026-06-02_11-47-21'
    
    # cfg,_ = load_config(resume_exp)
    # cfg = cfg._replace(n_agents_max=1000,n_agents_init=200,pre_growth_step=2000)