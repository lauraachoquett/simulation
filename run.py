
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"
from jax import random
import jax.numpy as jnp
import jax
import os
import numpy as np
from simulation.one_simulation import run_simulation_chunk
from simulation.utils import plot_evolution,plot_initial_config,save_checkpoint,_video_worker,save_config,create_exp_file,load_config,load_checkpoint,outputs_to_numpy,sec_to_minutes
from simulation.data_class import Config
from EcoEvoJax.source.agent import MetaRnnPolicy_bcppr
from simulation.utils import init_state,load_checkpoint,save_checkpoint
import multiprocessing as mp
import time


from concurrent.futures import ProcessPoolExecutor, as_completed

    
def classify_outcome(pop_full, res_full, cfg):
    """
    Returns 'extinction', 'overpopulation', 'depletion', or 'interesting'.
    """
    n_chunks = 15
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


def make_harder(cfg, factor=1.3):
    """Rend la survie plus difficile : moins de ressources, plus de dépense énergétique."""
    return cfg._replace(
        prob_factor=max(cfg.prob_factor / factor, 1e-3),
        energy_decay=min(cfg.energy_decay * factor, 0.5),
    )


def make_easier(cfg, factor=1.3):
    """Rend la survie plus facile : plus de ressources, moins de dépense énergétique."""
    return cfg._replace(
        prob_factor=min(cfg.prob_factor * factor, 5.0),
        energy_decay=max(cfg.energy_decay / factor, 1e-3),
    )


def parameter_search(key, cfg, n_trials=10, n_video_workers=1):
    rng = np.random.default_rng(int(jax.random.bits(key)))
    for trial in range(n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} | prob_factor={cfg.prob_factor:.4f} | energy_decay={cfg.energy_decay:.4f} ===")

        key, subkey = jax.random.split(key)
        _, _, exp_dir, outcome = launch_simulation_chunked(subkey, cfg, n_video_workers=n_video_workers)

        print(f"  → Outcome : {outcome} | Expérience : {exp_dir}")

        if outcome == 'interesting':
            print("Paramètres intéressants trouvés !")
            return cfg, exp_dir

        factor = rng.uniform(1.1, 1.7)
        cfg = make_easier(cfg, factor) if outcome == ('extinction' or 'depletion') else make_harder(cfg, factor)

    print("Recherche terminée sans trouver de dynamique intéressante.")
    return cfg, None
    
    
def launch_simulation_chunked(key, cfg, resume_exp=None, n_video_workers=2, chunk_id=0):

    start_time_sim = time.time()
    exp_dir = create_exp_file()
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
    initial_grid = state.grid[0, :, :]
    
    plot_initial_config(initial_grid, state.agents.position, state.agents.alive, exp_dir)

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
        num_chunks = 100,
        checkpoint_freq = 10,
        video_freq = 50,

        energy_decay=0.03,
        n_agents_max=(500),
        n_agents_init=200,
        time_to_die=30,
        time_above_repr = 15,
        min_energy_repr = 1.5,
        prob_factor = 0.05,
        pre_growth_step = 5000,
        mutation_var = 0.02,
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

    # cfg, exp_dir = parameter_search(key, cfg, n_trials=20, n_video_workers=1)
    
    resume_exp = 'exp/2026-05-29_10-30-59'
    chunk_id=50
    state_final, output, exp_dir,_ = launch_simulation_chunked(key,cfg,n_video_workers = 1,resume_exp=resume_exp,chunk_id=chunk_id)
