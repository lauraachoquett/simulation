
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"
from jax import random
import jax.numpy as jnp
import jax
import os
import json
from datetime import datetime
import numpy as np
from simulation.one_simulation import run_simulation_chunk
from simulation.utils import plot_evolution,plot_initial_config
from simulation.data_class import Config
from EcoEvoJax.source.agent import MetaRnnPolicy_bcppr
from simulation.utils import init_state,load_checkpoint,save_checkpoint
from simulation.utils_video import save_chunk_video
import multiprocessing as mp
import time


from concurrent.futures import ProcessPoolExecutor, as_completed


def sec_to_minutes(secondes):
    minutes, secondes_restantes = divmod(secondes, 60)
    return minutes, secondes_restantes

# --- Sérialiseur : convertit outputs JAX → numpy avant envoi inter-process ---
def outputs_to_numpy(outputs):
    """
    Descend récursivement dans le pytree et convertit chaque feuille en np.array.
    À adapter selon la structure de ton SimOutputs.
    """
    import jax
    return jax.tree_util.tree_map(np.array, outputs)


# --- Wrapper picklable pour le worker ---
def _video_worker(outputs_np, vid_path, fps, scale):
    t0 = time.time()
    print(f"  [video | PID {os.getpid()}] START  {vid_path}  @ {time.strftime('%H:%M:%S')}")
    save_chunk_video(outputs_np, vid_path, fps=fps, scale=scale)
    print(f"  [video | PID {os.getpid()}] DONE   {vid_path}  ({time.time()-t0:.2f}s)")
    return vid_path


def launch_simulation_chunked(key, cfg, resume_checkpoint=None, n_video_workers=2):
    start_time_sim = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("exp", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)

    key, *subkeys = random.split(key, cfg.num_chunks + 1)
    cfg_dict = cfg._asdict()
    cfg_dict["seeds"] = [int(k[0]) for k in subkeys]
    cfg_dict["seeds_full"] = [k.tolist() for k in subkeys]
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    model = MetaRnnPolicy_bcppr(
        input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 2),
        hidden_dim=4, output_dim=4, encoder_layers=[], hidden_layers=[8]
    )

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Reprise depuis {resume_checkpoint}")
        state = load_checkpoint(resume_checkpoint)
    else:
        key, subkey = jax.random.split(key)
        state = init_state(subkey, cfg, model)

    pop_history = []
    res_history = []
    initial_grid = state.grid[0, :, :]
    plot_initial_config(initial_grid, state.agents.position, state.agents.alive, exp_dir)

    pending_futures = {}  # future -> chunk_idx
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_video_workers,mp_context=ctx) as executor:
        for chunk_idx in range(cfg.num_chunks):

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
                exp_dir
            )

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
    return state, outputs, exp_dir

if __name__ =='__main__':
    
    cfg = Config(
        n=200,
        
        chunk_size = 1000,       # Nombre de steps par itération JIT
        num_chunks = 50,       # Nombre total de chunks (1000 * 1000 = 1M steps)
        checkpoint_freq = 50,    # Sauvegarde de l'état tous les 10 chunks
        video_freq = 50,
        
        energy_decay=0.035,
        n_agents_max=1000,
        n_agents_init=200,
        time_to_die=25,
        time_above_repr = 15,
        min_energy_repr = 1.5,
        prob_factor = 0.2,
        pre_growth_step = 100,
        mutation_var = 0.02,
        starting_energy= 1,
        agent_view = 3,
        prob_init_resources=0.01,
    )


    kernel = jnp.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=jnp.float32)


    seed = 5
    key = random.PRNGKey(seed)
    
    print(jax.devices())
    resume_checkpoint = 'exp/2026-05-27_16-52-31/checkpoints/state_chunk_2.pkl'
    state_final, output, exp_dir = launch_simulation_chunked(key,cfg,n_video_workers = 3)
