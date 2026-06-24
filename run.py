
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"
from jax import random
import jax.numpy as jnp
import jax
import numpy as np
from simulation.one_simulation import run_simulation_chunk
from simulation.utils import save_checkpoint,_video_worker,save_config,create_exp_file,load_config,load_checkpoint,outputs_to_numpy,sec_to_minutes
from simulation.plots import plot_current_config
from simulation.data_class import Config
from EcoEvoJax.source.agent import MetaRnnPolicy_bcppr
from simulation.utils import init_state,load_checkpoint,save_checkpoint
from simulation.simulation_data import simulation_data

import multiprocessing as mp
import time
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

import os
### Force the same implementation for random numbers in jax 
jax.config.update("jax_threefry_partitionable", True)



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

        


    
        
def launch_simulation_chunked(key, cfg, resume_exp=None, n_video_workers=2, chunk_id=0,save_dir=''):
    
    start_time_sim = time.time()
    if save_dir == '':
        now = datetime.now()
        save_dir = os.path.join("exp", now.strftime("%Y-%m-%d"))
        
    exp_dir,data_dir = create_exp_file(save_dir)
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
            input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 3),
            hidden_dim=2, output_dim=4, encoder_layers=[], hidden_layers=[4]
        )
    else:
        model = MetaRnnPolicy_bcppr(
            input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 3),
            hidden_dim=4, output_dim=4, encoder_layers=[], hidden_layers=[8]
        )
        key, *subkeys = random.split(key, num_chunks_exp + 1)
        key, subkey_state = jax.random.split(key)
        state = init_state(subkey_state, cfg, model)
        start_chunk = 0
        
    
    save_config(cfg,subkeys, exp_dir)
    start_step = start_chunk * cfg.chunk_size
    

    sim_data = simulation_data(cfg=cfg,start_step=start_step)



    initial_grid_res = state.grid[0, :, :]
    chunks_survived = 0
    
    plot_current_config(initial_grid_res,state.grid[-1, :, :] ,state.agents.position, state.agents.alive, exp_dir,name_fig=f'init')

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
            subkey,subkey_lab = random.split(subkey)
            keys_chunk = jax.random.split(subkey, cfg.chunk_size)
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx+1} START  @ {time.strftime('%H:%M:%S')}")
            state, outputs = run_simulation_chunk(state, model, keys_chunk, cfg)
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx+1} DONE   @ {time.strftime('%H:%M:%S')}")
            
  

            sim_data.update_data_with_chunk(outputs,data_dir)

            if (chunk_idx + 1) % 100  == 0 or  (chunk_idx + 1) == 10:
                sim_data.plot(state,exp_dir)
            
            current_sim_state = sim_data.check_end_condition()
            
            if current_sim_state != 'interesting':
                sim_data.plot(state,exp_dir)
                print(f"Stopping criterion : {current_sim_state}")
                vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{chunk_idx+1}.mp4")
                outputs_np = outputs_to_numpy(outputs)
                future = executor.submit(_video_worker, outputs_np, vid_path, 20, 5)
                pending_futures[future] = chunk_idx + 1
                return state, outputs, exp_dir, current_sim_state,chunks_survived
            
            ## Genealogy and MCRA
            sim_data.update_genealogy(outputs,exp_dir)
            sim_data.update_mrca(outputs,exp_dir)
            
            
            if ((chunk_idx + 1) % cfg.pca == 0) or (chunk_idx ==10):
                sim_data.launch_env(outputs,subkey_lab,model,exp_dir,n=10)
                sim_data.plot_pca_and_mrca_and_lab_env(outputs,data_dir,exp_dir)

            # Stop simulation ? 
            chunks_survived+=1
            # --- Checkpoint (synchrone) ---
            if (chunk_idx + 1) % cfg.checkpoint_freq == 0:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"state_chunk_{chunk_idx+1}.pkl")
                save_checkpoint(state, ckpt_path)

            # --- Vidéo (asynchrone) ---
            if (chunk_idx + 1) % cfg.video_freq == 0 or chunk_idx==start_chunk:
                vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{chunk_idx+1}.mp4")
                outputs_np = outputs_to_numpy(outputs)
                future = executor.submit(_video_worker, outputs_np, vid_path, 20, 5)
                pending_futures[future] = chunk_idx + 1

        print("Simulation terminée. Attente des vidéos en cours...")
        for f in as_completed(pending_futures):
            cidx = pending_futures[f]
            try:
                print(f"  [video] chunk {cidx} finalisé : {f.result()}")
            except Exception as e:
                print(f"  [video] ERREUR chunk {cidx} : {e}")


    sim_data.save_mrca_end_sim(data_dir)
    delta_sim = time.time()-start_time_sim
    delta_min,delta_sec = sec_to_minutes(delta_sim)
    print(f"Time to compute the simulation : {delta_min} min and {delta_sec:.1f} s")
    return state, outputs, exp_dir,current_sim_state,chunks_survived



if __name__ =='__main__':
    
    
    cfg = Config(
            n=300,

            ### Simulation computation :
            chunk_size = 1000,
            num_chunks = 1000,
            checkpoint_freq = 50,
            video_freq = 50,
            pca=100,

            ### AGENTS : 
            # Agents number : 
            n_agents_max=1400,
            n_agents_init=20,
            agent_view = 5,
            temperature=20,
            
            #Physiologie
            energy_decay=0.07,
            factor_energy_decay_not_moving = 0.2,
            
            time_to_die=70,
            time_above_repr = 60,
            min_energy_repr = 2.5,
            starting_energy= 0.5,
            
            # Mutation parameters
            mutation_var = 0.02,
            param_mutate = 0.9,
            
            # RESOURCES : 
            prob_factor = 0.065,
            pop_res_prob = 0.00000001,
            
            # INIT RESOURCES MAP
            pre_growth_step = 3000,
            init_number_of_resources=2,
            random_pos_offspring = True,
            dumb_agent= False,
        )
    


    

    # Sanity check : 
    a = cfg.starting_energy - cfg.energy_decay * cfg.time_above_repr
    b = cfg.min_energy_repr
    assert(  b > a)
    


    seed = 4
    key = random.PRNGKey(seed)
    
    print(jax.devices())
    
    state_final, output, exp_dir,_,_ = launch_simulation_chunked(key,cfg,n_video_workers = 4)


    
    
    # Classic simulation : 
    
    # cfg,_ = load_config(resume_exp)
    # cfg = cfg._replace(n_agents_max=1000,n_agents_init=200,pre_growth_step=2000)