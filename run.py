
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"
from jax import random
import jax
from simulation.one_simulation import run_simulation_chunk
from simulation.utils.utils_sim import save_checkpoint,_video_worker,save_config,create_exp_file,load_config,load_checkpoint,outputs_to_numpy,video_payload,sec_to_minutes,shuffle_resources
from simulation.utils.plots import plot_current_config
from simulation.data_class import Config, ResourceConfig, BASE_RESOURCES, LABELS
from EcoEvoJax.source.agent import MetaRnnPolicy_bcppr
from simulation.utils. utils_sim import init_state,load_checkpoint,save_checkpoint, log_resource_shuffle
from simulation.simulation_data.core import simulation_data

import multiprocessing as mp
import time
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

import os
### Force the same implementation for random numbers in jax 
jax.config.update("jax_threefry_partitionable", True)



def _hide_gpu():
    """Initializer pour les process vidéo (spawn) : ils encodent sur CPU via
    moviepy et n'ont aucun besoin du GPU. Sans ça, chaque worker ré-importe JAX
    et préalloue ~75 % de la mémoire GPU -> OOM du process de simulation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"


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

        


    
        
def launch_simulation_chunked(key, cfg, resume_exp=None, n_video_workers=2, chunk_id= 1 ,save_dir=''):
    
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
            input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 2 +  len(cfg.resources)),
            hidden_dim=4, output_dim=4, encoder_layers=[], hidden_layers=[8]
        )
    else:
        model = MetaRnnPolicy_bcppr(
            input_dim=((cfg.agent_view * 2 + 1), (cfg.agent_view * 2 + 1), 2 +  len(cfg.resources)),
            hidden_dim=4, output_dim=4, encoder_layers=[], hidden_layers=[8]
        )
        key, *subkeys = random.split(key, num_chunks_exp + 1)
        key, subkey_state = jax.random.split(key)
        state = init_state(subkey_state, cfg, model)
        start_chunk = 1
        

    save_config(cfg,subkeys, exp_dir)
    start_step = start_chunk * cfg.chunk_size

    sim_data = simulation_data(cfg=cfg,start_step=start_step, start_chunk =start_chunk)
    sim_data.register_founders(state,model)
    
    
    n_types = len(cfg.resources)
    initial_grid_res = state.grid[:n_types, :, :]       
    chunks_survived = start_chunk
    
    key,subkey_lab = random.split(key) #same lab env for every test
    subkey_lab,subkey_env_lab = random.split(subkey_lab) #same lab env for every test
    
    plot_current_config(initial_grid_res,state.grid[-1, :, :] ,state.agents.position, state.agents.alive, exp_dir , cfg.resources,name_fig=f'init')

    pending_futures = {}  # future -> chunk_idx
    ctx = mp.get_context('spawn')
    



    with ProcessPoolExecutor(max_workers=n_video_workers,mp_context=ctx,initializer=_hide_gpu) as executor:
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
            subkey = subkeys[chunk_idx-1]

            keys_chunk = jax.random.split(subkey, cfg.chunk_size)
            
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx} START  @ {time.strftime('%H:%M:%S')}")
            state, outputs = run_simulation_chunk(state, model, keys_chunk, cfg)
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx} DONE   @ {time.strftime('%H:%M:%S')}")
            
  

            sim_data.update_data_with_chunk(outputs,data_dir,chunk_idx)

            if (chunk_idx) % 100  == 0 or  (chunk_idx) == 10:
                sim_data.plot(state,exp_dir)
                sim_data.compute_R0_and_plot(state,state.step,exp_dir)
            
            current_sim_state = sim_data.check_end_condition()
            
            if current_sim_state != 'interesting':
                sim_data.plot(state,exp_dir)
                print(f"Stopping criterion : {current_sim_state}")
                
                if state.step > 100000 :
                    
                    vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{chunk_idx}.mp4")
                    outputs_np = video_payload(outputs, stride=cfg.video_stride)
                    future = executor.submit(_video_worker, outputs_np, vid_path, 20, 5, cfg.resources)
                    pending_futures[future] = chunk_idx
                    
                return state, outputs, exp_dir, current_sim_state,chunks_survived
            
            ## Genealogy and MCRA
            sim_data.update_genealogy(outputs,state,exp_dir)
            sim_data.update_mrca_and_plot(outputs,exp_dir)
            

            ## TEST AGENTS IN LAB ENV ##
            if ((chunk_idx) % cfg.pca_save_freq == 0) or (chunk_idx ==10):
                subkey_lab,subkey_sim = random.split(subkey_lab)
                def submit_video(outputs_np, vid_path, *args, label=None):
                    os.makedirs(os.path.dirname(vid_path), exist_ok=True)  # avant submit
                    fut = executor.submit(_video_worker, outputs_np, vid_path, *args)
                    pending_futures[fut] = label if label is not None else vid_path
                    return fut
                sim_data.launch_env(state = state,key_env = subkey_env_lab,subkey_sim = subkey_sim,model = model,exp_dir = exp_dir,n=50,submit_video=submit_video)

            chunks_survived+=1
            # --- Checkpoint (synchrone) ---
            if (chunk_idx) % cfg.checkpoint_freq == 0:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"state_chunk_{chunk_idx}.pkl")
                save_checkpoint(state, ckpt_path)

            # --- Vidéo (asynchrone) ---
            if (chunk_idx) % cfg.video_freq == 0 or chunk_idx==start_chunk:
                vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{chunk_idx}.mp4")
                outputs_np = video_payload(outputs, stride=cfg.video_stride)
                future = executor.submit(_video_worker, outputs_np, vid_path, 20, 5, cfg.resources)
                pending_futures[future] = chunk_idx


            ## SHUFFLE RESOURCES ##
            if (chunk_idx) % cfg.cycle_period == 0:
                new_resources = shuffle_resources(BASE_RESOURCES, subkey)

                old_resources = cfg.resources                    # (2) état COURANT, pas BASE
                new_resources = shuffle_resources(BASE_RESOURCES, subkey)

                print("----------------- Change config at step : ---------------", state.step)
                for k, (old, new) in enumerate(zip(old_resources, new_resources)):
                    flag = "" if old.id == new.id else "  <-- changé"
                    print(f"  canal {k} : {LABELS[old.id]:>7} -> {LABELS[new.id]:<7}{flag}")

                log_resource_shuffle(exp_dir, chunk_idx, int(state.step),
                                     old_resources, new_resources)

                cfg = cfg._replace(resources=new_resources)
                sim_data.cfg = cfg
                
        print("Simulation terminée. Attente des vidéos en cours...")
        for f in as_completed(pending_futures):
            cidx = pending_futures[f]
            try:
                print(f"  [video] chunk {cidx} finalisé : {f.result()}")
            except Exception as e:
                print(f"  [video] ERREUR chunk {cidx} : {e}")


    sim_data.save_mrca_sim(data_dir)
    delta_sim = time.time()-start_time_sim
    delta_min,delta_sec = sec_to_minutes(delta_sim)
    print(f"Time to compute the simulation : {delta_min} min and {delta_sec:.1f} s")
    return state, outputs, exp_dir,current_sim_state,chunks_survived



if __name__ =='__main__':
    
    cfg = Config(
            grid_length=200,

            ### Simulation computation :
            chunk_size = 1000,
            num_chunks = 1600,
            checkpoint_freq = 50,
            video_freq = 50,
            pca_save_freq=50,

            ### AGENTS : 
            # Agents number : 
            n_agents_max=2000,
            n_agents_init=20,
            agent_view = 5,
            temperature=1/40,
            
            #Physiologie
            energy_decay=0.07/8,
            factor_energy_decay_not_moving = 0.3,
            energy_max = 8.0,
            
            time_to_die=100*4,
            time_above_repr = 80*4,
            min_energy_repr = 6.,
            starting_energy= 1.5,
            
            # Mutation parameters
            mutation_var = 0.02,
            param_mutate = 0.99,


            # INIT RESOURCES MAP
            pre_growth_step = 500,
            random_pos_offspring = False,
            dumb_agent = False,
            letal_wall=False,
            cycle_period = 100,
            # ~84% du log, jamais lu hors du lab (qui le reactive de son cote)
            log_obs = False,
        )
    
    
    
    

    # Sanity check : 
    a = cfg.starting_energy - cfg.energy_decay * cfg.time_above_repr
    b = cfg.min_energy_repr
    assert(  b > a)
    


    seed = 105
    key = random.PRNGKey(seed)
    print(jax.devices())
    
    # resume_exp = "exp/2026-08-07/2026-08-07_17-45-30"
    # cfg,_ = load_config(resume_exp)
    
    state_final, output, exp_dir,_,_ = launch_simulation_chunked(key,cfg,n_video_workers = 4)
