
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"
from jax import jit, vmap
from jax import random
import jax.numpy as jnp
import jax
import os
import json
from datetime import datetime

from simulation.one_simulation import simulation_resources_scan_agents_dying_nn, sim_scan_agents_dying_jit_nn
from simulation.utils import plot_several_sim_seeds, plot_evolution
from simulation.data_class import Config



def run_simulations(keys,cfg,kernel):
    state_final_seeds, output_seeds = vmap(simulation_resources_scan_agents_dying_nn,in_axes=(0,None,None))(keys,cfg,kernel)
    return state_final_seeds,output_seeds

runs_simulation_jit = jit(run_simulations,static_argnums=[1])

def launch_simulation_vmap(key, cfg, kernel,n_sims):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("exp", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    subkeys = random.split(key, n_sims)  

    cfg_dict = cfg._asdict()
    cfg_dict["seeds"] = [int(k[0]) for k in subkeys]  
    cfg_dict["seeds_full"] = [k.tolist() for k in subkeys]  

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    state_final_seeds, output_seeds = runs_simulation_jit(subkeys, cfg, kernel)  

    plot_several_sim_seeds(output_seeds, cfg, exp_dir)
    
    return state_final_seeds, output_seeds, exp_dir  

def launch_simulation(key, cfg, kernel, n_sims):
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("exp", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    key, *subkeys = random.split(key, n_sims + 1)

    cfg_dict = cfg._asdict()
    cfg_dict["seeds"] = [int(k[0]) for k in subkeys] 
    cfg_dict["seeds_full"] = [k.tolist() for k in subkeys]  

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    for i, subkey in enumerate(subkeys):
        state_final, output = simulation_resources_scan_agents_dying_nn(subkey, cfg, kernel)
        plot_evolution(output, exp_dir, name_fig=f'seed_{i}')
        print(f'Finish simulation {i}')

    return state_final, output, exp_dir

if __name__ =='__main__':
    
    cfg = Config(
        n=20,
        generations=20,
        prob_init_resources=0.01,
        energy_decay=0.03,
        n_agents_max=500,
        n_agents_init=50,
        time_to_die=25,
        time_above_repr = 15,
        min_energy_repr = 1.5,
        prob_factor = 0.8,
        pre_growth_step = 400,
        mutation_var = 0.02,
        starting_energy= 1,
        agent_view = 3
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
    state_final, output, exp_dir = launch_simulation(key,cfg,kernel,2)
