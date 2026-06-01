
import os
import numpy as np
import matplotlib.colors as mcolors
import jax
from jax import random
import jax.numpy as jnp

from simulation.data_class import AgentState,SimState
from simulation.agent_mov import get_obs_vector
from simulation.update_env import resources_growth
from simulation.data_class import Config

import numpy as np
import graphviz
import matplotlib.pyplot as plt
from simulation.utils_video import save_chunk_video
import json
from datetime import datetime
import time
import pickle

    
def plot_evolution(pop_history, res_history, exp_dir,start_step=0, name_fig=''):
    """Plot population and resource dynamics over time."""
    generations = np.arange(start_step,start_step+len(pop_history))

    fig, ax_evo_agents = plt.subplots(figsize=(12, 6))

    color_agents = 'tab:red'
    ax_evo_agents.set_xlabel('Steps')
    ax_evo_agents.set_ylabel('Population size', color=color_agents)
    ax_evo_agents.plot(generations, pop_history, color=color_agents, linewidth=2, label='Agents')
    ax_evo_agents.tick_params(axis='y', labelcolor=color_agents)
    ax_evo_agents.grid(True, alpha=0.3)

    ax_evo_res = ax_evo_agents.twinx()
    color_res = 'tab:green'
    ax_evo_res.set_ylabel('Quantité de ressources', color=color_res)
    ax_evo_res.plot(generations, res_history, color=color_res, linewidth=2, label='Ressources')
    ax_evo_res.tick_params(axis='y', labelcolor=color_res)

    ax_evo_agents.set_title('Dynamique de la simulation')

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_evo_{name_fig}.png'))
    plt.close()


def plot_current_config(current_grid_res, pos, alive, exp_dir, name_fig=''):
    """Plot the initial grid configuration with agent positions."""
    fig, ax = plt.subplots(figsize=(5, 5))

    cmap_res = mcolors.ListedColormap(["white", "#4CAF50"])
    ax.imshow(current_grid_res, cmap=cmap_res, vmin=0, vmax=1, interpolation="nearest")

    fig.canvas.draw()
    bbox = ax.get_window_extent()
    grid_h, grid_w = current_grid_res.shape
    cell_px = min(bbox.width / grid_w, bbox.height / grid_h)
    cell_pts = cell_px * 72 / fig.dpi

    for i in range(len(alive)):
        if alive[i]:
            ax.plot(pos[i, 1], pos[i, 0], 's',
                    color="purple" if i == 1 else "red",
                    markersize=cell_pts)

    ax.set_title('Configuration initiale ($t=0$)')
    ax.axis("off")

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig/sim_grid/')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_sim_chunk_{name_fig}.png'))
    plt.close()
    
def plot_several_sim_seeds(output_seeds,cfg,exp_dir):
    n_sims = output_seeds.grid.shape[0]
    for i in range(n_sims):
        outputs_i = jax.tree_util.tree_map(lambda x: x[i], output_seeds)
        plot_evolution(outputs_i,exp_dir,name_fig =f'seed_{i}')
        
        
def init_state(key, cfg, model):
    # 1. Grille de ressources
    key, subkey_grid = random.split(key)
    grid_resources = random.bernoulli(subkey_grid, p=cfg.prob_init_resources, shape=(cfg.n, cfg.n)).astype(jnp.int32)
    
    # 2. Préparation des agents
    key, *subkeys = random.split(key, 4)
    sk_pos, sk_or, sk_params = subkeys
    
    orientations_pool = jnp.array([0, jnp.pi/2, jnp.pi, -jnp.pi/2])
    idx_or = random.randint(sk_or, (cfg.n_agents_max,), 0, 4)
    
    # Gestion de la survie initiale
    alive_mask = jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32).at[1:cfg.n_agents_init+1].set(1)
    
    # Paramètres réseau et état RNN
    params = random.normal(sk_params, (cfg.n_agents_max, model.num_params)) / 100
    policy_states = model.reset_b(jnp.zeros(cfg.n_agents_max))
    
    # Création de l'objet AgentState
    agents = AgentState(
        position=random.randint(sk_pos, (cfg.n_agents_max, 2), 0, cfg.n),
        orientation=orientations_pool[idx_or],
        energy=jnp.ones((cfg.n_agents_max,))*cfg.starting_energy,
        time_under_min_energy=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        time_over_energy_repr=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        alive=alive_mask,
        parent_id=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        born_step=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        params=params,
        policy_states=policy_states
    )

    # 3. Grille d'occupation et observations
    grid_agents = jnp.zeros((cfg.n, cfg.n), dtype=jnp.int32)
    grid_agents = grid_agents.at[agents.position[:, 0], agents.position[:, 1]].add(agents.alive)
    grid = jnp.stack((grid_resources, grid_agents))
    pos = agents.position
    obs = get_obs_vector(grid, pos,cfg.agent_view)
    # 4. État final
    
    key, key_env = jax.random.split(key)

    init_carry = (grid_resources, key_env)

    grid_resources_grown, _ = jax.lax.fori_loop(
        0,
        cfg.pre_growth_step,
        lambda i, carry: resources_growth(carry, cfg),
        init_carry
    )
    
    grid = jnp.stack((grid_resources_grown, grid_agents))

    state = SimState(
        grid=grid,
        agents=agents,
        step=0,
        obs=obs,
        last_actions=jnp.zeros((cfg.n_agents_max, 4)),
        rewards=jnp.zeros((cfg.n_agents_max, 1))
    )
    
    return state


def plot_full_genealogy_robust(agents_state_history):
    parent_ids = np.array(agents_state_history.parent_id)
    born_steps = np.array(agents_state_history.born_step)
    alive = np.array(agents_state_history.alive)
    
    T, N = born_steps.shape
    
    nodes_by_step = {}
    edges = set()
    
    def add_node(idx, step):
        if step not in nodes_by_step:
            nodes_by_step[step] = set()
        nodes_by_step[step].add(idx)
        
    # 1. Capture des agents initiaux (t=0)
    for idx in range(1, N):
        if alive[0, idx] == 1:
            add_node(idx, born_steps[0, idx])
            
    # 2. Détection infaillible de toutes les naissances
    for t in range(1, T):
        # Une naissance = le step de naissance a changé par rapport à l'étape précédente
        changed_indices = np.where(born_steps[t] != born_steps[t-1])[0]
        
        for idx in changed_indices:
            if idx == 0:
                continue # On exclut l'agent 0 mort par défaut
                
            c_b_step = born_steps[t, idx]
            p_idx = parent_ids[t, idx]
            
            if p_idx == 0:
                continue
                
            p_b_step = born_steps[t, p_idx]
            
            # Ajouter l'enfant et le parent
            add_node(idx, c_b_step)
            add_node(p_idx, p_b_step)
            
            # Créer le lien généalogique
            edges.add((f"A{p_idx}_T{p_b_step}", f"A{idx}_T{c_b_step}"))

    # 3. Identification des survivants finaux pour le style
    survivors = set()
    last_step = T - 1
    for idx in range(1, N):
        if alive[last_step, idx] == 1:
            survivors.add((idx, born_steps[last_step, idx]))

    # 4. Tracé Graphviz
    dot = graphviz.Digraph(engine='dot', format='pdf')
    dot.attr(rankdir='LR', nodesep='0.15', ranksep='0.4')
    
    if not nodes_by_step:
        return dot
        
    max_step = max(nodes_by_step.keys())
    cmap = plt.cm.Blues_r
    norm = mcolors.Normalize(vmin=0, vmax=max_step if max_step > 0 else 1)

    for step, indices in nodes_by_step.items():
        rgba = cmap(norm(step))
        hex_color = mcolors.to_hex(rgba)
        
        with dot.subgraph() as s:
            s.attr(rank='same')
            for idx in indices:
                node_id = f"A{idx}_T{step}"
                node_label = f"A_{idx}\nStep {step}"
                
                is_survivor = (idx, step) in survivors
                border_color = '#2ca02c' if is_survivor else 'black' # Vert si survivant
                pen_width = '3' if is_survivor else '1'
                
                s.node(node_id, 
                       label=node_label, 
                       style='filled,rounded', 
                       fillcolor=hex_color, 
                       color=border_color,
                       penwidth=pen_width,
                       shape='box', 
                       fontsize='9', 
                       fontcolor='white' if step < max_step/2 else 'black')

    for parent, child in edges:
        dot.edge(parent, child, arrowsize='0.6')

    dot.render('arbre_genealogique_complet', view=True)
    return dot



def save_checkpoint(state, filepath):
    """Sauvegarde l'état de la simulation sur le disque."""
    # Convertit le PyTree JAX en PyTree NumPy
    state_np = jax.tree_util.tree_map(np.asarray, state)
    with open(filepath, 'wb') as f:
        pickle.dump(state_np, f)

def load_checkpoint(resume_exp,chunk_id):
    """Charge l'état de la simulation depuis le disque."""
    path = os.path.join(resume_exp,f'checkpoints/state_chunk_{chunk_id}.pkl')
    with open(path, 'rb') as f:
        state_np = pickle.load(f)
    # Reconvertit le PyTree NumPy en PyTree JAX
    return jax.tree_util.tree_map(jnp.asarray, state_np)



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


def create_exp_file(dir):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
    return exp_dir

def save_config(cfg,subkeys,exp_dir):
    cfg_dict = cfg._asdict()
    cfg_dict["seeds"] = [int(k[0]) for k in subkeys]
    cfg_dict["seeds_full"] = [k.tolist() for k in subkeys]
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

def load_config(resume_exp):
    with open(os.path.join(resume_exp, 'config.json'), 'r') as f:
        cfg_dict = json.load(f)
    seeds_full = [jnp.array(s, dtype=jnp.uint32) for s in cfg_dict.pop("seeds_full")]
    cfg_dict.pop("seeds", None)
    return Config(**cfg_dict), seeds_full

def print_params(params, prefix="", total=0):
    for name, value in params.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if hasattr(value, 'items'):   # duck typing — FrozenDict et dict
            total = print_params(value, prefix=full_name, total=total)
        else:
            n_p = int(np.prod(value.shape))
            total += n_p
            print(f"{full_name:<55} {str(value.shape):<20} {n_p:>10,}")
    return total