import graphviz
import matplotlib.pyplot as plt  # used by plot_full_genealogy_robust for colormap
import matplotlib.colors as mcolors
import numpy as np
import os
import jax
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_evolution(pop_history, res_history, exp_dir,start_step=0):
    plot_evolution_png(pop_history, res_history, exp_dir,start_step=start_step)
    plot_evolution_html(pop_history, res_history, exp_dir,start_step=start_step)

def plot_evolution_png(pop_history, res_history, exp_dir,start_step=0):
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
    ax_evo_res.set_ylabel('Resources amount', color=color_res)
    ax_evo_res.plot(generations, res_history, color=color_res, linewidth=2, label='Resources')
    ax_evo_res.tick_params(axis='y', labelcolor=color_res)

    ax_evo_agents.set_title('Simulation dynamic')

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_evo.png'))
    plt.close()



def plot_several_sim_seeds(output_seeds,cfg,exp_dir):
    n_sims = output_seeds.grid.shape[0]
    for i in range(n_sims):
        outputs_i = jax.tree_util.tree_map(lambda x: x[i], output_seeds)
        plot_evolution_png(outputs_i,exp_dir,name_fig =f'seed_{i}')
        
            

def plot_evolution_html(pop_history, res_history, exp_dir, start_step=0):
    """Plot population and resource dynamics over time."""
    generations = np.arange(start_step, start_step + len(pop_history))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=generations, y=pop_history, name='Agents', line=dict(color='red', width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=generations, y=res_history, name='Ressources', line=dict(color='green', width=2)),
        secondary_y=True,
    )
    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='Population size', secondary_y=False,
                     title_font=dict(color='red'), tickfont=dict(color='red'))
    fig.update_yaxes(title_text='Resources amount', secondary_y=True,
                     title_font=dict(color='green'), tickfont=dict(color='green'))
    fig.update_layout(title='Simulation dynamic')

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_evo.html'))


def plot_current_config(current_grid_res, pos, alive, exp_dir, name_fig='sim'):
    plot_current_config_png(current_grid_res, pos, alive, exp_dir, name_fig)

def plot_current_config_png(current_grid_res, pos, alive, exp_dir, name_fig='sim'):
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

    ax.set_title('Configuration at step ($t=0$)')
    ax.axis("off")

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig/sim_grid/')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_sim_chunk_{name_fig}.png'))
    plt.close()

def plot_current_config_html(current_grid_res, pos, alive, exp_dir, name_fig=''):
    """Plot the initial grid configuration with agent positions."""
    grid_h, grid_w = current_grid_res.shape

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=current_grid_res,
        colorscale=[[0, 'white'], [1, '#4CAF50']],
        zmin=0, zmax=1,
        showscale=False,
    ))

    x_agents = [pos[i, 1] for i in range(len(alive)) if alive[i]]
    y_agents = [pos[i, 0] for i in range(len(alive)) if alive[i]]
    colors_agents = ['purple' if i == 1 else 'red' for i in range(len(alive)) if alive[i]]

    fig.add_trace(go.Scatter(
        x=x_agents, y=y_agents,
        mode='markers',
        marker=dict(symbol='square', color=colors_agents, size=10),
        name='Agents',
    ))

    fig.update_layout(
        title='Configuration initiale (t=0)',
        xaxis=dict(showticklabels=False, range=[-0.5, grid_w - 0.5]),
        yaxis=dict(showticklabels=False, scaleanchor='x', range=[grid_h - 0.5, -0.5]),
        width=500, height=500,
    )

    path_save_fig = os.path.join(exp_dir, 'fig/sim_grid/')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_sim_chunk_{name_fig}.html'))
    





def plot_full_genealogy_robust(outputs):
    agents = outputs.agents
    parent_ids = np.array(agents.parent_id)
    born_steps = np.array(agents.born_step)
    alive = np.array(agents.alive)
    
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


### MOVEMENT
def compute_mean_movement_chunk(outputs, n):
    """Compute mean movement magnitude (in cells) per step from a simulation chunk.

    Returns an array of shape (chunk_size - 1,) — one value per step transition.
    Handles toroidal wrap-around for a grid of size n.
    """
    positions = np.array(outputs.agents.position)  # (T, N, 2)
    alive = np.array(outputs.agents.alive)          # (T, N)

    delta = positions[1:] - positions[:-1]  # (T-1, N, 2)
    delta = np.where(delta > n // 2, delta - n, delta)
    delta = np.where(delta < -(n // 2), delta + n, delta)

    magnitude = np.sqrt(delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2)  # (T-1, N)
    alive_mask = alive[1:]  # (T-1, N)
    n_alive = alive_mask.sum(axis=1)
    mean_mov = np.where(n_alive > 0, (magnitude * alive_mask).sum(axis=1) / n_alive, 0.0)
    return mean_mov  # (T-1,)

def plot_mean_movement(mov_history, res_history,exp_dir, start_step=0, name_fig='sim'):
    plot_mean_movement_png(mov_history, exp_dir, start_step=start_step, name_fig=name_fig)
    plot_mean_movement_html(mov_history, res_history,exp_dir, start_step=start_step, name_fig=name_fig)

def plot_mean_movement_png(mov_history, exp_dir, start_step=0, name_fig='sim'):
    """Plot mean movement of the population over time."""
    steps = np.arange(start_step, start_step + len(mov_history))

    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Mean movement (cell)')
    ax.plot(steps, mov_history, color='tab:blue', linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Mean population movement')

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_movement_{name_fig}.png'))
    plt.close()

def plot_mean_movement_html(mov_history, res_history, exp_dir,
                            start_step=0, name_fig='sim'):
    """Plot mean movement, points colored by resource presence."""
    steps = np.arange(start_step, start_step + len(mov_history))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=mov_history,
        mode='markers',
        marker=dict(
            color=res_history,            # tableau -> couleur par point
            colorscale='Viridis',
            showscale=True,
            size=5,
            colorbar=dict(title='Resources'),
        ),
        name='Mean movement',
    ))
    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='Mean movement (steps)')
    fig.update_layout(title='Mean population movement')

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_movement_{name_fig}.html'))



    
## LIFETIME
def compute_lifetime_chunk(outputs, cfg):
    alive = np.array(outputs.agents.alive)
    born  = np.array(outputs.agents.born_step)
    step  = np.array(outputs.step)

    a_t, a_tp1 = alive[:-1], alive[1:]
    b_t, b_tp1 = born[:-1], born[1:]

    death_event = (a_t == 1) & ((a_tp1 == 0) | (b_tp1 != b_t))

    death_step = np.broadcast_to(step[:-1][:, None], death_event.shape)[death_event]
    ages       = (step[:-1][:, None] - b_t)[death_event]
    return ages,death_step         # deux tableaux 1D, longueur = nb de morts

def plot_lifetime_vs_step(death_steps, lifetimes, exp_dir, cfg,name_fig='sim'):
    plot_lifetime_vs_step_png(death_steps, lifetimes, exp_dir,cfg, name_fig)
    plot_lifetime_vs_step_html(death_steps, lifetimes, exp_dir,cfg, name_fig)

def plot_lifetime_vs_step_png(death_steps, lifetimes, exp_dir, cfg,name_fig='sim'):
    death_steps = np.asarray(death_steps, dtype=float)
    lifetimes   = np.asarray(lifetimes, dtype=float)

    time_to_die_without_eating = cfg.starting_energy/cfg.energy_decay + cfg.time_to_die
    
    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Death step')
    ax.set_ylabel('Age at death time (steps)')
    ax.scatter(death_steps, lifetimes, s=4, alpha=0.8,
               color='tab:green', edgecolors='none')
    ax.grid(True, alpha=0.3)
    ax.set_title("Agents age at death time")
    ax.axhline(time_to_die_without_eating, color='tab:red', linewidth=1,
               linestyle='--', label=f'Minimum age ({time_to_die_without_eating:.0f} steps)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_lifetime_scatter_{name_fig}.png'), dpi=120)
    plt.close()

def plot_lifetime_vs_step_html(death_steps, lifetimes, exp_dir,cfg, name_fig='sim'):
    death_steps = np.asarray(death_steps, dtype=float)
    lifetimes   = np.asarray(lifetimes, dtype=float)

    nbins   = 100
    edges   = np.linspace(death_steps.min(), death_steps.max(), nbins + 1)
    time_to_die_without_eating = cfg.starting_energy/cfg.energy_decay + cfg.time_to_die

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=death_steps, y=lifetimes,
        mode='markers',
        marker=dict(color='green', size=5, opacity=0.8),
        name='Death age',
    ))

    fig.add_hline(y=time_to_die_without_eating, line_dash='dash', line_color='red',
                  annotation_text=f'Minimum age ({time_to_die_without_eating:.0f} steps)',
                  annotation_position='top left')

    fig.update_xaxes(title_text='Death Step')
    fig.update_yaxes(title_text='Death age (steps)')
    fig.update_layout(title="Agents age at death time")

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_lifetime_scatter_{name_fig}.html'))
    
    
## LIFE EXPECTANCY
def compute_life_expectancy(death_steps, lifetimes, bin_width=100):
    """Espérance de vie par cohorte : regroupe par step de naissance
    (largeur bin_width) et renvoie (centres, âge moyen à la mort, effectif)."""
    death_steps = np.asarray(death_steps, dtype=float)
    lifetimes   = np.asarray(lifetimes, dtype=float)
    born        = death_steps - lifetimes

    edges      = np.arange(born.min(), born.max() + bin_width, bin_width)
    sum_age, _ = np.histogram(born, bins=edges, weights=lifetimes)   # Σ âges par bin
    count,   _ = np.histogram(born, bins=edges)                      # effectif par bin
    mean_age   = np.divide(sum_age, count,
                           out=np.full(sum_age.shape, np.nan), where=count > 0)
    centers    = 0.5 * (edges[:-1] + edges[1:])
    return centers, mean_age, count
 
def plot_life_expectancy(death_steps, lifetimes, exp_dir, bin_width=100, name_fig='sim'):
    plot_life_expectancy_png(death_steps, lifetimes, exp_dir, bin_width, name_fig)
    plot_life_expectancy_html(death_steps, lifetimes, exp_dir, bin_width, name_fig)

def plot_life_expectancy_png(death_steps, lifetimes, exp_dir, bin_width=10, name_fig='sim'):
    centers, mean_age, _ = compute_life_expectancy(death_steps, lifetimes, bin_width)

    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Born step')
    ax.set_ylabel('Life expectancy (steps)')
    ax.scatter(centers, mean_age, color='tab:purple')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Life expectancy with lifetime average over {bin_width} agents")

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_life_exp_{name_fig}.png'), dpi=120)
    plt.close()

def plot_life_expectancy_html(death_steps, lifetimes, exp_dir, bin_width=10, name_fig='sim'):
    centers, mean_age, _ = compute_life_expectancy(death_steps, lifetimes, bin_width)


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=centers, y=mean_age,
        mode='markers',
        marker=dict(color='purple', size=5, opacity=0.8),
        name='Born step',
    ))
    fig.update_xaxes(title_text='Born step')
    fig.update_yaxes(title_text='Life expectancy (steps)')
    fig.update_layout(title=f"Life expectancy with lifetime average over {bin_width} agents")

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_life_exp_{name_fig}.html'))