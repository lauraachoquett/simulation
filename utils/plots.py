import graphviz
import matplotlib.pyplot as plt  # used by plot_full_genealogy_robust for colormap
import matplotlib.colors as mcolors
import numpy as np
import os
import jax
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

GROUPS = ["encoder", "lstm_input", "lstm_recurrent", "controller"]

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

from matplotlib.collections import LineCollection

def plot_phase_portrait_png(pop_history, res_history, exp_dir, start_step=0,
                            fixed_point=None):
    """Phase portrait: trajectory (resources, population) colored by time."""
    if len(pop_history)<2000:
        return
    R = np.asarray(res_history)[2000:]
    N = np.asarray(pop_history)[2000:]
    start_step=2000
    steps = np.arange(start_step, start_step + len(N))

    # Segments consécutifs pour colorer la ligne par le temps
    points = np.array([R, N]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))

    lc = LineCollection(segments, cmap='viridis', linewidth=2)
    lc.set_array(steps[:-1])           # couleur = step
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Step')

    # Début (cercle) et fin (croix)
    ax.scatter(R[0], N[0], color='black', s=60, zorder=3, label='Start')
    ax.scatter(R[-1], N[-1], color='red', marker='X', s=80, zorder=3, label='End')

    # Point fixe théorique si fourni : (R*, N*)
    if fixed_point is not None:
        ax.scatter(*fixed_point, color='blue', marker='*', s=200,
                   zorder=3, label='Fixed point')

    ax.set_xlabel('Resources amount')
    ax.set_ylabel('Population size')
    ax.set_title('Phase portrait (resources vs population)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.autoscale()                     # LineCollection ne fixe pas les limites seule

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, 'plot_phase.png'))
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


def plot_current_config(current_grid_res, grid_walls,pos, alive, exp_dir, name_fig='sim'):
    plot_current_config_png(current_grid_res, grid_walls,pos, alive, exp_dir, name_fig)

def plot_current_config_png(current_grid_res,grid_walls, pos, alive, exp_dir, name_fig='sim'):
    """Plot the initial grid configuration with agent positions."""
    fig, ax = plt.subplots(figsize=(5, 5))

    cmap_res = mcolors.ListedColormap(["white", "#4CAF50"])
    ax.imshow(current_grid_res, cmap=cmap_res, vmin=0, vmax=1, interpolation="nearest")

    walls = np.asarray(grid_walls)
    walls_masked = np.ma.masked_where(walls != 1, walls)   # masque tout sauf les murs
    cmap_walls = mcolors.ListedColormap(["black"])
    ax.imshow(walls_masked, cmap=cmap_walls, vmin=1, vmax=1, interpolation="nearest")
    
    
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    grid_h, grid_w = current_grid_res.shape
    cell_px = min(bbox.width / grid_w, bbox.height / grid_h)
    cell_pts = cell_px * 72 / fig.dpi

    for i in range(len(alive)):
        if alive[i]:
            ax.plot(pos[i, 1], pos[i, 0], 's',
                    color="red",
                    markersize=cell_pts)

    ax.set_title('Configuration')
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
    positions = np.array(outputs.position)  # (T, N, 2)
    alive = np.array(outputs.alive)          # (T, N)

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
    alive = np.array(outputs.alive)
    born  = np.array(outputs.born_step)
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
    
import glob 
import re
import json
       
def plot_lab_metrics(exp_dir):
    data_dir = os.path.join(exp_dir, "lab_data")
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*_summary.json")),
                key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1)))
    files = [f for f in files                                   # <== AJOUT
            if re.fullmatch(r"chunk_\d+_summary\.json", os.path.basename(f))]
    if not files:
        print("No summary to plot.")
        return

    S = [json.load(open(f)) for f in files]
    x = np.array([s["chunk"] for s in S])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    # Trois mesures avec moyenne ± 1σ
    specs = [
        (axes[0, 0], "duree_vie_moy",    "duree_vie_std",    "Lifespan",     "steps"),
        (axes[0, 1], "consommation_moy", "consommation_std", "Mean intake",  "/step"),
        (axes[1, 0], "mouvement_moy",    "mouvement_std",    "Mean motion",  "/step"),
    ]
    for ax, km, ks, title, unit in specs:
        m  = np.array([s[km] for s in S])
        sd = np.array([s[ks] for s in S])
        ax.plot(x, m, marker="o", color="C0", label="mean")
        ax.fill_between(x, m - sd, m + sd, alpha=0.25, color="C0", label="±1σ")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)

    # Mortalité : deux fractions empilables, sans écart-type
    ax = axes[1, 1]
    mur  = np.array([s["frac_mort_mur"]  for s in S])
    faim = np.array([s["frac_mort_faim"] for s in S])
    ax.plot(x, mur,  marker="o", color="C3", label="wall deaths")
    ax.plot(x, faim, marker="s", color="C1", label="starvation deaths")
    ax.set_title("Cause of death (fraction of tested agents)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("chunk")
    axes[0, 0].legend(loc="best")

    fig.tight_layout()
    out = os.path.join(data_dir, "lab_metrics_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


def plot_lab_exploration(exp_dir):
    """Évolution de l'EXPLORATION (env low_res) chunk après chunk.
    Deux panneaux :
      - frac_found_food : part des agents qui trouvent au moins une ressource
      - explore_time    : délai moyen (± 1σ) avant la 1re ressource, calculé sur
                          les seuls agents qui ont mangé (grandeur conditionnelle).
    """
    data_dir = os.path.join(exp_dir, "lab_data")
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*_lowres_summary.json")),
                   key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1)))
    if not files:
        print("No low_res summary to plot.")
        return

    S = [json.load(open(f)) for f in files]
    x = np.array([s["chunk"] for s in S])

    frac = np.array([s["frac_found_food"]  for s in S])
    tm   = np.array([s["explore_time_moy"] for s in S], dtype=float)
    ts   = np.array([s["explore_time_std"] for s in S], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    axes[0].plot(x, frac, marker="o", color="C2")
    axes[0].set_title("Fraction of agents that found food")
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    valid = ~np.isnan(tm)                      # chunks où au moins un agent a mangé
    axes[1].plot(x[valid], tm[valid], marker="o", color="C0", label="mean")
    axes[1].fill_between(x[valid], (tm - ts)[valid], (tm + ts)[valid],
                         alpha=0.25, color="C0", label="±1σ")
    axes[1].set_title("Time to first resource (exploration)")
    axes[1].set_ylabel("steps")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    for ax in axes:
        ax.set_xlabel("chunk")

    fig.tight_layout()
    out = os.path.join(data_dir, "lab_exploration_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


def plot_alone_vs_clones(exp_dir):
    """Évolution de l'EFFET DES PAIRS (env clones) chunk après chunk.
    Un sous-graphe par métrique : comportement de l'agent SEUL (high_res) vs
    comportement MOYEN des clones du même génome. L'écart entre les deux courbes
    = effet des pairs. Le liseré ±1σ autour de la courbe clones vient de
    delta_std (variance de l'effet apparié entre génomes)."""
    data_dir = os.path.join(exp_dir, "lab_data")
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*_alone_vs_clones.json")),
                   key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1)))
    if not files:
        print("No alone_vs_clones data to plot.")
        return

    P = [json.load(open(f)) for f in files]
    x = np.array([p["chunk"] for p in P])

    metrics = ["age", "mean_rew", "mean_speed", "energy_end", "wall_death"]
    titles  = {"age": "Lifespan (steps)", "mean_rew": "Consumption /step",
               "mean_speed": "Movement /step", "energy_end": "Final energy",
               "wall_death": "Fraction wall deaths"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()

    for ax, k in zip(axes, metrics):
        alone  = np.array([p["metrics"][k]["alone_moy"]  for p in P], dtype=float)
        clones = np.array([p["metrics"][k]["clones_moy"] for p in P], dtype=float)
        dstd   = np.array([p["metrics"][k]["delta_std"]  for p in P], dtype=float)
        ax.plot(x, alone,  marker="o", color="C0", label="alone")
        ax.plot(x, clones, marker="s", color="C1", label="clones (mean of peers)")
        ax.fill_between(x, clones - dstd, clones + dstd, alpha=0.20, color="C1")
        ax.set_title(titles[k])
        ax.grid(alpha=0.3)
        ax.set_xlabel("chunk")

    axes[0].legend(loc="best")
    axes[-1].axis("off")            # 6e case vide (5 métriques)

    fig.suptitle("Focal agent alone vs among identical clones", y=1.0)
    fig.tight_layout()
    out = os.path.join(data_dir, "lab_alone_vs_clones_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")

# =====================================================================
#  NOUVEAU — ENERGIE AU COURS DU TEMPS (les 3 labs)
#  Necessite en tete de plots.py :
#      from matplotlib.collections import LineCollection
#      from matplotlib.lines import Line2D
# =====================================================================
 
def _colored_line(ax, x, y, min_repr, e_die):
    """Trace y(x) en coloriant chaque segment selon le regime energetique."""
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ymid = 0.5 * (y[:-1] + y[1:])
    c = np.where(ymid >= min_repr, "#2e7d4f",
        np.where(ymid <  e_die,    "#c0392b", "#7f8c8d"))
    ax.add_collection(LineCollection(segs, colors=c, linewidths=1.5))
 
 
def _repro_events(e, min_repr, time_above):
    """c s'incremente tant que E>=min_repr, reset des que E repasse dessous.
       c atteint time_above -> evenement REPRO (et reset).
       E redescend avec c>0 -> evenement DROP (montee avortee)."""
    repro, drop, c = [], [], 0
    for t in range(len(e)):
        if e[t] >= min_repr:
            c += 1
            if c >= time_above:
                repro.append(t); c = 0
        else:
            if c > 0:
                drop.append(t)
            c = 0
    return np.array(repro, int), np.array(drop, int)
 
 
def _y_limits(y, y_min, y_max):
    """Cadrage : (y_min, y_max) est un PLANCHER, pas une borne dure. On voit
    toujours au moins cette fenetre (donc les deux seuils), et l'axe s'etend
    si la courbe sort."""
    pad = 0.05 * max(y.max() - y.min(), 1e-9)
    return min(y_min, y.min() - pad), max(y_max, y.max() + pad)
 
 
def plot_lab_energy(energy, alive, exp_dir, lab_dir, chunk,
                    min_energy_repr, time_above_repr, energy_to_die=0.0,
                    n_envs=10, env_title="", y_min=-1.0, y_max=4.5,
                    share_y=True):
    """Figure au format 6:4. L'axe y couvre au minimum (y_min, y_max) et
    s'agrandit si l'energie sort de cette fenetre.
    share_y=True : tous les sous-graphes d'une figure partagent le meme axe y
    (les clones sont alors comparables entre eux) ; False : un cadrage par
    agent, plus detaille mais non comparable."""
    energy = np.asarray(energy)
    alive  = np.asarray(alive)
    if energy.ndim == 4 and energy.shape[-1] == 1:
        energy = energy[..., 0]
 
    out_dir = os.path.join(exp_dir, "energy", lab_dir)
    os.makedirs(out_dir, exist_ok=True)
    B = min(n_envs, energy.shape[0])
 
    for b in range(B):
        E, A = energy[b], alive[b]
        slots = [n for n in range(A.shape[1]) if A[:, n].any()]
        if not slots:
            continue
 
        fig, axes = plt.subplots(len(slots), 1, figsize=(12, 8),   # rapport 6:4
                                 squeeze=False, sharex=True)
        axes = axes[:, 0]
 
        # trajectoires vivantes de cette figure, puis cadrage
        traj = []
        for n in slots:
            idx = np.where(A[:, n] == 1)[0]
            t0, t1 = idx[0], idx[-1]
            traj.append((n, np.arange(t0, t1 + 1), E[t0:t1 + 1, n]))
 
        if share_y:
            allv = np.concatenate([yy for _, _, yy in traj])
            lims = [_y_limits(allv, y_min, y_max)] * len(traj)
        else:
            lims = [_y_limits(yy, y_min, y_max) for _, _, yy in traj]
 
        for ax, (n, x, y), (y_lo, y_hi) in zip(axes, traj, lims):
            # bandes de fond : regime reproductif / intermediaire / letal
            ax.axhspan(min_energy_repr, y_hi,            facecolor="#e8f2ec", zorder=0)
            ax.axhspan(energy_to_die,   min_energy_repr, facecolor="#f7f7f7", zorder=0)
            ax.axhspan(y_lo,            energy_to_die,   facecolor="#fdeceb", zorder=0)
 
            _colored_line(ax, x, y, min_energy_repr, energy_to_die)
            ax.axhline(min_energy_repr, color="#2e7d4f", ls=":", lw=1, alpha=.8)
            ax.axhline(energy_to_die,   color="#c0392b", ls=":", lw=1, alpha=.8)
 
            repro, drop = _repro_events(y, min_energy_repr, time_above_repr)
            for t in repro:   # pointilles larges, vert fonce
                ax.axvline(x[t], color="#146b34", lw=2.0, alpha=.95, dashes=(6, 3))
            for t in drop:    # pointilles serres, orange
                ax.axvline(x[t], color="#c77d0a", lw=1.1, alpha=.85, dashes=(2, 2))
 
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel(f"energy (slot {n})")
            ax.grid(alpha=.25)
 
        handles = [
            Line2D([], [], color="#146b34", lw=2.0, dashes=(6, 3), label=f"reproduction reached ({time_above_repr:g} steps above)"),
            Line2D([], [], color="#c77d0a", lw=1.1, dashes=(2, 2), label="dropped below before threshold"),
        ]
        axes[-1].set_xlabel("step")
        fig.suptitle(f"{env_title} — chunk {chunk}, agent {b}", y=0.995)
        frac = 0.75 / 8.0                         # ~0.75 pouce reserve pour la legende
        fig.tight_layout(rect=[0, frac, 1, 1])
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(os.path.join(out_dir, f"chunk_{chunk}_agent_{b:02d}.png"), dpi=130)
        plt.close(fig)
    print(f"Energy plots saved: {out_dir}")
    
"""À intégrer dans simulation/plots.py
(remplace plots_metrics_weight_magnitude_distance).

`neutral` est désormais, comme `dist_list`, une liste de {groupe: (mean, std)} :
chaque bloc a sa propre référence neutre, parce que la chaîne de réduction
(moyenne par couche puis moyenne non pondérée sur les couches du groupe) ne
produit pas la même dispersion selon la taille et le nombre de couches.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


GROUPS = ["encoder", "lstm_input", "lstm_recurrent", "controller"]


def _xaxis(steps, gen_depth, x_axis):
    if x_axis == "generation":
        if gen_depth is None:
            raise ValueError("x_axis='generation' requiert gen_depth")
        return np.asarray(gen_depth, float), "générations (profondeur généalogique)"
    return np.asarray(steps, float), "step"


def _connect_origin(ax, x, y, color, alpha=0.55):
    """Raccorde (0, 0) au premier point mesuré, en pointillé léger.

    L'origine est exacte : au step 0 chaque agent est son propre ancêtre, donc
    distance nulle. Mais la mesure ne démarre qu'après coalescence, souvent
    très loin dans le run. Le segment qui l'y relie ne traverse AUCUNE donnée —
    on le distingue visuellement pour ne pas le faire passer pour une mesure.
    (En abscisse 'generation' il reste une bonne approximation, la dérive
    neutre y étant linéaire ; en abscisse 'step' il ne l'est que si le temps de
    génération est stationnaire.)
    """
    ok = np.nonzero(np.isfinite(y))[0]
    if len(ok) == 0:
        return
    j = ok[0]
    ax.plot([0.0, x[j]], [0.0, y[j]], ls=":", lw=1.0, color=color,
            alpha=alpha, zorder=1)
    ax.plot([0.0], [0.0], marker="o", ms=3, color=color, alpha=alpha, zorder=1)


def plots_metrics_weight_distance(dist_list, exp_dir, steps,
                                  neutral=None, gen_depth=None,
                                  x_axis="step", show_origin=False,
                                  share_y=True, fname="weight_distance.png"):
    """Distance à l'ancêtre commun : un panneau par bloc, observé vs neutre.

    Un panneau par groupe plutôt que huit courbes superposées — avec l'observé
    et le neutre si proches, la version empilée ne permet pas de voir lequel
    passe au-dessus de l'autre.

    Args:
        dist_list   : [{groupe: (mean, std)}] observé, un élément par chunk
        neutral     : [{groupe: (mean, std)}] neutre, même structure
        share_y     : axe y commun aux quatre panneaux. Recommandé : c'est ce
                      qui rend les blocs comparables entre eux.
        show_origin : raccorder (0, 0) au premier point. Désactivé par défaut :
                      la mesure ne démarre qu'à la coalescence, donc ce segment
                      couvrirait des dizaines de milliers de steps sans donnée.
    """
    x, xlab = _xaxis(steps, gen_depth, x_axis)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             sharex=True, sharey=share_y)

    for i, (g, ax) in enumerate(zip(GROUPS, axes.ravel())):
        c = f"C{i}"
        m = np.array([d[g][0] for d in dist_list], dtype=float)
        s = np.array([d[g][1] for d in dist_list], dtype=float)

        ax.plot(x, m, marker="o", ms=3, color=c, lw=1.4,
                label="observé", zorder=3)
        ax.fill_between(x, m - s, m + s, alpha=0.18, color=c, lw=0)
        if show_origin:
            _connect_origin(ax, x, m, c)

        if neutral is not None:
            nm = np.array([n[g][0] for n in neutral], dtype=float)
            ns = np.array([n[g][1] for n in neutral], dtype=float)
            ax.plot(x, nm, ls="--", lw=1.4, color="0.25",
                    label=r"neutre  $p\sigma^2 g$", zorder=2)
            ax.fill_between(x, nm - ns, nm + ns, color="0.25",
                            alpha=0.12, lw=0)
            if show_origin:
                _connect_origin(ax, x, nm, "0.25", alpha=0.35)

            # Fraction de points sous le neutre : >0.5 suggère une contrainte
            # sélective, <0.5 une dérive accélérée. Purement indicatif.
            ok = np.isfinite(m) & np.isfinite(nm)
            if ok.any():
                frac = float(np.mean(m[ok] < nm[ok]))
                ax.text(0.97, 0.05, f"{frac:.0%} sous le neutre",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color="0.35")

        ax.set_title(g, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        if not show_origin:
            ax.set_xlim(left=float(np.nanmin(x)))
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel(xlab)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\langle (w - w_{\mathrm{anc}})^2 \rangle$")

    fig.suptitle("Distance des poids à l'ancêtre commun — observé vs dérive neutre",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(exp_dir, "fig", fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


def plots_weight_selection(dist_list, neutral, exp_dir, steps, gen_depth=None,
                           x_axis="step", fname="weight_selection.png"):
    """Écart à la neutralité — le graphe à regarder en premier.

    Panneau gauche, rapport observé/neutre :
        ~1 -> neutralité   <1 -> sélection stabilisante   >1 -> directionnelle
    Il élimine la croissance en p*sigma^2*g commune aux quatre blocs et rend
    les blocs comparables malgré leurs tailles de couches différentes.

    Panneau droit, écart normalisé  (obs - neutre) / std_neutre :
        combien d'écarts-types neutres séparent l'observé de la neutralité.
        C'est la version quantitative : un rapport de 0.9 n'a pas le même sens
        selon que la bande neutre est large ou serrée, et la largeur de cette
        bande dépend du bloc (petites couches = bande large).
    """
    x, xlab = _xaxis(steps, gen_depth, x_axis)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, g in enumerate(GROUPS):
        m  = np.array([d[g][0] for d in dist_list])
        nm = np.array([n[g][0] for n in neutral])
        ns = np.array([n[g][1] for n in neutral])

        axes[0].plot(x, m / np.where(nm > 0, nm, np.nan),
                     marker="o", ms=3.5, color=f"C{i}", label=g)
        axes[1].plot(x, (m - nm) / np.where(ns > 0, ns, np.nan),
                     marker="o", ms=3.5, color=f"C{i}", label=g)

    axes[0].axhline(1.0, color="k", ls="--", lw=1.5, label="neutre")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("distance observée / neutre")
    axes[0].set_title("Rapport à la neutralité")

    axes[1].axhline(0.0, color="k", ls="--", lw=1.5)
    axes[1].axhspan(-2, 2, color="k", alpha=0.07, lw=0)
    axes[1].set_ylabel(r"$(\mathrm{obs} - \mathrm{neutre})\,/\,\sigma_{\mathrm{neutre}}$")
    axes[1].set_title("Écart normalisé (bande grise : $\\pm 2\\sigma$)")

    for ax in axes:
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(exp_dir, "fig", fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


def plot_generation_rate(steps, gen_depth, gen_depth_std, exp_dir,
                         fname="generation_rate.png"):
    """Diagnostic : g(t) est-il linéaire ?

    Si oui, le temps de génération est stationnaire et 'step' est
    interchangeable avec 'generation'. Sinon la démographie dérive, et seule
    l'abscisse 'generation' isole proprement la dérive des poids.
    """
    x = np.asarray(steps, float)
    g = np.asarray(gen_depth, float)
    s = np.asarray(gen_depth_std, float)

    alpha, b = np.polyfit(x, g, 1)
    pred = alpha * x + b
    r2 = 1.0 - np.sum((g - pred) ** 2) / np.sum((g - g.mean()) ** 2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, g, marker="o", ms=3.5, color="C0", label="profondeur mesurée")
    ax.fill_between(x, g - s, g + s, alpha=0.15, color="C0", lw=0)
    ax.plot(x, pred, "k--", lw=1.4,
            label=fr"{alpha:.3g} gén/step, $R^2$={r2:.4f}")
    ax.set_xlabel("step")
    ax.set_ylabel("générations depuis l'ancêtre")
    ax.set_title("Temps de génération")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(exp_dir, "fig", fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")