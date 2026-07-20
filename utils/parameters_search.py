import os 
import optuna
import logging
import json
import jax
from simulation.run import launch_simulation_chunked
from simulation.utils.utils_sim import outputs_to_numpy

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

    
    pop_full = outputs_to_numpy(outputs.alive).sum(axis=1)
    
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
    