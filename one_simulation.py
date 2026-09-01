# JAX's syntax is (for the most part) same as NumPy's!
# There is also a SciPy API support (jax.scipy)
import jax.numpy as jnp
import numpy as np
import jax
# Special transform functions (we'll understand what these are very soon!)

# JAX's low level API 
# (lax is just an anagram for XLA, not completely sure how they came up with name JAX)
from jax import lax

from jax import random
from functools import partial

from EcoEvoJax.source.agent import metaRNNPolicyState_bcppr
from simulation.update_env import resources_growth
from simulation.agent_mov import vmap_update_agents_position, get_obs_vector
from simulation.data_class import SimState
from simulation.oracle import oracle_actions
from simulation.utils.utils_sim import masque_valeur, masque_politique

from typing import NamedTuple
import jax

class StepLog(NamedTuple):
    position:  jax.Array   # (N, 2)
    alive:     jax.Array   # (N,)
    time_under_min_energy :     jax.Array   # (N,)
    energy:    jax.Array   # (N,)
    parent_id: jax.Array   # (N,)  -> généalogie
    born_step: jax.Array   # (N,)  -> généalogie / MRCA
    actions:   jax.Array   # (N, 4) -> si besoin pour les vidéos
    rewards:   jax.Array   # (N, 1) -> R0
    grid:      jax.Array   # (3, L, L) -> si tes vidéos affichent la grille
    step :     int
    obs :      jax.Array
    consumed_res: jax.Array   # (n_types,) -> unités retirées de la grille PENDANT ce step
    saw_res:   jax.Array   # (N, n_types) -> COMBIEN de cases de ce type dans la vue
    ate_res:   jax.Array   # (N, n_types) -> ce type a-t-il été consommé PENDANT ce step ?
    is_oracle: jax.Array   # (N,) -> 1 pour les envahisseurs
    perte_pred: jax.Array  # (N,) -> erreur de prediction de la boucle interne,
                           #         vide si inner_loop=False
    divergence_pol: jax.Array  # (N,) -> ecart entre politique apprise et genome
    

def ecrete(grad, seuil):
    """Ramene la norme du gradient de chaque agent sous `seuil`."""
    if not seuil:
        return grad
    norme = jnp.linalg.norm(grad, axis=1, keepdims=True)
    return grad * jnp.minimum(1.0, seuil / (norme + 1e-8))


def pas_gradient_intra_vie(model, cfg, inner, vivants, masque):
    """Un pas de SGD sur la voie LSTM -> tete de valeur, par agent.

    BPTT tronque sur la fenetre : le carry de depart est celui du DEBUT de
    fenetre, garde dans InnerState. Sans cette troncature le gradient ne
    remonterait que d'un pas, et le reseau ne pourrait pas apprendre a RETENIR
    une bouchee passee -- ce qui est precisement ce qu'on lui demande.

    Le conv et la tete de politique ne sont pas dans le graphe : en mode separee
    la voie de valeur en est independante, donc la passe arriere ne porte que sur
    ~600 parametres.
    """
    def perte(theta, h0, c0, t_in, t_eat, t_r):
        p = model.format_one_fn(theta)

        def avance(carry, x):
            h, c = carry
            mem_in, eat, r = x
            h, c, v = model.valeur_apply(p, h, c, mem_in)
            # v est la prediction faite AVANT que (eat, r) n'entre dans le LSTM :
            # ils n'y arriveront qu'au pas suivant. C'est donc une prediction,
            # pas une recopie de l'entree.
            return (h, c), ((v - r) ** 2 * eat).sum()

        _, err = lax.scan(avance, (h0, c0), (t_in, t_eat, t_r))
        # Moyenne par BOUCHEE : les pas sans repas ne portent pas de cible.
        # Fenetre sans aucune bouchee -> NaN, et non zero : un zero se lirait
        # comme "predit parfaitement" sur la courbe. Le gradient, lui, vaut bien
        # zero des deux facons (la branche NaN est une constante).
        n = t_eat.sum()
        return jnp.where(n > 0, err.sum() / jnp.maximum(n, 1.0), jnp.nan)

    val, grad = jax.vmap(jax.value_and_grad(perte))(
        inner.params_vie, inner.carry_h, inner.carry_c,
        inner.tampon_in, inner.tampon_eaten, inner.tampon_r)
    grad = ecrete(grad * masque[None], cfg.inner_clip)
    maj = inner.params_vie - cfg.inner_lr * grad * vivants[:, None]
    return maj, val


# Indice de l'action "avancer" : la seule qui deplace sur une case, donc la
# seule dont la valeur immediate soit definie. Cf. agent_mov.action_depl_theta.
AVANCER = 3


def pas_gradient_politique(model, cfg, params_vie, h, c, obs, mem_in,
                           actifs, masque):
    """Un pas de SGD sur -somme_a pi(a) * v_hat(a), par agent.

    v_hat n'est defini que pour "avancer" : les autres actions ne posent l'agent
    sur aucune case, leur valeur immediate est nulle. La loss se reduit donc a
    -pi(avancer) * valeur_de_la_case_devant, ce qui pousse a avancer sur ce que
    l'agent croit bon et a s'en detourner sinon.

    La croyance passe par stop_gradient : c'est une CIBLE. Sans ca le reseau
    baisserait la loss en gonflant v_pred au lieu de corriger ses actions.

    Calculee au pas courant et non sur la fenetre : elle ne demande aucune
    assignation de credit dans le temps, et stocker les observations d'une
    fenetre couterait des gigaoctets.
    """
    n_types = len(cfg.resources)
    centre = cfg.agent_view

    def perte(theta, h, c, obs, mem_in):
        p = model.format_one_fn(theta)
        # meme decoupage que l'entree du LSTM, cf. get_actions
        la, rw = mem_in[:cfg.output_dim], mem_in[cfg.output_dim:cfg.output_dim + 1]
        en, le = mem_in[cfg.output_dim + 1:cfg.output_dim + 2], mem_in[cfg.output_dim + 2:]
        _, _, v = model.valeur_apply(jax.lax.stop_gradient(p), h, c, mem_in)
        v = jax.lax.stop_gradient(v * cfg.vpred_gain)

        logits = model.logits_apply(p, h, c, obs, la, rw, en, le)
        pi = jax.nn.softmax(logits / cfg.temperature)
        # la case devant : ligne +1 dans la vue egocentrique (cf. oracle.py)
        v_devant = (obs[centre + 1, centre, :n_types] * v).sum()
        return -pi[AVANCER] * v_devant

    grad = jax.vmap(jax.grad(perte))(params_vie, h, c, obs, mem_in)
    grad = ecrete(grad * masque[None], cfg.inner_clip)
    poids = actifs.astype(params_vie.dtype)[:, None]
    return params_vie - cfg.inner_policy_lr * grad * poids


def divergences(model, cfg, params_vie, genome, h, c, obs, mem_in, masque_pol):
    """Deux distances en variation totale, sur la meme vue.

    `totale`    : politique apprise contre celle du genome. Tout ce que la
                  boucle interne a change, croyance comprise.
    `politique` : politique apprise contre la MEME croyance mais les poids de
                  politique remis a ceux du genome. Isole donc la seule
                  contribution de la retropropagation sur la tete.

    La distinction compte : v_pred entre dans la tete et dans la carte, donc une
    croyance apprise change deja le comportement sans qu'aucun poids de politique
    n'ait bouge. Confondre les deux fait croire que la loss de politique agit
    alors qu'elle n'a peut-etre jamais tourne.

    0 = aucun effet, 1 = plus rien en commun. Aux fins de fenetre seulement :
    trois passes avant.
    """
    def paires(vie, gen, h, c, obs, mem_in):
        la, rw = mem_in[:cfg.output_dim], mem_in[cfg.output_dim:cfg.output_dim + 1]
        en, le = mem_in[cfg.output_dim + 1:cfg.output_dim + 2], mem_in[cfg.output_dim + 2:]
        pi = lambda th: jax.nn.softmax(
            model.logits_apply(model.format_one_fn(th), h, c, obs, la, rw, en, le)
            / cfg.temperature)
        # croyance apprise, politique du genome
        mixte = vie * (1.0 - masque_pol) + gen * masque_pol
        p_vie = pi(vie)
        return (0.5 * jnp.abs(p_vie - pi(gen)).sum(),
                0.5 * jnp.abs(p_vie - pi(mixte)).sum())

    return jax.vmap(paires)(params_vie, genome, h, c, obs, mem_in)


@partial(jax.jit, static_argnames=['cfg','model'])
def run_simulation_chunk(state,model,keys, cfg):
    

    def step(state, subkey):
        
        grid         = state.grid
        agents       = state.agents
        step_idx     = state.step
        
        n_types = grid.shape[0] - 2          # tout sauf agents + murs
        grid_resources = grid[:n_types]      # (n_types, L, L)  -> slice, garde l'axe type
        grid_agents    = grid[n_types]       # (L, L)           -> index, un seul canal
        grid_walls     = grid[n_types + 1] 

        key_env, key_action, key_respawn, key_mut = random.split(subkey, 4)
        res = jax.tree.map(lambda *xs: jnp.stack(xs), *cfg.resources) # Resource parameters

        # ------- 1. Evaluate alive agents and energy : -------
        energies = agents.energy
        is_alive = agents.alive > 0

        # Incremente time if energy is above or below a set threshold
        new_time_under = jnp.where((energies < 0) & is_alive, agents.time_under_min_energy + 1, 0) #If true a bit closer to death time
        new_time_over = jnp.where((energies > cfg.min_energy_repr) & is_alive, agents.time_over_energy_repr + 1, 0) # If true a bit closer to reproduction time

        survives = (new_time_under < cfg.time_to_die) & is_alive #Surviving agents, already alive and not about to die
        reproduces = new_time_over > cfg.time_above_repr # Check if reproducing time

        new_time_over = jnp.where(reproduces, 0, new_time_over) # If going to reproduce reset reproduction time to zero
        survives_int = jnp.where(survives, 1, 0) #Bool alive array to Int alive array
        
        cut_rec = cfg.ablate_memory or cfg.ablate_recurrence
        cut_int = cfg.ablate_memory or cfg.ablate_interoception
        cut_fb  = cfg.ablate_memory or cfg.ablate_feedback

        state_in = state
        if cut_fb:
            state_in = state_in.replace(
                last_actions=jnp.zeros_like(state.last_actions),
                rewards=jnp.zeros_like(state.rewards),
                last_eaten=jnp.zeros_like(state.last_eaten),
            )
        if cut_int:
            # constante, pas zero : zero est la valeur de MORT (energy_to_die)
            state_in = state_in.replace(
                agents=state_in.agents.replace(
                    energy=jnp.full_like(state.agents.energy, cfg.starting_energy)),
            )

        # Avec la boucle interne, la passe avant lit params_vie (poids appris
        # pendant la vie) ; agents.params reste le genome, intact, et c'est lui
        # seul qui est transmis a la reproduction.
        params_avant = (state.agents.params if agents.inner is None
                        else agents.inner.params_vie)
        actions_logit, new_policy_states = model.get_actions(
            state_in, params_avant, state.agents.policy_states
        )
        if cut_rec:
            # apres l'appel : rien ne passe au pas suivant
            new_policy_states = metaRNNPolicyState_bcppr(
                lstm_h=jnp.zeros_like(new_policy_states.lstm_h),
                lstm_c=jnp.zeros_like(new_policy_states.lstm_c),
                keys=new_policy_states.keys,
            )
        if cfg.dumb_agent:
            acts_idx = random.randint(key_action, shape=(cfg.n_agents_max,),
                                      minval=0, maxval=cfg.output_dim)
        else:
            acts_idx = random.categorical(key_action, actions_logit / cfg.temperature, axis=-1)

        # L'oracle remplace le reseau AGENT PAR AGENT : cfg.oracle_agent le force
        # pour tous, is_oracle ne le donne qu'aux envahisseurs. Il est calcule
        # pour tout le monde -- quelques operations sur (N, 11, 11, C), du meme
        # ordre que la passe conv -- et selectionne ensuite.
        if cfg.oracle_agent or cfg.invasion_start > 0:
            a_oracle = oracle_actions(
                state.obs, cfg.resources,
                energy=state.agents.energy if cfg.oracle_wait else None,
                energy_max=cfg.energy_max if cfg.oracle_wait else None,
                croyance=agents.croyance if cfg.oracle_apprend else None)
            prend = (agents.is_oracle > 0) if not cfg.oracle_agent else jnp.ones_like(agents.is_oracle, bool)
            acts_idx = jnp.where(prend, a_oracle, acts_idx)

        actions_id = jax.nn.one_hot(acts_idx, cfg.output_dim)

        acts = jnp.argmax(actions_id, axis=1)
        agents= vmap_update_agents_position(agents,acts,cfg.grid_length) 
        
        pos = agents.position
        
        if cfg.letal_wall :
            survives_int = jnp.where(grid_walls[pos[:, 0], pos[:, 1]]==1,0,survives_int)
            reproduces = jnp.where(grid_walls[pos[:, 0], pos[:, 1]]==1,0,reproduces)
        

        # Compute intern energy : resource consumption - energy decay
        local_resources = grid_resources[:, pos[:, 0], pos[:, 1]].T
        gain = local_resources @ res.delta_energy
        rewards = survives_int * gain
        new_energy = jnp.minimum(energies + rewards - cfg.energy_decay * jnp.where(acts==0, cfg.factor_energy_decay_not_moving,1) * survives_int, cfg.energy_max)

        ate_res_step = (local_resources > 0) & (survives_int[:, None] > 0)   # (N, n_types)

        # Ce que l'agent APPREND de sa bouchee. Les trois delta_energy etant
        # distincts, une seule suffit a identifier le canal goute. La croyance
        # est par CANAL, donc elle devient fausse apres un shuffle : c'est
        # exactement la reevaluation intra-vie qu'on veut mesurer.
        croyance_maj = jnp.where(ate_res_step, rewards[:, None], agents.croyance)
        saw_res_step = (state.obs[..., :n_types] > 0).sum(axis=(1, 2))       # (N, n_types)

        # ------- 2 bis. Boucle interne : predire la valeur de sa bouchee -------
        # cut_rec remet le carry a zero apres chaque pas : rien ne s'accumule,
        # donc il n'y a plus rien a predire. On gele la boucle interne au lieu
        # d'entrainer dans le vide -- sinon le bras ablate du lab differerait de
        # l'intact par ses POIDS autant que par sa memoire, et la comparaison
        # appariee ne mesurerait plus ce qu'elle annonce.
        inner_maj = agents.inner
        if inner_maj is not None and not cut_rec:
            # Reconstitue l'entree du LSTM A L'IDENTIQUE de get_actions : la
            # rejouer autrement ferait apprendre le gradient sur un autre reseau
            # que celui qui a agi.
            mem_in = jnp.concatenate([
                state_in.last_actions,
                state_in.rewards,
                jnp.expand_dims(state_in.agents.energy, 1).astype(jnp.float32),
                state_in.last_eaten.astype(jnp.float32)], axis=1)      # (N, d_mem)

            # Loss de politique : chaque pas, sur la vue et la croyance courantes.
            # Reservee aux agents dont la croyance a fait ses preuves. Sinon la
            # population apprend des le pas 1 a suivre du bruit, que vpred_gain
            # amplifie -- des agents qui foncent avec conviction sur n'importe
            # quoi. NaN (jamais mesure) donne faux, donc pas de gradient.
            masque_pol = masque_politique(model)
            if cfg.inner_policy:
                confiant = (inner_maj.perte < cfg.inner_policy_seuil)
                inner_maj = inner_maj.replace(
                    params_vie=pas_gradient_politique(
                        model, cfg, inner_maj.params_vie,
                        agents.policy_states.lstm_h, agents.policy_states.lstm_c,
                        state.obs, mem_in,
                        (survives_int > 0) & confiant, masque_pol))

            case = step_idx % cfg.inner_window
            inner_maj = inner_maj.replace(
                tampon_in=inner_maj.tampon_in.at[:, case].set(mem_in),
                tampon_eaten=inner_maj.tampon_eaten.at[:, case].set(
                    ate_res_step.astype(jnp.float32)),
                tampon_r=inner_maj.tampon_r.at[:, case].set(rewards),
            )

            # La fenetre est pleine : un pas de gradient, puis le carry courant
            # devient le point de depart de la fenetre suivante. step_idx est un
            # scalaire partage par tous les agents, donc lax.cond branche vraiment
            # au lieu de calculer les deux cotes.
            masque = masque_valeur(model)

            def maj(inn):
                vie, perte = pas_gradient_intra_vie(
                    model, cfg, inn, survives_int.astype(jnp.float32), masque)
                if cfg.inner_policy:
                    div_tot, div = divergences(
                        model, cfg, vie, agents.params,
                        agents.policy_states.lstm_h, agents.policy_states.lstm_c,
                        state.obs, mem_in, masque_pol)
                else:
                    div_tot = div = inn.divergence
                if cfg.inner_policy and cfg.log_inner:
                    # Une ligne par fin de fenetre : c'est la seule occasion ou
                    # `perte` change, donc ou la liste des agents confiants bouge.
                    # Sans ca on ne sait pas si la loss de politique agit ou si
                    # le seuil est trop serre.
                    en_vie = survives_int > 0
                    jax.debug.print(
                        "[politique] pas {p} : {n} agents sur {t} sous le seuil "
                        "{s} -- erreur mediane {m}, minimale {b}",
                        p=step_idx, s=cfg.inner_policy_seuil,
                        n=((perte < cfg.inner_policy_seuil) & en_vie).sum(),
                        t=en_vie.sum(),
                        m=jnp.nanmedian(jnp.where(en_vie, perte, jnp.nan)),
                        b=jnp.nanmin(jnp.where(en_vie, perte, jnp.nan)))
                    ecart = jnp.linalg.norm((vie - agents.params) * masque_pol[None],
                                            axis=1)
                    jax.debug.print(
                        "[politique] pas {p} : divergence due a la TETE {d} "
                        "(max {x}), due a tout {t} ; poids de tete deplaces chez "
                        "{n} agents, ecart max {w}",
                        p=step_idx,
                        d=jnp.nanmedian(jnp.where(en_vie, div, jnp.nan)),
                        x=jnp.nanmax(jnp.where(en_vie, div, jnp.nan)),
                        t=jnp.nanmedian(jnp.where(en_vie, div_tot, jnp.nan)),
                        n=((ecart > 0) & en_vie).sum(),
                        w=jnp.nanmax(jnp.where(en_vie, ecart, jnp.nan)))
                return inn.replace(params_vie=vie, perte=perte, divergence=div,
                                   carry_h=new_policy_states.lstm_h,
                                   carry_c=new_policy_states.lstm_c)

            inner_maj = lax.cond(case == cfg.inner_window - 1,
                                 maj, lambda inn: inn, inner_maj)


        # ------- 3. Environment dynamic -------
        # Agents consume resources on the grid
        # Use max() instead of set() to avoid non-determinism with repeated indices on GPU
        consumed = jnp.zeros((cfg.grid_length, cfg.grid_length), dtype=grid_resources.dtype) # (L, L)
        consumed = consumed.at[pos[:, 0], pos[:, 1]].max(survives_int)     # (L, L)

        consumed_per_type = (grid_resources * consumed[None]).sum(axis=(1, 2))   # (n_types,)

        grid_resources = grid_resources * (1 - consumed[None]) #  consumed[None]: (n_types, L , L)

        # Resources spread via convolution
        if cfg.resources_growth : 
            grid_resources,_ = resources_growth((grid_resources,key_env),cfg,step=step_idx)
            grid_resources = jnp.where(grid_walls==1,0,grid_resources)



        # ------- 4. Reproduction -------
        # Compute the number of births (Number of parents VS Number free idx)
        
        if cfg.reproduction_on:
            nb_parents = reproduces.sum()
            nb_free_places = cfg.n_agents_max - survives.sum()
            nb_births = jnp.minimum(nb_parents, nb_free_places)

            free_idx = jnp.nonzero(survives == 0, size=cfg.n_agents_max, fill_value=0)[0]
            sort_free_idx = jnp.sort(free_idx)
            sort_free_idx_ordered = jnp.flip(sort_free_idx)

            spawn_mask = jnp.arange(cfg.n_agents_max) < nb_births
            free_indices = jnp.where(spawn_mask, sort_free_idx_ordered, 0)

            all_potential_parents = jnp.nonzero(reproduces, size=cfg.n_agents_max, fill_value=0)[0]
            parent_indices = jnp.where(spawn_mask, all_potential_parents, 0)
            
            # Draw initial positions for the new borns
            if cfg.random_pos_offspring:
                new_positions = random.randint(key_respawn, (cfg.n_agents_max, 2), minval=1, maxval=cfg.grid_length-1)
            else :
                new_positions = pos[parent_indices,:]


            # ------- 5. Global update -------
            # Put new born in the state with their initial state
            final_pos = pos.at[free_indices].set(new_positions)
            final_energy = new_energy.at[free_indices].set(cfg.starting_energy)
            final_time_under = new_time_under.at[free_indices].set(0)
            final_time_over = new_time_over.at[free_indices].set(0)
            final_alive = survives_int.at[free_indices].set(1)
            final_parent_id = agents.parent_id.at[free_indices].set(parent_indices)
            final_born_step = agents.born_step.at[free_indices].set(step_idx)

            # Invasion : l'enfant herite du drapeau de son parent, sauf pendant
            # la fenetre ou on convertit les naissances en oracles jusqu'a
            # atteindre la cible. rang = combien d'oracles cette naissance-ci
            # ajouterait, pour ne pas depasser.
            herite = agents.is_oracle[parent_indices]
            n_vivants = (agents.is_oracle * survives_int).sum()
            cible = cfg.invasion_frac * cfg.n_agents_max
            rang = jnp.cumsum(spawn_mask) - 1
            # `invasion_faite` verrouille : une fois la cible atteinte,
            # l'injection ne se rallume plus jamais. Sans ce verrou, des
            # envahisseurs en declin seraient re-remplis a 10% et la courbe
            # afficherait un plateau au lieu de la chute.
            convertit = (spawn_mask & (step_idx >= cfg.invasion_start)
                         & (cfg.invasion_start > 0) & (n_vivants + rang < cible)
                         & (state.invasion_faite == 0))
            final_is_oracle = agents.is_oracle.at[free_indices].set(
                jnp.where(convertit, 1.0, herite))
            # le nouveau-ne ne sait rien : c'est ce qui force a reapprendre a
            # chaque vie, et donc ce qui fait du test un test intra-vie
            final_croyance = croyance_maj.at[free_indices].set(cfg.croyance_init)
            
            final_alive_without_0 = final_alive.at[0].set(0)
            
            key_child_param_mutate,key_params = random.split(key_mut)
            
            parameters_to_mutate = random.bernoulli(key_child_param_mutate, p=cfg.param_mutate, shape=(cfg.n_agents_max, agents.params.shape[1])).astype(jnp.int32)
            number_of_parameters_mutating = parameters_to_mutate.sum(axis=1)
            
            mutation = cfg.mutation_var * random.normal(key_params, shape=(cfg.n_agents_max, agents.params.shape[1]))
            final_params = agents.params.at[free_indices].set(
                        agents.params[parent_indices] +mutation*parameters_to_mutate)
            
            # L'enfant repart de son genome mute : les poids appris par le
            # parent meurent avec lui. C'est ce qui rend le mecanisme
            # baldwinien et non lamarckien. Ses tampons sont vides -- il doit
            # tout reapprendre, c'est la mesure intra-vie.
            if agents.inner is not None:
                z = jnp.zeros_like
                final_inner = inner_maj.replace(
                    params_vie=inner_maj.params_vie.at[free_indices].set(
                        final_params[free_indices]),
                    tampon_in=inner_maj.tampon_in.at[free_indices].set(
                        z(inner_maj.tampon_in[0])),
                    tampon_eaten=inner_maj.tampon_eaten.at[free_indices].set(
                        z(inner_maj.tampon_eaten[0])),
                    tampon_r=inner_maj.tampon_r.at[free_indices].set(
                        z(inner_maj.tampon_r[0])),
                    carry_h=inner_maj.carry_h.at[free_indices].set(
                        z(inner_maj.carry_h[0])),
                    carry_c=inner_maj.carry_c.at[free_indices].set(
                        z(inner_maj.carry_c[0])),
                    # NaN et non zero : le nouveau-ne n'a pas encore de perte,
                    # un zero le ferait compter comme "predit parfaitement" et
                    # tirerait la courbe vers le bas a chaque renouvellement.
                    perte=inner_maj.perte.at[free_indices].set(jnp.nan),
                    divergence=inner_maj.divergence.at[free_indices].set(0.0),
                )
            else:
                final_inner = None

            final_policy_states =metaRNNPolicyState_bcppr(
                    lstm_h=new_policy_states.lstm_h.at[free_indices].set(jnp.zeros(new_policy_states.lstm_h.shape[1])),
                    lstm_c=new_policy_states.lstm_c.at[free_indices].set(jnp.zeros(new_policy_states.lstm_c.shape[1])),
                    keys=new_policy_states.keys)

        else : 
            final_pos= pos
            final_energy = new_energy
            final_time_under = new_time_under
            final_time_over = new_time_over
            final_alive = survives_int
            final_parent_id = agents.parent_id
            final_born_step = agents.born_step
            final_alive_without_0 = final_alive.at[0].set(0)
            final_policy_states = new_policy_states
            final_params = agents.params
            final_is_oracle = agents.is_oracle
            final_croyance = croyance_maj
            final_inner = inner_maj
            
            
        # Update spatial grid with agents positions
        grid_agents = jnp.zeros_like(grid_walls, dtype=jnp.int32)
        grid_agents = grid_agents.at[final_pos[:, 0], final_pos[:, 1]].add(final_alive_without_0)
        
        new_agents = agents.replace(
            position=final_pos,
            energy=final_energy,
            time_under_min_energy=final_time_under,
            time_over_energy_repr=final_time_over,
            alive=final_alive_without_0,
            parent_id=final_parent_id,
            born_step=final_born_step,
            policy_states = final_policy_states,
            params = final_params,
            is_oracle = final_is_oracle,
            croyance = final_croyance,
            inner = final_inner,
        )
        
        n_oracles = (final_is_oracle * final_alive_without_0).sum()
        invasion_faite = jnp.where(
            n_oracles >= cfg.invasion_frac * cfg.n_agents_max, 1.0,
            state.invasion_faite)

        new_grid = jnp.concatenate([grid_resources, grid_agents[None], grid_walls[None]], axis=0)
        obs = get_obs_vector(new_grid, (final_pos, new_agents.orientation), cfg.agent_view)

        
        new_state = SimState(
            grid=new_grid,
            agents=new_agents,
            step=step_idx+1,
            obs=obs,
            last_actions=actions_id,
            rewards=jnp.expand_dims(rewards, 1).astype(jnp.float32),
            last_eaten=ate_res_step.astype(jnp.float32),
            invasion_faite=invasion_faite,
        )
        
        log = StepLog(
            position=state.agents.position,
            alive=state.agents.alive,
            energy=state.agents.energy,
            parent_id=state.agents.parent_id,
            born_step=state.agents.born_step,
            actions=state.last_actions,
            rewards=state.rewards,
            grid = state.grid if cfg.log_grid else jnp.zeros((0,), dtype=state.grid.dtype),
            step = step_idx,
            time_under_min_energy = state.agents.time_under_min_energy,
            obs = state.obs if cfg.log_obs else jnp.zeros((0,), dtype=state.obs.dtype),
            consumed_res = consumed_per_type,
            saw_res = saw_res_step,
            ate_res = ate_res_step,
            is_oracle = state.agents.is_oracle,
            perte_pred = (jnp.zeros((0,)) if state.agents.inner is None
                          else state.agents.inner.perte),
            divergence_pol = (jnp.zeros((0,)) if state.agents.inner is None
                              else state.agents.inner.divergence),
        )
        
        return new_state, log
    state_final, outputs = lax.scan(step, state, keys)

    return state_final, outputs

