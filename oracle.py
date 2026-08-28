"""Politique code en dur, sans reseau : cherche le meilleur, evite le poison.

Sert de PLAFOND. Si un agent qui discrimine parfaitement n'a pas d'avantage
dans l'environnement, alors l'environnement ne recompense pas la
discrimination, et aucune architecture ni aucun reglage de mutation n'y changera
quoi que ce soit.

Repere de la vue, mesure sur agent_mov (vue 11x11 centree en (5,5)) :
    avancer            -> case (6, 5), soit ligne +1
    action 1 (+pi/2)   -> amene la case (5, 6) devant, soit colonne +1
    action 2 (-pi/2)   -> amene la case (5, 4) devant, soit colonne -1
La rotation de get_single_obs canonise l'orientation, donc c'est vrai quel que
soit le cap de l'agent.
"""
import jax
import jax.numpy as jnp


def oracle_actions(obs, resources, cout_distance=0.05):
    """(N, side, side, C) -> (N,) indice d'action.

    `obs` est la vue egocentrique deja tournee. Le bourrage hors grille vaut -1,
    donc `> 0` l'exclut naturellement.
    """
    n_types = len(resources)
    de = jnp.array([r.delta_energy for r in resources])          # (n_types,)
    side = obs.shape[1]
    c = side // 2

    present = obs[..., :n_types] > 0                             # (N, s, s, n)
    # valeur d'une case = le meilleur delta_energy qui s'y trouve
    val = jnp.max(jnp.where(present, de, -jnp.inf), axis=-1)     # (N, s, s)
    val = jnp.where(jnp.isfinite(val), val, 0.0)                 # case vide -> 0

    lignes = jnp.arange(side)[:, None]
    cols = jnp.arange(side)[None, :]
    dist = jnp.abs(lignes - c) + jnp.abs(cols - c)               # (s, s)

    # Cout en NOMBRE DE PAS, pas en distance brute : atteindre une case hors de
    # l'axe demande une rotation en plus. Sans ce terme, plusieurs cibles a
    # egale distance sont departagees par l'indice aplati -- qui n'est pas
    # invariant par rotation : l'agent vise une autre case a chaque quart de
    # tour et pivote sur place indefiniment.
    tourne = (cols != c).astype(jnp.float32)
    score = jnp.where(val > 0, val - cout_distance * (dist + tourne), -jnp.inf)
    score = score.at[:, c, c].set(-jnp.inf)      # pas se viser soi-meme

    plat = score.reshape(score.shape[0], -1)
    idx = jnp.argmax(plat, axis=-1)
    a_cible = jnp.max(plat, axis=-1) > -jnp.inf
    dr = idx // side - c
    dc = idx % side - c

    # On ferme d'abord l'ecart en LIGNE, on tourne ensuite. Tourner des que
    # dc != 0 ne marche pas : un quart de tour depasse une cible peu decalee,
    # qui repasse de l'autre cote -> l'agent oscille entre 1 et 2 sans avancer.
    vers_cible = jnp.where(dr > 0, 3,
                           jnp.where(dc > 0, 1, jnp.where(dc < 0, 2, 1)))
    action = jnp.where(a_cible, vers_cible, 3)                   # rien en vue : avancer

    # Ne jamais avancer sur du poison ni dans un mur. Le mur est le dernier
    # canal ([ressources, agents, murs]) et il tue quand letal_wall est vrai --
    # sans cette garde l'oracle fonce dans la bordure et meurt en ~40 pas.
    devant = val[:, c + 1, c]
    mur_devant = obs[:, c + 1, c, n_types + 1] > 0
    action = jnp.where((action == 3) & ((devant < 0) | mur_devant), 1, action)
    return action.astype(jnp.int32)
