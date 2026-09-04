"""Le circuit memoire est-il EXPRIMABLE, et atteignable par gradient ?

    python -m simulation.tools.probe_memory --model v2
    python -m simulation.tools.probe_memory --model v1

On isole la question du reste : pas de politique, pas de RL, pas de rollout.
Le reseau recoit le flux qu'il recoit en simulation -- action precedente,
recompense, energie, ce qu'il vient de manger, et la vision -- et on lui demande
de dire QUEL CANAL PORTE LE POISON. La permutation change a chaque episode, donc
il ne peut pas la ranger dans ses poids : il doit la deduire de ce qu'il vit.

C'est exactement l'apprentissage intra-vie qu'on cherche a faire evoluer, pose
en supervise. Deux issues, toutes deux informatives :

  - le gradient trouve  -> le circuit tient dans 8 unites LSTM et il est
                           atteignable. Le probleme de la simulation est la
                           RECHERCHE evolutive, pas l'architecture.
  - le gradient echoue  -> c'est l'architecture ou l'information disponible.
                           Aucun reglage de mutation n'y changera rien.

La comparaison v1 / v2 est le second interet : en v1 le LSTM ne recoit PAS
last_eaten, donc il ne sait pas ce qui a produit la recompense.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from simulation.data_class import BASE_RESOURCES, LABELS, MODEL_VERSIONS, label_of

# Contournement LOCAL : flax 0.6.11 n'accepte pas LSTMCell(features=...), le
# cluster (>= 0.7) si. On ne touche a rien quand la signature l'accepte.
try:
    nn.LSTMCell(features=4)
except TypeError:
    _LSTM = nn.recurrent.LSTMCell

    class _LSTMCompat(_LSTM):
        def __init__(self, *a, features=None, **k):
            super().__init__(*a, **k)

    nn.recurrent.LSTMCell = _LSTMCompat
    print("[probe] flax ancien : shim LSTMCell(features=) applique")

from EcoEvoJax.source.agent import MetaRNN_bcppr


class Sonde(nn.Module):
    """Le reseau de la simulation, plus une lecture lineaire du carry."""
    memory_mode: str
    carry_size: int = 8
    hidden_layers: tuple = (32,)
    n_types: int = 3

    def setup(self):
        self.coeur = MetaRNN_bcppr(output_size=4, out_fn="categorical",
                                   hidden_layers=list(self.hidden_layers),
                                   encoder_in=False,
                                   encoder_layers=[], carry_size=self.carry_size,
                                   memory_mode=self.memory_mode)
        self.lecture = nn.Dense(self.n_types)

    def __call__(self, h, c, obs, last_action, reward, energy, last_eaten):
        h, c, _ = self.coeur(h, c, obs, last_action, reward, energy, last_eaten)
        return h, c, self.lecture(h)


def episodes(cle, n_ep, T, de_par_id, n_types=3, cote=11, n_canaux=None):
    """Un lot d'episodes. Par episode : une permutation canal -> identite tiree
    au hasard, donc la cible change d'un episode a l'autre."""
    # ressources + agents + murs : deduit, sinon la vue serait mal formee des
    # que le nombre de ressources change
    if n_canaux is None:
        n_canaux = n_types + 2
    c_perm, c_eat, c_act, c_obs = jax.random.split(cle, 4)

    # canal k porte l'identite perm[k]
    perm = jnp.stack([jax.random.permutation(k, n_types)
                      for k in jax.random.split(c_perm, n_ep)])       # (E, 3)
    # La plus NEFASTE, lue sur delta_energy : LABELS.index("poison") supposait
    # qu'un poison existe et qu'il porte l'identite 2. A une ou deux ressources
    # c'est faux, et rien ne garantit qu'un "poison" figure dans la liste.
    cible = jnp.argmax(perm == jnp.argmin(de_par_id), axis=1)         # (E,)

    # a chaque pas : rien mange (0) ou le canal 1..3
    tire = jax.random.randint(c_eat, (n_ep, T), 0, n_types + 1)
    mange = jax.nn.one_hot(tire - 1, n_types) * (tire > 0)[..., None]  # (E,T,3)

    de_canal = de_par_id[perm]                                        # (E,3)
    reward = (mange * de_canal[:, None, :]).sum(-1, keepdims=True)    # (E,T,1)

    # l'energie suit la recompense, bornee comme dans la simulation
    energie = jnp.clip(1.5 + jnp.cumsum(reward[..., 0], axis=1) * .3, 0., 8.)

    last_action = jax.nn.one_hot(jax.random.randint(c_act, (n_ep, T), 0, 4), 4)
    obs = (jax.random.uniform(c_obs, (n_ep, T, cote, cote, n_canaux)) < .2
           ).astype(jnp.float32)
    return obs, last_action, reward, energie[..., None], mange, cible


def deroule(appliquer, params, lot, carry_size):
    obs, la, rw, en, le, _ = lot
    E, T = obs.shape[0], obs.shape[1]
    h = jnp.zeros((E, carry_size))
    c = jnp.zeros((E, carry_size))

    def pas(etat, t):
        h, c = etat
        h, c, logits = jax.vmap(appliquer, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
            params, h, c, obs[:, t], la[:, t], rw[:, t], en[:, t], le[:, t])
        return (h, c), logits

    _, logits = jax.lax.scan(pas, (h, c), jnp.arange(T))
    return jnp.swapaxes(logits, 0, 1)                                 # (E,T,3)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-m", "--model", choices=["v1", "v2"], default="v2",
                   help="v1 = memoire jointe a la perception, v2 = separee")
    p.add_argument("--carry", type=int, default=None,
                   help="taille du carry (defaut : celle de MODEL_VERSIONS)")
    p.add_argument("--hidden", type=int, nargs="*", default=None,
                   help="tete (defaut : celle de MODEL_VERSIONS)")
    p.add_argument("--steps", type=int, default=40, help="pas par episode")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--chauffe", type=int, default=5,
                   help="pas ignores dans la loss : avant, rien n'est deductible")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    # la forme vient de MODEL_VERSIONS : la sonde doit tester TON reseau,
    # pas une variante codee en dur ici
    spec = MODEL_VERSIONS[a.model]
    mode = spec["memory_mode"]
    if a.carry is None:
        a.carry = spec["hidden_dim"]
    tete = tuple(a.hidden) if a.hidden is not None else tuple(spec["hidden_layers"])
    # Les identites REELLEMENT presentes, pas range(3) : BASE_RESOURCES peut en
    # porter une seule comme quatre.
    ids = sorted(r.id for r in BASE_RESOURCES)
    n_types = len(ids)
    if n_types < 2:
        print(f"probe_memory : {n_types} ressource(s). Il en faut au moins 2 "
              "pour qu'il y ait quelque chose a identifier.")
        return
    de_par_id = jnp.array([next(r.delta_energy for r in BASE_RESOURCES if r.id == i)
                           for i in ids])
    print(f"modele {a.model} ({mode})  carry={a.carry}  tete={list(tete)}  "
          f"{n_types} ressources\n"
          f"delta_energy par identite : "
          + "  ".join(f"{label_of(i)} {float(de_par_id[k]):+g}"
                      for k, i in enumerate(ids)))

    sonde = Sonde(memory_mode=mode, carry_size=a.carry, hidden_layers=tete,
                  n_types=n_types)
    cle = jax.random.PRNGKey(a.seed)
    cle, c0 = jax.random.split(cle)
    lot0 = episodes(c0, 2, a.steps, de_par_id, n_types)
    params = sonde.init(jax.random.PRNGKey(a.seed + 1),
                        jnp.zeros(a.carry), jnp.zeros(a.carry),
                        lot0[0][0, 0], lot0[1][0, 0], lot0[2][0, 0],
                        lot0[3][0, 0], lot0[4][0, 0])
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"parametres entraines : {n_par}")

    opt = optax.adam(a.lr)
    etat = opt.init(params)
    appliquer = lambda pr, h, c, o, la, rw, en, le: sonde.apply(pr, h, c, o, la, rw, en, le)

    def perte(params, lot):
        logits = deroule(appliquer, params, lot, a.carry)
        cible = lot[5]
        lp = jax.nn.log_softmax(logits[:, a.chauffe:])
        vrai = jnp.take_along_axis(lp, cible[:, None, None], axis=-1)[..., 0]
        exact = (jnp.argmax(logits[:, a.chauffe:], -1) == cible[:, None])
        return -vrai.mean(), exact.mean()

    @jax.jit
    def etape(params, etat, cle):
        lot = episodes(cle, a.batch, a.steps, de_par_id, n_types)
        (l, acc), g = jax.value_and_grad(perte, has_aux=True)(params, lot)
        maj, etat = opt.update(g, etat, params)
        return optax.apply_updates(params, maj), etat, l, acc

    print(f"\n{'iter':>6}{'loss':>9}{'exactitude':>12}"
          f"   (hasard = {1/n_types:.3f})")
    for it in range(a.iters + 1):
        cle, ce = jax.random.split(cle)
        params, etat, l, acc = etape(params, etat, ce)
        if it % max(1, a.iters // 10) == 0:
            print(f"{it:>6}{float(l):>9.4f}{float(acc):>12.3f}")

    # exactitude en fonction du nombre de pas ecoules dans l'episode
    cle, ct = jax.random.split(cle)
    lot = episodes(ct, 1024, a.steps, de_par_id, n_types)
    logits = deroule(appliquer, params, lot, a.carry)
    exact = (jnp.argmax(logits, -1) == lot[5][:, None])
    print("\nexactitude par pas de l'episode :")
    for t in range(0, a.steps, max(1, a.steps // 10)):
        print(f"  pas {t:>3} : {float(exact[:, t].mean()):.3f}")
    print(f"  pas {a.steps-1:>3} : {float(exact[:, -1].mean()):.3f}")


if __name__ == "__main__":
    main()
