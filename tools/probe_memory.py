"""Le circuit memoire est-il EXPRIMABLE, et atteignable par gradient ?

    python -m simulation_meta.tools.probe_memory --model v2
    python -m simulation_meta.tools.probe_memory --model v1

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
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn

from simulation_meta.data_class import (BASE_RESOURCES, LABELS,
                                   MODEL_VERSIONS, sous_ensemble)

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

from EcoEvoJax_meta.source.agent import MetaRNN_bcppr


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
    au hasard, donc la cible change d'un episode a l'autre.

    La cible est le canal portant la ressource la plus NEFASTE, designee par son
    delta_energy et non par son nom : le nombre de ressources est libre, et rien
    ne garantit qu'il y ait un "poison" dans la liste.
    """
    if n_canaux is None:
        n_canaux = n_types + 2        # ressources + agents + murs
    c_perm, c_eat, c_act, c_obs = jax.random.split(cle, 4)

    # canal k porte l'identite perm[k]
    perm = jnp.stack([jax.random.permutation(k, n_types)
                      for k in jax.random.split(c_perm, n_ep)])       # (E, 3)
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


def trace(a, iters, bouchees, n_types, res, tete):
    """Exactitude en fonction des mises a jour, une couleur par nb de bouchees.

    Les deux axes de la question : combien d'entrainement pour que le circuit
    existe (abscisse), et combien de bouchees pour qu'il tranche (couleur).
    """
    os.makedirs(a.fig_dir, exist_ok=True)
    hasard = 1.0 / n_types
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for b in range(bouchees.shape[1]):
        y = bouchees[:, b]
        if not np.isfinite(y).any():
            continue
        ax.plot(iters, y, lw=1.8, color=cmap(b / max(bouchees.shape[1] - 1, 1)),
                label=f"{b} bouchee" + ("s" if b != 1 else ""))
    ax.axhline(hasard, color="0.45", ls=":", lw=1.2)
    ax.annotate(f"hasard ({hasard:.2f})", (iters[0], hasard), xytext=(4, 5),
                textcoords="offset points", ha="left", fontsize=9, color="0.4")
    ax.set_xlabel("mises a jour de gradient")
    ax.set_ylabel("exactitude : quel canal est le plus nefaste ?")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(loc="lower right", fontsize=9, title="dans l'episode")
    noms = " ".join(f"{LABELS[r.id]} {r.delta_energy:+g}" for r in res)
    ax.set_title(f"probe_memory — modele {a.model}, carry {a.carry}, tete {list(tete)}"
                 f"\n{noms}")
    fig.tight_layout()
    sortie = os.path.join(a.fig_dir, f"probe_memory_{a.model}.png")
    fig.savefig(sortie, dpi=140)
    plt.close(fig)
    print(f"\nFigure saved: {sortie}")


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
    p.add_argument("--n-res", dest="n_res", type=int, default=len(BASE_RESOURCES),
                   help="nb de ressources, prises dans BASE_RESOURCES "
                        "(defaut %(default)s)")
    p.add_argument("--eval-tous", dest="eval_tous", type=int, default=25,
                   help="periode d'evaluation pour la figure (defaut %(default)s)")
    p.add_argument("--max-bouchees", dest="max_bouchees", type=int, default=8,
                   help="derniere courbe de la figure (defaut %(default)s)")
    p.add_argument("--fig-dir", dest="fig_dir", default="fig",
                   help="dossier de sortie de la figure (defaut %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    # la forme vient de MODEL_VERSIONS : la sonde doit tester TON reseau,
    # pas une variante codee en dur ici
    spec = MODEL_VERSIONS[a.model]
    mode = spec["memory_mode"]
    if a.carry is None:
        a.carry = spec["hidden_dim"]
    tete = tuple(a.hidden) if a.hidden is not None else tuple(spec["hidden_layers"])
    res = sous_ensemble(a.n_res)
    if len(res) < 2:
        raise SystemExit("Il faut au moins 2 ressources : avec une seule il n'y "
                         "a rien a distinguer.")
    de_par_id = jnp.array([r.delta_energy for r in res])
    n_types = len(res)
    print(f"modele {a.model} ({mode})  carry={a.carry}  tete={list(tete)}  "
          f"{n_types} ressources\n"
          f"delta_energy par identite : "
          + "  ".join(f"{LABELS[r.id]} {r.delta_energy:+g}" for r in res)
          + f"   (cible = la plus nefaste, hasard = {1/n_types:.3f})")

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

    def exactitude_par_bouchees(params, cle, n_ep=512):
        """(n_bouchees,) : exactitude selon le nombre de bouchees deja prises.

        C'est la variable qui compte : le reseau ne peut identifier le canal
        nefaste qu'apres l'avoir goute, ou avoir goute les autres. Le nombre de
        PAS ecoules ne dit rien -- un pas sans repas n'apporte aucune information.
        """
        lot = episodes(cle, n_ep, a.steps, de_par_id, n_types)
        logits = deroule(appliquer, params, lot, a.carry)
        juste = np.asarray(jnp.argmax(logits, -1) == lot[5][:, None])   # (E,T)
        # bouchees deja PRISES a l'instant ou la prediction est faite : celle du
        # pas courant n'est pas encore entree dans le LSTM
        mange = np.asarray(lot[4]).sum(-1)                              # (E,T)
        prises = np.cumsum(mange, axis=1) - mange
        out = np.full(a.max_bouchees + 1, np.nan)
        for b in range(a.max_bouchees + 1):
            m = prises == b
            if m.sum() >= 50:
                out[b] = juste[m].mean()
        return out

    print(f"\n{'iter':>6}{'loss':>9}{'exactitude':>12}"
          f"   (hasard = {1/n_types:.3f})")
    courbe_iters, courbe_bouchees = [], []
    for it in range(a.iters + 1):
        cle, ce = jax.random.split(cle)
        params, etat, l, acc = etape(params, etat, ce)
        if it % max(1, a.iters // 10) == 0:
            print(f"{it:>6}{float(l):>9.4f}{float(acc):>12.3f}")
        if it % a.eval_tous == 0:
            cle, cv = jax.random.split(cle)
            courbe_iters.append(it)
            courbe_bouchees.append(exactitude_par_bouchees(params, cv))

    # exactitude en fonction du nombre de pas ecoules dans l'episode
    cle, ct = jax.random.split(cle)
    lot = episodes(ct, 1024, a.steps, de_par_id, n_types)
    logits = deroule(appliquer, params, lot, a.carry)
    exact = (jnp.argmax(logits, -1) == lot[5][:, None])
    print("\nexactitude par pas de l'episode :")
    for t in range(0, a.steps, max(1, a.steps // 10)):
        print(f"  pas {t:>3} : {float(exact[:, t].mean()):.3f}")
    print(f"  pas {a.steps-1:>3} : {float(exact[:, -1].mean()):.3f}")

    trace(a, np.array(courbe_iters), np.array(courbe_bouchees), n_types,
          res, tete)


if __name__ == "__main__":
    main()
