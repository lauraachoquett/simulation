"""Le reseau PEUT-IL apprendre a se servir de v_pred pour choisir son action ?

    python -m simulation.tools.probe_action

On sort de l'evolution. Le reseau est entraine par descente de gradient a imiter
l'oracle : meme vue, meme v_pred, et pour cible l'action que l'oracle prendrait.
Pas de mutation, pas de derive, pas de bruit de reproduction -- seulement la
question de l'EXPRESSIVITE.

  - le gradient trouve  -> l'architecture sait le faire. Si l'evolution n'y
                           arrive pas, c'est un probleme de recherche : mutation,
                           echelle du signal, pression de selection.
  - le gradient echoue  -> aucune methode ne trouvera ces poids. C'est
                           l'architecture qu'il faut reprendre.

La valeur des canaux CHANGE a chaque echantillon (permutation tiree au hasard),
donc le reseau ne peut pas la ranger dans ses poids : il doit lire v_pred. C'est
le meme dispositif que probe_memory pour la memoire.

Variantes comparees :

    aveugle  v_pred remplace par des zeros -- le plancher, ce qu'on fait sans
             l'information. Toute variante qui ne le bat pas n'a rien appris.
    concat   le cablage actuel : v_pred concatene a l'entree de la tete
    carte    un canal d'observation en plus, ou chaque case porte la valeur de
             ce qui s'y trouve : carte[i,j] = somme_k obs[i,j,k] * v_pred[k].
             Le produit vision x valeur est fait en amont, la tete n'a plus qu'a
             suivre un champ scalaire (c'est ce que fait simulation/oracle.py).
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from simulation.data_class import BASE_RESOURCES, LABELS

# Contournement LOCAL : flax 0.6.11 n'accepte pas LSTMCell(features=...).
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


def echantillons(cle, n, cote, n_types=3):
    """Une ressource, juste devant. Faut-il avancer ?

    La tache est reduite a la seule decision qui nous interesse : la case devant
    porte le canal k, v_pred dit ce que vaut chaque canal, y aller ou non. Pas de
    navigation -- on avait etabli que se DIRIGER vers une case est un autre
    probleme ; ici on teste uniquement la lecture de la valeur.

    Le poison sort une fois sur deux, donc le hasard vaut 0.5. Et la permutation
    change a chaque echantillon : un reseau qui ne lit pas v_pred ne peut pas
    depasser ce plancher, la reponse n'etant pas dans l'indice du canal.
    """
    c_id, c_autre, c_perm = jax.random.split(cle, 3)
    de = jnp.array([r.delta_energy for r in BASE_RESOURCES])       # par identite
    i_poison = LABELS.index("poison")

    perm = jnp.stack([jax.random.permutation(k, n_types)
                      for k in jax.random.split(c_perm, n)])       # canal -> identite
    v = de[perm]                                                   # (n, n_types)

    autre = jax.random.randint(c_autre, (n,), 0, n_types - 1)
    autre = jnp.where(autre >= i_poison, autre + 1, autre)         # exclut le poison
    identite = jnp.where(jax.random.uniform(c_id, (n,)) < 0.5, i_poison, autre)
    canal = jnp.argmax(perm == identite[:, None], axis=1)          # ou il se trouve

    c = cote // 2
    obs = jnp.zeros((n, cote, cote, 5))
    obs = obs.at[jnp.arange(n), c + 1, c, canal].set(1.0)
    cible = jnp.where(de[identite] > 0, 3, 1)                      # avancer / tourner
    return obs, v, cible


def ajoute_carte(obs, v, n_types=3):
    """Canal supplementaire : la valeur de ce qui occupe chaque case.

    Ajoute EN DERNIER pour ne pas decaler l'indice du canal des murs.
    """
    carte = (obs[..., :n_types] * v[:, None, None, :]).sum(-1, keepdims=True)
    return jnp.concatenate([obs, carte], axis=-1)


def entraine(nom, a, cle, avec_carte, aveugle):
    n_types = 3
    n_can = 5 + (1 if avec_carte else 0)
    modele = MetaRNN_bcppr(4, out_fn="categorical", hidden_layers=list(a.hidden),
                           encoder_in=False, encoder_layers=[], carry_size=a.carry,
                           memory_mode="separee", predict_value=n_types)

    cle, c0 = jax.random.split(cle)
    obs0, v0, _ = echantillons(c0, 1, a.cote)
    if avec_carte:
        obs0 = ajoute_carte(obs0, v0)
    z = jnp.zeros(a.carry)
    params = modele.init(jax.random.PRNGKey(a.seed), z, z, obs0[0],
                         jnp.zeros(4), jnp.zeros(1), jnp.ones(1),
                         jnp.zeros(n_types), v0[0])
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))

    def logits(params, obs, v):
        vv = jnp.zeros_like(v) if aveugle else v
        f = lambda o, x: modele.apply(params, z, z, o, jnp.zeros(4), jnp.zeros(1),
                                      jnp.ones(1), jnp.zeros(n_types), x)[2]
        return jax.vmap(f)(obs, vv)

    def perte(params, lot):
        obs, v, cible = lot
        lg = logits(params, obs, v)
        return optax.softmax_cross_entropy_with_integer_labels(lg, cible).mean()

    opt = optax.adam(a.lr)
    etat = opt.init(params)

    @jax.jit
    def pas(params, etat, cle):
        obs, v, cible = echantillons(cle, a.batch, a.cote)
        if avec_carte:
            obs = ajoute_carte(obs, v)
        l, g = jax.value_and_grad(perte)(params, (obs, v, cible))
        maj, etat = opt.update(g, etat, params)
        return optax.apply_updates(params, maj), etat, l

    for it in range(a.iters):
        cle, ce = jax.random.split(cle)
        params, etat, l = pas(params, etat, ce)

    # evaluation sur un lot neuf, plus grand
    cle, ct = jax.random.split(cle)
    obs, v, cible = echantillons(ct, 4096, a.cote)
    obs_in = ajoute_carte(obs, v) if avec_carte else obs
    pred = jnp.argmax(logits(params, obs_in, v), -1)
    exact = float((pred == cible).mean())

    # le cas qui compte : du poison devant. Y avancer est l'erreur couteuse.
    poison = cible == 1
    fonce = float((pred[poison] == 3).mean())
    return dict(nom=nom, params=n_par, perte=float(l), exact=exact,
                fonce=fonce, n=int(poison.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hidden", type=int, nargs="+", default=[8],
                   help="tete de politique (defaut %(default)s)")
    p.add_argument("--carry", type=int, default=8)
    p.add_argument("--cote", type=int, default=11, help="cote de la vue")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    print(f"tete={a.hidden} carry={a.carry} | {a.iters} iterations, "
          f"lot={a.batch}\n"
          "valeurs par identite : "
          + "  ".join(f"{LABELS[r.id]} {r.delta_energy:+g}" for r in BASE_RESOURCES))

    cle = jax.random.PRNGKey(a.seed)
    res = []
    for nom, carte, aveugle in (("aveugle", False, True),
                                ("concat", False, False),
                                ("carte", True, False)):
        cle, ce = jax.random.split(cle)
        res.append(entraine(nom, a, ce, carte, aveugle))
        print(f"  [{nom}] termine")

    print(f"\n  {'variante':<10}{'params':>8}{'perte':>9}{'exactitude':>13}"
          f"{'fonce sur poison':>19}")
    for r in res:
        print(f"  {r['nom']:<10}{r['params']:>8}{r['perte']:>9.4f}"
              f"{r['exact']:>13.3f}{r['fonce']:>19.3f}")
    print(f"  (hasard et plafond aveugle : exactitude 0.50, fonce sur poison "
          f"0.50 ; n={res[0]['n']} cas de poison sur 4096)")

    aveugle, concat, carte = (r["exact"] for r in res)
    print()
    if concat > aveugle + 0.05:
        print("-> le cablage actuel SAIT se servir de v_pred : le gradient trouve "
              "les poids.\n"
              "   Si l'evolution n'y arrive pas, c'est la recherche qu'il faut "
              "corriger\n"
              "   (mutation, echelle du signal), pas l'architecture.")
    elif carte > aveugle + 0.05:
        print("-> le cablage actuel n'y arrive pas, la carte de valeur si. "
              "Le produit\n"
              "   vision x valeur est bien le point bloquant, et le faire en "
              "amont le leve.")
    else:
        print("-> ni l'un ni l'autre ne bat le plancher aveugle. Revoir la tache "
              "ou les\n"
              "   reglages d'entrainement avant de conclure sur l'architecture.")


if __name__ == "__main__":
    main()
