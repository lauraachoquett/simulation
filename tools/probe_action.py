"""Le reseau PEUT-IL apprendre a se servir de v_pred pour choisir son action ?

    python -m simulation_meta.tools.probe_action

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
    carte    l'option value_map du reseau : un canal d'observation en plus, ou
             chaque case porte la valeur de ce qui s'y trouve. Le produit
             vision x valeur est fait avant le conv, la tete n'a plus qu'a
             suivre un champ scalaire (c'est ce que fait simulation/oracle.py).
             C'est le MEME code que la simulation, pas une imitation.
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

from simulation_meta.data_class import BASE_RESOURCES, LABELS, sous_ensemble


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

from EcoEvoJax_meta.source.agent import MetaRNN_bcppr


def echantillons(cle, n, cote, de, bons, mauvais):
    """Une ressource, juste devant. Faut-il avancer ?

    La tache est reduite a la seule decision qui nous interesse : la case devant
    porte le canal k, v_pred dit ce que vaut chaque canal, y aller ou non. Pas de
    navigation -- on avait etabli que se DIRIGER vers une case est un autre
    probleme ; ici on teste uniquement la lecture de la valeur.

    La classe est tiree d'abord, puis une identite compatible : le hasard vaut
    donc 0.5 quel que soit le nombre de ressources et la repartition des signes.
    Et la permutation change a chaque echantillon, donc un reseau qui ne lit pas
    v_pred ne peut pas depasser ce plancher.
    """
    n_types = de.shape[0]
    c_cls, c_bon, c_mauvais, c_perm = jax.random.split(cle, 4)
    perm = jnp.stack([jax.random.permutation(k, n_types)
                      for k in jax.random.split(c_perm, n)])       # canal -> identite
    v = de[perm]                                                   # (n, n_types)

    avancer = jax.random.uniform(c_cls, (n,)) < 0.5
    identite = jnp.where(
        avancer,
        bons[jax.random.randint(c_bon, (n,), 0, len(bons))],
        mauvais[jax.random.randint(c_mauvais, (n,), 0, len(mauvais))])
    canal = jnp.argmax(perm == identite[:, None], axis=1)          # ou il se trouve

    c = cote // 2
    obs = jnp.zeros((n, cote, cote, n_types + 2))
    obs = obs.at[jnp.arange(n), c + 1, c, canal].set(1.0)
    cible = jnp.where(avancer, 3, 1)                               # avancer / tourner
    return obs, v, cible


def entraine(nom, a, cle, de, bons, mauvais, avec_carte, aveugle):
    n_types = de.shape[0]
    modele = MetaRNN_bcppr(4, out_fn="categorical", hidden_layers=list(a.hidden),
                           encoder_in=False, encoder_layers=[], carry_size=a.carry,
                           memory_mode="separee", predict_value=n_types,
                           value_map=avec_carte)

    cle, c0 = jax.random.split(cle)
    obs0, v0, _ = echantillons(c0, 1, a.cote, de, bons, mauvais)
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
        exact = (jnp.argmax(lg, -1) == cible).mean()
        return optax.softmax_cross_entropy_with_integer_labels(lg, cible).mean(), exact

    opt = optax.adam(a.lr)
    etat = opt.init(params)

    @jax.jit
    def pas(params, etat, cle):
        obs, v, cible = echantillons(cle, a.batch, a.cote, de, bons, mauvais)
        (l, ex), g = jax.value_and_grad(perte, has_aux=True)(params, (obs, v, cible))
        maj, etat = opt.update(g, etat, params)
        return optax.apply_updates(params, maj), etat, l, ex

    # Courbe d'apprentissage : combien de mises a jour pour atteindre quel
    # niveau. Le lot etant tire neuf a chaque pas, l'exactitude mesuree dessus
    # n'est pas biaisee -- le reseau ne l'a jamais vu.
    courbe = []
    for it in range(a.iters):
        cle, ce = jax.random.split(cle)
        params, etat, l, ex = pas(params, etat, ce)
        courbe.append(float(ex))

    # evaluation sur un lot neuf, plus grand
    cle, ct = jax.random.split(cle)
    obs, v, cible = echantillons(ct, 4096, a.cote, de, bons, mauvais)
    pred = jnp.argmax(logits(params, obs, v), -1)
    exact = float((pred == cible).mean())

    # le cas qui compte : du poison devant. Y avancer est l'erreur couteuse.
    poison = cible == 1
    fonce = float((pred[poison] == 3).mean())
    # premiere mise a jour ou l'exactitude lissee franchit chaque seuil
    lisse = np.convolve(np.array(courbe), np.ones(20) / 20, mode="valid")
    franchit = {}
    for seuil in (0.75, 0.90, 0.95, 0.99):
        au_dessus = np.flatnonzero(lisse >= seuil)
        franchit[seuil] = int(au_dessus[0]) + 20 if au_dessus.size else None
    return dict(nom=nom, params=n_par, perte=float(l), exact=exact,
                fonce=fonce, n=int(poison.sum()), franchit=franchit,
                courbe=courbe)


def trace(a, sorties, res):
    """Exactitude en fonction des mises a jour, une courbe par variante.

    C'est le resultat central : la variante `aveugle` plafonne au hasard, les
    deux autres atteignent 1.0 -- l'architecture SAIT exploiter v_pred. Ce qui
    les separe est la vitesse, et c'est ce que l'ecart des courbes montre.
    """
    os.makedirs(a.fig_dir, exist_ok=True)
    couleurs = {"aveugle": "#8D99A6", "concat": "#1D3557", "carte": "#2A7F31"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for r in sorties:
        # lisse sur 20 iterations : le lot est neuf a chaque pas, donc la courbe
        # brute est bruitee sans que le reseau ait change
        y = np.convolve(np.array(r["courbe"]), np.ones(20) / 20, mode="valid")
        x = np.arange(len(y)) + 20
        ax.plot(x, y, lw=1.9, color=couleurs.get(r["nom"], "0.4"), label=r["nom"])
        f = r["franchit"].get(0.99)
        if f:
            ax.plot([f], [0.99], "o", ms=6, color=couleurs.get(r["nom"], "0.4"))
            ax.annotate(f"99 % en {f}", (f, 0.99), xytext=(8, -12),
                        textcoords="offset points", fontsize=9,
                        color=couleurs.get(r["nom"], "0.4"))
    ax.axhline(0.5, color="0.45", ls=":", lw=1.2)
    ax.annotate("hasard (0.50)", (len(sorties[0]["courbe"]), 0.5), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=9, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("mises a jour de gradient (echelle log)")
    ax.set_ylabel("exactitude : avancer ou tourner ?")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=.3, which="both")
    ax.legend(loc="lower right", fontsize=9, title="v_pred donne a la tete")
    noms = " ".join(f"{LABELS[r.id]} {r.delta_energy:+g}" for r in res)
    ax.set_title(f"probe_action — tete {list(a.hidden)}, carry {a.carry}, "
                 f"lot {a.batch}\n{noms}")
    fig.tight_layout()
    sortie = os.path.join(a.fig_dir, "probe_action.png")
    fig.savefig(sortie, dpi=140)
    plt.close(fig)
    print(f"\nFigure saved: {sortie}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hidden", type=int, nargs="+", default=[8],
                   help="tete de politique (defaut %(default)s)")
    p.add_argument("--carry", type=int, default=8)
    p.add_argument("--cote", type=int, default=11, help="cote de la vue")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-res", dest="n_res", type=int, default=len(BASE_RESOURCES),
                   help="nb de ressources, prises dans BASE_RESOURCES "
                        "(defaut %(default)s)")
    p.add_argument("--fig-dir", dest="fig_dir", default="fig",
                   help="dossier de sortie de la figure (defaut %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    res = sous_ensemble(a.n_res)
    de = jnp.array([r.delta_energy for r in res])
    n_types = len(res)
    bons = np.flatnonzero(np.asarray(de) > 0)
    mauvais = np.flatnonzero(np.asarray(de) < 0)
    if not (len(bons) and len(mauvais)):
        raise SystemExit("Il faut au moins une ressource benefique et une "
                         "nefaste : sans les deux la decision n'existe pas.")
    bons, mauvais = jnp.asarray(bons), jnp.asarray(mauvais)
    print(f"tete={a.hidden} carry={a.carry} | {a.iters} iterations, "
          f"lot={a.batch} | {n_types} ressources\n"
          "valeurs par identite : "
          + "  ".join(f"{LABELS[r.id]} {r.delta_energy:+g}" for r in res))

    cle = jax.random.PRNGKey(a.seed)
    sorties = []
    for nom, carte, aveugle in (("aveugle", False, True),
                                ("concat", False, False),
                                ("carte", True, False)):
        cle, ce = jax.random.split(cle)
        sorties.append(entraine(nom, a, ce, de, bons, mauvais, carte, aveugle))
        print(f"  [{nom}] termine")

    print(f"\n  {'variante':<10}{'params':>8}{'perte':>9}{'exactitude':>13}"
          f"{'fonce sur nefaste':>19}")
    for r in sorties:
        print(f"  {r['nom']:<10}{r['params']:>8}{r['perte']:>9.4f}"
              f"{r['exact']:>13.3f}{r['fonce']:>19.3f}")
    print(f"  (hasard et plafond aveugle : exactitude 0.50, fonce sur nefaste "
          f"0.50 ; n={sorties[0]['n']} cas nefastes sur 4096)")

    print(f"\n  mises a jour pour atteindre :  (lot de {a.batch}, "
          f"donc x{a.batch} exemples etiquetes)")
    print(f"  {'variante':<10}" + "".join(f"{f'{100*s:.0f}%':>10}"
                                          for s in (0.75, 0.90, 0.95, 0.99)))
    for r in sorties:
        cases = "".join(f"{(r['franchit'][s] if r['franchit'][s] else '-'):>10}"
                        for s in (0.75, 0.90, 0.95, 0.99))
        print(f"  {r['nom']:<10}{cases}")

    aveugle, concat, carte = (r["exact"] for r in sorties)
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

    trace(a, sorties, res)


if __name__ == "__main__":
    main()
