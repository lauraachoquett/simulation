"""La politique se sert-elle de v_pred ?

    python -m simulation.tools.probe_vpred --exp DIR --chunk 100

On prend les genomes vivants d'un checkpoint et on les rejoue dans l'env de lab
sous trois conditions qui ne different QUE par le contenu des trois nombres que
la tete de politique recoit :

    appris  : la tete de valeur, telle que le run l'a apprise
    vrai    : les delta_energy exacts, imposes
    fort    : les memes, multiplies par 10
    inverse : les memes x -10, donc le poison devient la meilleure case
    nul     : des zeros, aucune information

Le bras "fort" existe parce qu'un resultat NUL n'a de valeur que si l'instrument
sait detecter un effet. Il separe deux causes que "vrai == nul" confond :

    fort == nul   -> la tete n'ecoute pas du tout cette entree, ses poids y sont
                     nuls. Aucune amplification n'y changera rien.
    fort != nul   -> elle l'ecoute, mais trop faiblement pour peser sur l'action.
                     Le cablage existe, il est seulement sous-dimensionne.

Le bras "inverse" tranche une objection sur "fort" : une entree dix fois plus
grande sur trois dimensions PERTURBE forcement la politique, qu'elle soit lue ou
non. Ce qui prouve une lecture, c'est que le comportement suive le SENS des
valeurs. Si en annoncant le poison comme la meilleure ressource on en fait manger
davantage, alors la tete lit vraiment ce nombre. Sinon "fort" n'etait qu'un choc.

Lecture :

  - vrai >> nul  -> la politique SAIT lire v_pred. Ce qui manque, c'est
                    l'apprentissage : fenetre, forget bias, ou heritage.
  - vrai ~= nul  -> la politique de CETTE POPULATION ignore v_pred.

Attention a la seconde lecture : elle ne dit pas que l'architecture en est
incapable. Si v_pred n'a jamais porte d'information utile pendant le run,
l'evolution n'avait aucune raison de lui donner du poids -- et un poids nul rend
le signal invisible, donc inutile. Les deux boucles ne s'amorcent pas.

Pour trancher entre "pas encore evolue" et "inevoluable", il faut faire evoluer
une population avec v_pred force aux vraies valeurs EN PERMANENCE, et voir si la
politique finit par s'en servir.

Ne modifie rien : lit un checkpoint, ne touche ni au run ni aux fichiers.
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from simulation.data_class import LABELS
from simulation.lab_env import vmap_over_agents_env_lab_high_res
from simulation.utils.plots import _test_signes
from simulation.utils.utils_sim import load_config, load_checkpoint
from simulation.run import build_model


def mesures(outputs, canal_poison):
    """(B,) x2 : pas-agents vecus, et bouchees de poison par 1000 pas-agents.

    On somme sur TOUS les emplacements : l'env high_res autorise la
    reproduction, donc la descendance compte comme du succes.
    """
    vivant = np.asarray(outputs.alive)                    # (B, T, N)
    pas_vecus = vivant.sum(axis=(1, 2)).astype(float)
    poison = (np.asarray(outputs.ate_res)[..., canal_poison]
              * vivant).sum(axis=(1, 2)).astype(float)
    return pas_vecus, 1000.0 * poison / np.maximum(pas_vecus, 1.0)


def ligne(nom, v, ref=None):
    med = np.median(v)
    q1, q3 = np.percentile(v, [25, 75])
    s = f"  {nom:<10}{med:>10.1f}  [{q1:>8.1f}, {q3:>8.1f}]"
    if ref is not None:
        d = v - ref
        k, m, p = _test_signes(d)
        s += f"{np.median(d):>12.1f}{k:>7}/{m:<5}{p:>9.3f}"
        if m == 0:
            s += "   <-- AUCUNE paire ne differe"
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True, help="dossier d'experience")
    p.add_argument("--chunk", type=int, required=True, help="chunk du checkpoint")
    p.add_argument("-n", type=int, default=64, help="genomes testes (defaut %(default)s)")
    p.add_argument("--gain", type=float, default=10.0,
                   help="facteur du bras 'fort' (defaut %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfg, _ = load_config(a.exp)
    # la tete existe des que l'un des deux flags la demande. Sur un run
    # --vpred-oracle le bras "appris" n'a pas de sens (la tete est
    # court-circuitee et n'a jamais ete entrainee) : seuls vrai vs nul comptent.
    if not (cfg.inner_loop or cfg.vpred_oracle):
        raise SystemExit("Ce run n'a pas de tete de valeur : il n'y a pas de "
                         "v_pred a sonder.")
    bras = (("appris", ()),) if cfg.inner_loop else ()

    state = load_checkpoint(a.exp, a.chunk)
    vivants = np.asarray(state.agents.alive) > 0
    idx = np.nonzero(vivants)[0]
    if idx.size == 0:
        raise SystemExit("Aucun agent vivant dans ce checkpoint.")
    tirage = np.random.default_rng(a.seed).choice(idx, size=min(a.n, idx.size),
                                                  replace=False)
    genomes = jnp.asarray(np.asarray(state.agents.params)[tirage])

    # valeurs vraies DANS L'ORDRE DES CANAUX du run : cfg.resources a pu etre
    # permute par les shuffles, les lire ailleurs donnerait un oracle faux.
    vraies = tuple(float(r.delta_energy) for r in cfg.resources)
    canal_poison = next(i for i, r in enumerate(cfg.resources)
                        if LABELS[r.id] == "poison")
    print(f"checkpoint {a.exp} chunk {a.chunk} | {len(tirage)} genomes tires "
          f"parmi {idx.size} vivants")
    print("valeurs par canal : "
          + "  ".join(f"c{i}={LABELS[r.id]} {r.delta_energy:+g}"
                      for i, r in enumerate(cfg.resources)))

    # memes cles pour les trois bras : l'ecart ne vient que de v_pred
    cle = jax.random.PRNGKey(a.seed)
    k_env, k_sim = jax.random.split(cle)
    k_sims = jax.random.split(k_sim, len(tirage))
    cfg_lab = cfg._replace(log_grid=False, log_obs=False)

    res = {}
    fort = tuple(a.gain * v for v in vraies)
    inverse = tuple(-v for v in fort)
    for nom, forcee in bras + (("vrai", vraies), ("fort", fort),
                               ("inverse", inverse),
                               ("nul", (0.0,) * len(cfg.resources))):
        modele = build_model(cfg, valeur_forcee=forcee)
        _, out = vmap_over_agents_env_lab_high_res(genomes, k_env, k_sims,
                                                   modele, cfg_lab)
        res[nom] = mesures(out, canal_poison)
        print(f"  [{nom}] termine")

    for j, (titre, unite) in enumerate([("PAS-AGENTS VECUS", "plus = mieux"),
                                        ("POISON pour 1000 pas", "moins = mieux")]):
        print(f"\n--- {titre} ({unite}) ---")
        print(f"  {'bras':<10}{'mediane':>10}  {'[q25, q75]':>20}"
              f"{'Δ vs nul':>12}{'signes':>13}{'p':>9}")
        nul = res["nul"][j]
        print(ligne("nul", nul))
        for nom in [n for n, _ in bras] + ["vrai", "fort", "inverse"]:
            print(ligne(nom, res[nom][j], ref=nul))

    # Zero paire discordante n'est pas un resultat nul : c'est le signe que les
    # deux bras ont tourne le MEME reseau. Un effet reellement nul laisse quand
    # meme du bruit de trajectoire, donc des paires qui different.
    if (np.array_equal(res["vrai"][0], res["nul"][0])
            and np.array_equal(res["fort"][0], res["nul"][0])):
        raise SystemExit(
            "\nERREUR : les bras 'vrai' et 'nul' donnent des trajectoires\n"
            "identiques au bit pres sur les " + str(len(tirage)) + " genomes.\n"
            "Ce n'est pas un effet nul, c'est que le forcage n'atteint pas le\n"
            "reseau -- un effet vraiment nul laisserait du bruit de trajectoire.\n"
            "Verifier que rien ne remplace le modele en aval (cf. model_tourne).")

    d_vrai = np.median(res["vrai"][0] - res["nul"][0])
    d_fort = np.median(res["fort"][0] - res["nul"][0])
    print(f"\nLecture : information parfaite vaut {d_vrai:+.1f} pas-agents, "
          f"amplifiee x{a.gain:g} elle en vaut {d_fort:+.1f}.")
    if bras:
        d_appris = np.median(res["appris"][0] - res["nul"][0])
        print(f"          l'information apprise en vaut {d_appris:+.1f}.")
    # `nul` sortait de la boucle d'affichage ci-dessus et valait donc le POISON,
    # pas les pas-agents : le seuil etait 25x trop bas et validait n'importe quoi.
    seuil = 0.02 * np.median(res["nul"][0])
    # le poison est la mesure DIRIGEE : c'est elle qui dit si le comportement
    # suit le sens des valeurs, et pas seulement qu'il a ete perturbe
    poison_fort = np.median(res["fort"][1] - res["nul"][1])
    poison_inv = np.median(res["inverse"][1] - res["nul"][1])
    print(f"          poison : {poison_fort:+.1f} avec fort, "
          f"{poison_inv:+.1f} avec inverse (pour 1000 pas)")

    if abs(d_vrai) >= seuil:
        print("-> la politique lit v_pred. Ce qui manque est l'apprentissage, "
              "pas le cablage.")
    elif abs(d_fort) >= seuil and poison_fort < 0 < poison_inv:
        print("-> la politique lit v_pred, mais trop faiblement pour que "
              "l'information pese\n"
              "   sur l'action. Amplifiee elle evite le poison, inversee elle "
              "le recherche :\n"
              "   le comportement suit le SENS des valeurs, donc la lecture est "
              "reelle.\n"
              "   Le cablage existe et il est sous-dimensionne -- probleme "
              "d'echelle, pas d'architecture.")
    elif abs(d_fort) >= seuil:
        print("-> amplifiee, l'information change le comportement, mais celui-ci "
              "ne suit pas le\n"
              "   SENS des valeurs (le bras inverse devrait faire manger PLUS de "
              "poison).\n"
              "   C'est un choc sur l'entree, pas une lecture. A interpreter "
              "avec prudence.")
    else:
        print("-> cette population n'ecoute pas du tout v_pred : meme amplifiee "
              f"x{a.gain:g},\n"
              "   l'information ne change rien. Les poids de la tete vers cette "
              "entree sont nuls.")


if __name__ == "__main__":
    main()
