"""Démographie / écologie.

Accumule, chunk par chunk, les séries temporelles population / ressources /
mouvement / durée de vie, les sauve sur disque, et fournit la condition d'arrêt
de la simulation.
"""

import os
import numpy as np

from simulation.utils.plots import compute_mean_movement_chunk, compute_lifetime_chunk
from simulation.utils.utils_sim import classify_outcome

# W : pas apres l'observation ou la consommation compte encore. Doit rester egal
# a ENERGY_EAT_WINDOW (energy_response.py), sinon la sim principale et le lab ne
# mesurent plus la meme chose. = 2 * cfg.agent_view, cf. le commentaire la-bas.
EAT_WINDOW = 10

# Resolution de la courbe erreur-vs-age de la boucle interne.
AGE_BIN = 50
# Borne de depart, doublee des qu'un agent la depasse : une constante en dur
# tronquait la courbe et empilait tout le surplus dans le dernier casier, ce qui
# faisait un pic artificiel. Elle se cale donc sur la duree de vie reelle.
N_AGE_BINS = 240
# Plafond de securite : un agent qui ne meurt jamais ferait croitre les tableaux
# sans fin. Au-dela, le dernier casier redevient un fourre-tout -- mais il n'est
# jamais trace, donc il ne peut pas mentir.
MAX_AGE_BINS = 1000        # 1000 x AGE_BIN = 50 000 pas

# Largeur des cohortes de naissance pour le succes reproducteur.
COHORTE = 1000


def compute_seen_eaten_chunk(outputs, window=EAT_WINDOW):
    """(T, n_types) x2 : n = agents voyant le type k, k_ = ceux qui le mangent.

    On conditionne sur "type k dans le champ de vision au pas t" et on regarde
    s'il y a consommation dans [t, t+window] : voir une ressource et l'atteindre
    prend le temps d'y marcher, donc exiger la consommation au pas t sous-estime
    massivement la propension (meme fenetre que energy_response).

    Reduit des le chunk : garder (T, N, n_types) pour tout le run ferait
    2000 agents x 1000 pas x 3 types et par chunk."""
    saw = np.asarray(outputs.saw_res).astype(bool)      # (T, N, n_types)
    ate = np.asarray(outputs.ate_res).astype(bool)      # (T, N, n_types)

    ate_w = ate.copy()                                  # consommation dans [t, t+W]
    for d in range(1, window + 1):
        ate_w[:-d] |= ate[d:]

    n_seen  = saw.sum(axis=1)                           # (T, n_types)
    n_eaten = (saw & ate_w).sum(axis=1)                 # (T, n_types)
    return n_seen, n_eaten


class DemographyMixin:

    def _init_demography(self):
        self.pop_history = []
        self.res_history = []
        self.oracle_history = []
        self.consumed_history = []
        self.seen_history = []
        self.eaten_seen_history = []
        self.mov_history = []
        self.life_history = []
        self.perte_history = []
        # succes reproducteur : {(emplacement, born_step) -> nb d'enfants}. La
        # cle est le COUPLE, pas l'emplacement seul : les emplacements sont
        # reutilises a chaque naissance, parent_id ne designe donc pas un agent.
        self.enfants = {}
        # cohorte -> histogramme du nb d'enfants, une fois les agents morts
        self.enfants_par_cohorte = {}
        self.dernier_born = None      # derniere ligne du chunk precedent
        self.age_somme = []
        self.age_compte = []
        self.n_age_bins = N_AGE_BINS

    def update_data_with_chunk(self, outputs, data_dir,chunk_idx):
        self.chunk_idx = chunk_idx
        pop_chunk  = np.array(outputs.alive.sum(axis=1))          # (T,)
        # envahisseurs vivants, pour la courbe d'invasion
        oracle_chunk = np.array((outputs.alive * outputs.is_oracle).sum(axis=1))
        n_types = len(self.cfg.resources)
        res_chunk = np.array(outputs.grid[:, :n_types, :, :].sum(axis=(2, 3)))
        consumed_chunk = np.array(outputs.consumed_res)           # (T, n_types)
        seen_chunk, eaten_seen_chunk = compute_seen_eaten_chunk(outputs)
        # Erreur de prediction, moyennee sur les agents vivants qui ont mange
        # dans la fenetre. Les autres portent un NaN (cf. one_simulation) et sont
        # ecartes : les compter comme zero ferait croire a une prediction parfaite.
        if self.cfg.inner_loop:
            perte = np.asarray(outputs.perte_pred)             # (T, N)
            ok = (np.asarray(outputs.alive) > 0) & np.isfinite(perte)
            valides = np.where(ok, perte, np.nan)
            n_valides = np.isfinite(valides).sum(axis=1)
            perte_chunk = np.divide(np.nansum(valides, axis=1), n_valides,
                                    out=np.full(len(pop_chunk), np.nan),
                                    where=n_valides > 0)
            # Erreur en fonction de l'AGE de l'agent, accumulee en casiers.
            # C'est la seule vue qui montre l'apprentissage INTRA-VIE : la courbe
            # precedente melange nouveau-nes et agents murs, et ne peut donc pas
            # distinguer "il apprend pendant sa vie" de "il nait mieux equipe".
            age = np.asarray(outputs.step)[:, None] - np.asarray(outputs.born_step)
            self._elargir_casiers(int(age.max()) // AGE_BIN + 1)
            casier = np.clip(age // AGE_BIN, 0, self.n_age_bins - 1)
            somme_chunk = np.bincount(casier[ok], weights=perte[ok],
                                      minlength=self.n_age_bins)
            compte_chunk = np.bincount(casier[ok], minlength=self.n_age_bins)
        else:
            perte_chunk = np.zeros(len(pop_chunk))
            somme_chunk = compte_chunk = np.zeros(self.n_age_bins)
        mov_chunk  = compute_mean_movement_chunk(outputs, self.cfg.grid_length)
        life_chunk = compute_lifetime_chunk(outputs, self.cfg)

        self.pop_history.append(pop_chunk)
        self.res_history.append(res_chunk)
        self.oracle_history.append(oracle_chunk)
        self.consumed_history.append(consumed_chunk)
        self.seen_history.append(seen_chunk)
        self.eaten_seen_history.append(eaten_seen_chunk)
        self._suivre_descendance(outputs)
        self.perte_history.append(perte_chunk)
        self.age_somme.append(somme_chunk)
        self.age_compte.append(compte_chunk)
        self.mov_history.append(mov_chunk)
        self.life_history.append(life_chunk)

        np.savez(
            os.path.join(data_dir, f"chunk_{self.chunk_idx:05d}.npz"),
            population    = pop_chunk,
            resources     = res_chunk,
            oracles       = oracle_chunk,
            consumed      = consumed_chunk,
            n_seen        = seen_chunk,
            n_eaten_seen  = eaten_seen_chunk,
            perte_pred    = perte_chunk,
            perte_age_somme  = somme_chunk,
            perte_age_compte = compte_chunk,
            mean_movement = mov_chunk,
            mean_life     = life_chunk,
        )

    def _elargir_casiers(self, besoin):
        """Double la borne d'age jusqu'a couvrir `besoin`, et complete l'historique.

        Les tableaux deja accumules sont rallonges de zeros : ils doivent tous
        garder la meme longueur pour que le trace les empile.
        """
        if besoin <= self.n_age_bins or self.n_age_bins >= MAX_AGE_BINS:
            return
        neuf = self.n_age_bins
        while neuf < besoin and neuf < MAX_AGE_BINS:
            neuf *= 2
        neuf = min(neuf, MAX_AGE_BINS)
        pad = neuf - self.n_age_bins
        self.age_somme = [np.pad(a, (0, pad)) for a in self.age_somme]
        self.age_compte = [np.pad(a, (0, pad)) for a in self.age_compte]
        print(f"[age] borne portee a {neuf * AGE_BIN} pas "
              f"({neuf} casiers) : un agent l'a depassee")
        self.n_age_bins = neuf

    def _suivre_descendance(self, outputs):
        """Compte les enfants de chaque agent, et ferme les cohortes achevees.

        Un agent n'est comptabilise qu'une fois MORT : tant qu'il vit il peut
        encore se reproduire, et l'inclure sous-estimerait sa descendance. Les
        cohortes recentes restent donc incompletes, et le trace les ecarte.
        """
        born = np.asarray(outputs.born_step)          # (T, N)
        par = np.asarray(outputs.parent_id)
        alive = np.asarray(outputs.alive)

        if self.dernier_born is None:
            # Les FONDATEURS n'ont pas de naissance dans les logs : sans cette
            # inscription, seuls ceux qui se reproduisent existeraient (crees par
            # la branche parent), et la cohorte 0 perdrait tous ceux restes sans
            # descendance -- sa moyenne serait tiree vers le haut.
            for i in range(1, born.shape[1]):
                if alive[0, i] > 0:
                    self.enfants.setdefault((i, int(born[0, i])), 0)
        precedent = (self.dernier_born if self.dernier_born is not None
                     else born[0])
        lignes = np.vstack([precedent[None], born])   # (T+1, N)
        # une naissance = born_step qui change dans un emplacement. L'emplacement
        # 0 est sacrificiel (cf. final_alive_without_0) et recoit le bourrage des
        # index de naissance : on l'ignore.
        nouveaux = np.nonzero(lignes[1:] != lignes[:-1])
        for t, i in zip(*nouveaux):
            if i == 0:
                continue
            self.enfants.setdefault((int(i), int(born[t, i])), 0)
            p = int(par[t, i])
            if p != 0:
                # born_step du parent AU MOMENT de la naissance : c'est ce qui
                # identifie l'occupant de l'emplacement a cet instant-la
                cle = (p, int(born[t, p]))
                self.enfants[cle] = self.enfants.get(cle, 0) + 1

        # cloture : tout agent qui n'occupe plus son emplacement, ou qui l'occupe
        # mort, ne fera plus d'enfants
        vivants = {(i, int(born[-1, i])) for i in range(born.shape[1])
                   if alive[-1, i] > 0}
        for cle in [k for k in self.enfants if k not in vivants]:
            n = self.enfants.pop(cle)
            c = cle[1] // COHORTE
            h = self.enfants_par_cohorte.setdefault(c, np.zeros(1, dtype=np.int64))
            if n >= len(h):
                h = np.pad(h, (0, n + 1 - len(h)))
                self.enfants_par_cohorte[c] = h
            h[n] += 1

        self.dernier_born = born[-1]

    def cohortes_ouvertes(self):
        """cohorte -> nb d'agents encore vivants, dont la descendance n'est pas close."""
        d = {}
        for _, b in self.enfants:
            c = b // COHORTE
            d[c] = d.get(c, 0) + 1
        return d

    def check_end_condition(self):
        pop_full = np.concatenate(self.pop_history)
        res_full = np.concatenate(self.res_history)
        current_sim_state = classify_outcome(pop_full, res_full, self.cfg)
        return current_sim_state