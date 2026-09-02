
"""Généalogie et ancêtre commun (MRCA).

Construit incrémentalement l'arbre (node_parent / node_children), détecte la
coalescence de la population (tous les vivants descendent d'un même ancêtre)
et déclenche les tracés associés (TMRCA, PCA de clade).

Couplage : update_genealogy appelle self.update_weight_metrics (WeightsMixin)
quand la population est coalescée.
"""

import os
import numpy as np

import jax.numpy as jnp
from jax import random

from simulation.genealogy.genealogy import update_genealogy, chaine_ancetres
from simulation.genealogy.pca import save_alive_snapshot, load_clade_snapshots
from simulation.genealogy.r0 import plot_r0, r0_by_birth_window
from simulation.genealogy.mrca import coalescence_point, plot_tmrca_gen
from simulation.lab_env import vmap_over_agents_env_lab_high_res
from simulation.utils.plots import plot_lineage_simplex
from simulation.utils.utils_sim import build_id_timeline, load_shuffle_log


class GenealogyMixin:

    def _init_genealogy(self):
        self.node_parent = {}
        self.node_children = {}
        self.tmrca_gen = []
        self.prev_born = None
        self.prev_parent = None
        self.coalesced = False

    def update_genealogy(self, outputs,state, exp_dir):
        self.prev_born, self.prev_parent = update_genealogy(
            outputs, self.node_parent, self.node_children,
            self.prev_born, self.prev_parent)   # arbre complet, pas cher
        
        if self.cfg.track_weights:
            if self.coalesced:
                self.update_weight_metrics(state)      # -> WeightsMixin
            save_alive_snapshot(state, self.chunk_idx, os.path.join(exp_dir, 'params'))

    def update_mrca_and_plot(self, outputs, exp_dir):
        outputs_mcra = coalescence_point(outputs, self.node_parent)
        self.coalesced = outputs_mcra['coalesced']
        tmrca_generations = outputs_mcra['tmrca_generations']
        self.tmrca_gen.append(tmrca_generations)
        if self.coalesced:
            plot_tmrca_gen(np.concatenate(self.pop_history, axis=0), self.tmrca_gen, exp_dir)

    def plot_lineage_simplex(self, state, key_env, subkey_sim, model, exp_dir,
                             n_lignees=4, max_generations=80):
        """Trajectoire de quelques lignees dans le simplex des ressources.

        On tire des vivants au hasard, on remonte leurs ancetres, et on
        REEVALUE chaque genome au lab plutot que d'accumuler ce qu'il a mange
        dans le monde : on mesure ainsi un comportement, pas une circonstance
        (ce qui passait a portee, la concurrence, ou il avait marche).

        Chaque ancetre est evalue sous la configuration en vigueur A SA
        NAISSANCE. Un deplacement dans le triangle peut donc venir du genome ou
        de la permutation -- ce sont les losanges de la figure qui permettent de
        demeler les deux.
        """
        from simulation.simulation_data.lab import EVO_BATCH

        n_types = len(self.cfg.resources)
        if n_types != 3:
            print(f"[lineage] {n_types} ressources, il en faut 3 pour un simplex")
            return
        # Les deux messages nomment le drapeau plutot que le symptome :
        # track_weights commande save_alive_snapshot, donc l'existence meme des
        # genomes. C'est la cause dans la quasi-totalite des cas.
        if not self.cfg.track_weights:
            print("[lineage] track_weights=False : aucun genome n'est sauve, "
                  "relancer avec --weights pour obtenir la figure")
            return
        params_dir = os.path.join(exp_dir, "params")
        if not os.path.isdir(params_dir):
            print("[lineage] pas de dossier params/ : aucun genome a recharger, "
                  "relancer avec --weights")
            return

        survivants = self.compute_survivors(state)
        if not survivants:
            print("[lineage] aucun survivant, saute")
            return

        # tirage reproductible : deux lectures de la meme figure doivent porter
        # sur les memes lignees
        rng = np.random.default_rng(self.chunk_idx)
        tires = [survivants[i] for i in rng.choice(
            len(survivants), size=min(n_lignees, len(survivants)), replace=False)]

        chaines = [chaine_ancetres(v, self.node_parent, max_generations)[::-1]
                   for v in tires]

        # Les lignees coalescent : l'union est bien plus petite que la somme,
        # et un genome partage par deux chaines n'est evalue qu'une fois.
        union = {n for c in chaines for n in c}
        genomes = load_clade_snapshots(union, params_dir)
        if len(genomes) < 2:
            print(f"[lineage] {len(genomes)} genome(s) retrouve(s) sur "
                  f"{len(union)}, trop peu pour tracer")
            return

        # config a la naissance de chaque ancetre. shuffle_resources permute les
        # ResourceConfig ENTIERS, donc les parametres de croissance suivent
        # l'identite : cette table est une bijection stable sur tout le run.
        par_id = {r.id: r for r in self.cfg.resources}
        shuffle_log = load_shuffle_log(exp_dir)
        pas_shuffle = np.array([e["step"] for e in shuffle_log], dtype=np.int64)
        noeuds = sorted(genomes)
        naissances = np.array([b for _, b in noeuds], dtype=np.int64)
        ordres = build_id_timeline(naissances, shuffle_log, self.initial_order_ids)

        # Regroupe par ORDRE DE CANAUX, pas par epoque : il n'existe que 3! = 6
        # ordres possibles, donc au plus 6 configurations de lab et 6
        # compilations, quel que soit le nombre de permutations traversees.
        groupes = {}
        for i, n in enumerate(noeuds):
            groupes.setdefault(tuple(int(x) for x in ordres[i]), []).append(n)

        cfg_m = self.cfg._replace(log_grid=False)
        regime = {}
        for ordre, membres in groupes.items():
            cfg_g = cfg_m._replace(resources=tuple(par_id[i] for i in ordre))
            X = np.stack([genomes[n] for n in membres])
            for deb in range(0, len(membres), EVO_BATCH):
                lot = X[deb:deb + EVO_BATCH]
                subkey_sim, k = random.split(subkey_sim)
                _, out = vmap_over_agents_env_lab_high_res(
                    jnp.asarray(lot), key_env, random.split(k, len(lot)),
                    model, cfg_g)
                mange = self.eaten_by_type(out)               # (B, n_types) par CANAL
                for j, n in enumerate(membres[deb:deb + EVO_BATCH]):
                    # canal -> identite. LABELS etant (good, medium, poison),
                    # l'indice d'identite est deja l'ordre des sommets.
                    par_identite = np.zeros(n_types)
                    for k_canal, ident in enumerate(ordre):
                        par_identite[ident] = mange[j, k_canal]
                    regime[n] = par_identite

        # disponible sur la grille, par identite. Invariant d'une epoque a
        # l'autre puisque les parametres de croissance suivent l'identite : un
        # seul point de reference suffit pour toute la figure.
        subkey_sim, k_g = random.split(subkey_sim)
        _, out_g = vmap_over_agents_env_lab_high_res(
            jnp.asarray(genomes[noeuds[0]])[None], key_env,
            random.split(k_g, 1), model, self.cfg._replace(log_grid=True))
        dispo_canal = self.available_by_type(out_g, n_types)
        dispo = np.zeros(n_types)
        for k_canal, r in enumerate(self.cfg.resources):
            dispo[r.id] = dispo_canal[k_canal]

        # Mise en forme. Un ancetre sans genome retrouve, ou n'ayant rien mange
        # au lab, est ABSENT de la chaine : sa position est indefinie, pas
        # nulle. Le trait l'enjambe.
        sorties, gardes, total = [], 0, 0
        for chaine in chaines:
            points = []
            for n in chaine:
                total += 1
                r = regime.get(n)
                if r is None or r.sum() <= 0:
                    continue
                gardes += 1
                born = n[1]
                parent = self.node_parent.get(n)
                # premiere generation d'une nouvelle epoque : une permutation
                # tombe entre la naissance du parent et la sienne
                apres = bool(parent is not None and pas_shuffle.size
                             and ((pas_shuffle > parent[1]) & (pas_shuffle <= born)).any())
                points.append(dict(regime=r, born=born, post_shuffle=apres))
            if len(points) >= 2:
                sorties.append(points)

        plot_lineage_simplex(sorties, dispo, exp_dir, self.chunk_idx,
                             couverture=gardes / max(total, 1))

    def compute_survivors(self, state):
        alive_last = np.array(state.agents.alive)
        born_last  = np.array(state.agents.born_step)
        survivors  = [(i, int(born_last[i])) for i in range(1, len(alive_last))
                        if alive_last[i] == 1]
        return survivors

    def plot_pca(self, outputs, data_dir, exp_dir):
        return
        # survivors= self.compute_survivors(state)
        # if survivors:
        #     root  = find_root(survivors[0], self.node_parent)
        #     clade = collect_clade(root, self.node_children)
        #     name_save=list(np.arange(self.chunk_idx-(self.cfg.pca)//2,self.chunk_idx))
            #node_params = load_clade_snapshots(clade, os.path.join(exp_dir,'params'),name_save)
            #plot_clade_pca_html(node_params, os.path.join(exp_dir),name_fig=f'{self.chunk_idx}')

    def save_mrca_sim(self, data_dir):
        np.savez(
            os.path.join(data_dir, f"tmrca.npz"),
            tmrca=np.array(self.tmrca_gen),
        )
        
    def compute_R0_and_plot(self,state,current_step,exp_dir):
        survivors = self.compute_survivors(state)
        oldest_alive_birth = min(b for (_, b) in survivors) if survivors else current_step
        window = 2000
        
        r0_by_window=r0_by_birth_window(self.node_children, oldest_alive_birth, window=window)
        plot_r0(r0_by_window, window, exp_dir=exp_dir, smooth_w=5, fname="r0_evolution.png")