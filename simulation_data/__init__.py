"""Accumulateur d'état d'une simulation de vie artificielle évolutive.

Découpage par thème :
    demography.py  -> séries pop / ressources / mouvement / durée de vie
    genealogy.py   -> arbre généalogique, MRCA, coalescence
    weights.py     -> dérive des poids du réseau vs ancêtre commun
    lab.py         -> test contrôlé des survivants (métriques de mort)
    core.py        -> assemblage + __init__ + plot (orchestrateur)
"""


__all__ = ["simulation_data"]