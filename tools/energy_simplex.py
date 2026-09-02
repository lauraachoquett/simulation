"""Esperance d'energie d'un regime, sur le simplex. Hors simulation.

    python -m simulation_meta.tools.energy_simplex [sortie.png]

Ne lit que BASE_RESOURCES : la figure suit automatiquement les delta_energy
de data_class, il n'y a rien a tenir a jour ici.
"""
import sys

from simulation_meta.data_class import BASE_RESOURCES
from simulation_meta.utils.plots import plot_energy_expectation


if __name__ == "__main__":
    sortie = sys.argv[1] if len(sys.argv) > 1 else "energy_simplex.png"
    plot_energy_expectation(BASE_RESOURCES, sortie)
