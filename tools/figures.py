"""Toutes les figures d'analyse, sur une ou plusieurs experiences.

    python -m simulation.tools.figures exp/2026-09-12/*/fusion
    python -m simulation.tools.figures exp/A/fusion exp/B/fusion --pas 500000 --pas-chunks 100

Chaque outil est lance dans son propre processus : un echec n'arrete pas les
autres, et le bilan est affiche a la fin.
"""
import argparse
import os
import subprocess
import sys


def outils(source, a):
    """[(nom, argv)] a lancer sur `source`."""
    lab = [d for d in (os.path.join(source, "replay", "lab_data"),
                       os.path.join(source, "lab_data")) if os.path.isdir(d)]
    zoom = ["--chunks-zoom", *map(str, a.chunks)] if a.chunks else []
    pas_c = ["--pas-chunks", str(a.pas_chunks)] if a.pas_chunks else []
    liste = []
    if lab:
        liste += [
            ("lab metrics", ["replay_lab", source, "--plot-only", *zoom, *pas_c]),
            ("geometries alone", ["plot_geometries", source, *pas_c]
             + (["--chunks", *map(str, a.chunks)] if a.chunks else [])),
            ("geometries clones", ["plot_geometries", source, "--condition",
                                   "clones", *pas_c] + (["--chunks", *map(str, a.chunks)] if a.chunks else [])),
            ("geometries figurants", ["plot_geometries", source, "--condition",
                                      "figurants", *pas_c] + (["--chunks", *map(str, a.chunks)] if a.chunks else [])),
            ("conditions par env", ["plot_geometries", source, "--par-env",
                                    *pas_c] + (["--chunks", *map(str, a.chunks)] if a.chunks else [])),
            ("repro vs age", ["plot_repro_vs_age", source, "--pas", str(a.pas)]),
            ("greediness vs voisins",
             ["plot_greed_vs_voisins", source, "--pas", str(a.pas)]),
        ]
    if a.replot and os.path.isdir(os.path.join(source, "data")):
        liste.append(("series du run", ["replot", source]
                      + (["--chunks", *map(str, a.chunks)] if a.chunks else [])))
    return liste


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sources", nargs="+", metavar="DIR",
                   help="dossiers d'experience, de replay ou de fusion")
    p.add_argument("--pas", type=int, default=500_000,
                   help="largeur des vignettes en pas (defaut %(default)s)")
    p.add_argument("--pas-chunks", dest="pas_chunks", type=int, default=100,
                   metavar="N", help="un point tous les N chunks (defaut %(default)s)")
    p.add_argument("--chunks", type=int, nargs=2, default=None,
                   metavar=("DEBUT", "FIN"), help="restreindre a une plage")
    p.add_argument("--replot", action="store_true",
                   help="ajouter les series du run (demande data/, donc le cluster)")
    a = p.parse_args()

    bilan = []
    for source in a.sources:
        print(f"\n=== {source} ===", flush=True)
        for nom, argv in outils(source, a):
            r = subprocess.run([sys.executable, "-m", f"simulation.tools.{argv[0]}",
                                *argv[1:]])
            bilan.append((source, nom, r.returncode == 0))

    print("\nBilan :")
    for source, nom, ok in bilan:
        print(f"  {'OK  ' if ok else 'ECHEC'} {os.path.basename(os.path.normpath(source))} — {nom}")
    if not all(ok for _, _, ok in bilan):
        sys.exit(1)


if __name__ == "__main__":
    main()
