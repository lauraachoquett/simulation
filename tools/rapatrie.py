"""Copie en local la config et les figures d'experiences G5K, sans les donnees.

    python3 tools/rapatrie.py exp/2026-09-17/2026-09-17_17-05-33
    python3 tools/rapatrie.py 'exp/2026-09-17/*' --videos --sites lyon lille
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

SITES = ["grenoble", "lille", "luxembourg", "lyon", "nancy", "nantes",
         "rennes", "sophia", "strasbourg", "toulouse"]
FIGURES = ["config.json", "exp.json", "resource_shuffles.jsonl", "*.png", "*.html"]
VIDEOS = ["*.mp4", "*.gif"]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/rapatrie-%r@%h-%p",
       "-o", "ControlPersist=120"]


def cherche(site, base, motifs):
    """[(mtime, chemin relatif a base)] des experiences du site qui correspondent."""
    cmd = (f"cd {base} 2>/dev/null || exit 0; for d in {' '.join(motifs)}; do "
           f'[ -f "$d/config.json" ] && echo "$(stat -c %Y "$d/config.json") $d"; done')
    try:
        r = subprocess.run(SSH + [f"{site}.g5k", cmd], capture_output=True,
                           text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return []
    out = []
    for ligne in r.stdout.splitlines():
        t, _, d = ligne.partition(" ")
        if t.isdigit():
            out.append((int(t), d.rstrip("/")))
    return out


def fichiers(dossier):
    return {os.path.join(r, f) for r, _, fs in os.walk(dossier) for f in fs}


def rsync(site, base, rel, dest, motifs, *options):
    filtres = ["--include=*/"] + [f"--include={m}" for m in motifs] + ["--exclude=*"]
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["rsync", "-am", *options, "-e", " ".join(SSH), *filtres,
                    f"{site}.g5k:{base}/{rel}/", dest + "/"], check=False)


def reencode(mp4):
    tmp = mp4[:-4] + ".tmp.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
                        "-vf", "scale=trunc(iw/4)*2:-2", "-c:v", "libx264",
                        "-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p",
                        "-an", tmp])
    if r.returncode == 0 and os.path.getsize(tmp) < os.path.getsize(mp4):
        os.replace(tmp, mp4)
    elif os.path.exists(tmp):
        os.remove(tmp)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("motifs", nargs="+", help="exp/<date>/<heure> ou 'exp/<date>/*'")
    p.add_argument("--sites", nargs="+", default=SITES)
    p.add_argument("--distant", default="project/exp",
                   help="dossier des experiences sur les sites (defaut %(default)s)")
    p.add_argument("--dest", default=os.path.expanduser("~/Documents/BioTiC/exp_figs"))
    p.add_argument("--videos", action="store_true", help="copier et reencoder les videos")
    p.add_argument("--dry-run", action="store_true", help="lister sans copier")
    a = p.parse_args()

    motifs = [m[4:] if m.startswith("exp/") else m for m in a.motifs]
    print(f"recherche sur {len(a.sites)} site(s)...", flush=True)
    with ThreadPoolExecutor(len(a.sites)) as ex:
        trouves = dict(zip(a.sites, ex.map(lambda s: cherche(s, a.distant, motifs),
                                           a.sites)))

    choix = {}
    for site, lst in trouves.items():
        for t, rel in lst:
            if rel in choix:
                print(f"[info] {rel} sur {choix[rel][1]} et {site} : "
                      f"{site if t > choix[rel][0] else choix[rel][1]} garde (plus recent)")
            if rel not in choix or t > choix[rel][0]:
                choix[rel] = (t, site)
    if not choix:
        raise SystemExit("aucune experience trouvee (ssh <site>.g5k fonctionne ?)")

    for rel, (_, site) in sorted(choix.items()):
        dest = os.path.join(a.dest, rel)
        if a.dry_run:
            print(f"{site}: {rel}")
            continue
        avant = fichiers(dest)
        rsync(site, a.distant, rel, dest, FIGURES)
        if a.videos:
            rsync(site, a.distant, rel, dest, VIDEOS, "--ignore-existing")
            for f in sorted(fichiers(dest) - avant):
                if f.endswith(".mp4"):
                    reencode(f)
        print(f"{site}: {rel} ({len(fichiers(dest) - avant)} nouveau(x) fichier(s))",
              flush=True)
    if not a.dry_run:
        print(f"Dans {a.dest}")


if __name__ == "__main__":
    main()
