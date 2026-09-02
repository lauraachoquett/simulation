
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
os.environ["JAX_DONT_UNROLL_LOOPS"] = "1"

# A DEFINIR AVANT TOUT IMPORT DE JAX : lus une seule fois, a l'initialisation du
# backend. Sans eux, JAX prealloue ~75% de la carte d'un bloc -- 23,8 Gio sur
# un V100 32 Go -- et echoue des qu'un autre processus en detient une part, meme
# petite. L'allocateur asynchrone prend au contraire ce dont il a besoin.
#
# Verification qu'ils sont actifs : la ligne "maybe the environment variable
# 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation" doit
# DISPARAITRE des logs.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
from jax import random
import jax
from simulation_meta.one_simulation import run_simulation_chunk
from simulation_meta.utils.utils_sim import save_checkpoint,_video_worker,save_config,create_exp_file,load_config,load_checkpoint,outputs_to_numpy,video_payload,sec_to_minutes,shuffle_resources,shuffle_resources_v1
from simulation_meta.utils.plots import plot_current_config
from simulation_meta.data_class import Config, ResourceConfig, BASE_RESOURCES, LABELS, MODEL_VERSIONS, resolve_model
from EcoEvoJax_meta.source.agent import MetaRnnPolicy_bcppr
from simulation_meta.utils. utils_sim import init_state,load_checkpoint,save_checkpoint, log_resource_shuffle, masque_valeur, build_model, model_pour, periode_inner
from simulation_meta.simulation_data.core import simulation_data

import argparse
import multiprocessing as mp
import time
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

import os
### Force the same implementation for random numbers in jax 
jax.config.update("jax_threefry_partitionable", True)



def _hide_gpu():
    """Initializer pour les process vidéo (spawn) : ils encodent sur CPU via
    moviepy et n'ont aucun besoin du GPU. Sans ça, chaque worker ré-importe JAX
    et préalloue ~75 % de la mémoire GPU -> OOM du process de simulation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"


def save_script(exp_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Liste des fichiers à sauvegarder
    files_to_save = ["run.py", "update_env.py", "one_simulation.py",'agent_mov.py']

    for file_name in files_to_save:
        source_path = os.path.join(current_dir, file_name)
        destination_path = os.path.join(exp_dir, file_name)

        # Vérifie si le fichier existe avant de le copier
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
        else:
            print(
                f"Attention : Le fichier {source_path} n'a pas pu être trouvé."
            )

        


    
        
def launch_simulation_chunked(key, cfg, resume_exp=None, n_video_workers=2, chunk_id= 1 ,save_dir='', subkeys_init=None):
    
    start_time_sim = time.time()

    # UNE seule fois, et avant save_config : c'est la config RESOLUE qui part
    # dans config.json. La branche resume la recharge deja resolue et ne doit
    # donc pas repasser par la.
    cfg = resolve_model(cfg)

    if save_dir == '':
        now = datetime.now()
        save_dir = os.path.join("exp", now.strftime("%Y-%m-%d"))
        
    exp_dir,data_dir = create_exp_file(save_dir)
    save_script(exp_dir)
    
    num_chunks_exp = cfg.num_chunks + chunk_id

    if resume_exp is not None and os.path.exists(resume_exp):
        print(f"Reprise depuis {resume_exp} au chunk {chunk_id}")
        state = load_checkpoint(resume_exp, chunk_id)
        # cfg et graines viennent de parse_cli, qui a deja lu ce dossier : les
        # recharger ici ecraserait les flags passes en ligne de commande.
        subkeys = list(subkeys_init) if subkeys_init is not None else []
        if len(subkeys) < num_chunks_exp:
            key, k_sup = random.split(key)
            subkeys.extend(random.split(k_sup, num_chunks_exp - len(subkeys)))

        start_chunk = chunk_id
        # La forme du reseau vient du cfg RECHARGE, pas de celui passe en argument :
        # les poids du checkpoint ne se remettent en forme que d'une seule facon.
        model = build_model(cfg)
        # Le seul controle qui attrape VRAIMENT une reprise mal formee. Les poids
        # d'un agent sont un vecteur plat : une forme differente ne leve pas
        # d'erreur, elle les reinterprete de travers et le run continue (963e465).
        # Necessaire aussi parce que les config.json anterieurs a 591269d n'ont
        # pas la cle memory_mode : ils retombent sur le defaut "separee" alors que
        # leur checkpoint a ete produit en "jointe". load_config ne peut pas le
        # deviner ; la comparaison des formes reelles, si.
        n_ckpt = state.agents.params.shape[1]
        if model.num_params != n_ckpt:
            raise ValueError(
                f"Forme du reseau incompatible avec le checkpoint : "
                f"le modele construit a {model.num_params} parametres, "
                f"le checkpoint en porte {n_ckpt}.\n"
                f"  model_version={cfg.model_version} memory_mode={cfg.memory_mode} "
                f"hidden_dim={cfg.hidden_dim} hidden_layers={list(cfg.hidden_layers)}\n"
                f"Config rechargee depuis {resume_exp}/config.json ; si elle est "
                f"anterieure a 591269d, y ajouter la cle memory_mode a la main.")
    else:
        if subkeys_init is not None:
            subkeys = list(subkeys_init)
            if len(subkeys) < num_chunks_exp:
                key, k_sup = random.split(key)
                subkeys.extend(random.split(k_sup, num_chunks_exp - len(subkeys)))
        else:
            key, *subkeys = random.split(key, num_chunks_exp + 1)
        model = build_model(cfg)
        key, subkey_state = jax.random.split(key)
        state = init_state(subkey_state, cfg, model)
        start_chunk = 1
        

    print(f"[reseau] {cfg.model_version} memory_mode={cfg.memory_mode} "
          f"hidden_dim={cfg.hidden_dim} hidden_layers={list(cfg.hidden_layers)} "
          f"output_dim={cfg.output_dim} -> {model.num_params} parametres")
    if cfg.inner_policy:
        print(f"[loss politique] portee={cfg.inner_policy_portee} "
              f"-somme_a pi(a) v_hat(a), lr={cfg.inner_policy_lr:g}, "
              f"a chaque pas, seulement si l'erreur de prediction < "
              f"{cfg.inner_policy_seuil:g}")
    if cfg.value_map:
        print("[value map] canal supplementaire : valeur de chaque case, "
              "peinte avant le conv")
    if cfg.vpred_gain != 1.0:
        print(f"[vpred gain] v_pred multiplie par {cfg.vpred_gain:g} avant la tete")
    if cfg.vpred_oracle:
        print("[vpred oracle] v_pred impose aux vraies valeurs : "
              + "  ".join(f"c{i}={LABELS[r.id]} {r.delta_energy:+g}"
                          for i, r in enumerate(cfg.resources))
              + "\n               la tete de valeur est court-circuitee "
                "(plafond : l'evolution part d'une information parfaite)")
    if cfg.inner_loop and cfg.inner_target == "delta":
        print("[cible] la loss de valeur vise le delta d'energie : le cout "
              "metabolique et la satiete y sont, une sortie 'rien' est ajoutee")
    if cfg.inner_loop:
        n_inner = int(masque_valeur(model).sum())
        _every, _anc = periode_inner(cfg)
        if cfg.inner_optim == "adam":
            _mo = 2 * cfg.n_agents_max * model.num_params * 4 / 1e6
            print(f"[optimiseur] adam, {_mo:.0f} Mo de moments pour "
                  f"{cfg.n_agents_max} agents")
        print(f"[boucle interne] lr={cfg.inner_lr} fenetre={cfg.inner_window} "
              f"maj tous les {_every} pas ({_anc} ancres) ecretage={cfg.inner_clip} "
              f"-> {n_inner} parametres appris pendant la vie, "
              f"{model.num_params - n_inner} laisses a l'evolution")
    save_config(cfg,subkeys, exp_dir)
    start_step = start_chunk * cfg.chunk_size

    sim_data = simulation_data(cfg=cfg,start_step=start_step, start_chunk =start_chunk)
    sim_data.register_founders(state,model)
    
    
    n_types = len(cfg.resources)
    initial_grid_res = state.grid[:n_types, :, :]       
    chunks_survived = start_chunk
    
    key,subkey_lab = random.split(key) #same lab env for every test
    subkey_lab,subkey_env_lab = random.split(subkey_lab) #same lab env for every test
    
    plot_current_config(initial_grid_res,state.grid[-1, :, :] ,state.agents.position, state.agents.alive, exp_dir , cfg.resources,name_fig=f'init')

    pending_futures = {}  # future -> chunk_idx
    ctx = mp.get_context('spawn')
    



    with ProcessPoolExecutor(max_workers=n_video_workers,mp_context=ctx,initializer=_hide_gpu) as executor:
        for chunk_idx in range(start_chunk, num_chunks_exp):
            
            # --- Collecte non-bloquante des vidéos déjà terminées ---
            done = [f for f in pending_futures if f.done()]
            for f in done:
                cidx = pending_futures.pop(f)
                try:
                    path = f.result()
                    print(f"  [video] chunk {cidx} sauvegardé : {path}")
                except Exception as e:
                    print(f"  [video] ERREUR chunk {cidx} : {e}")

            # --- Simulation GPU ---
            subkey = subkeys[chunk_idx-1]

            keys_chunk = jax.random.split(subkey, cfg.chunk_size)
            
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx} START  @ {time.strftime('%H:%M:%S')}")
            # avec vpred_oracle, le modele porte les valeurs de la permutation
            # en cours ; sans, model_pour renvoie `model` inchange
            state, outputs = run_simulation_chunk(state, model_pour(cfg, model),
                                                  keys_chunk, cfg)
            print(f"[sim   | PID {os.getpid()}] chunk {chunk_idx} DONE   @ {time.strftime('%H:%M:%S')}")
            
  

            sim_data.update_data_with_chunk(outputs,data_dir,chunk_idx)

            if (chunk_idx) % 100  == 0 or  (chunk_idx) == 10:
                sim_data.plot(state,exp_dir)
                sim_data.compute_R0_and_plot(state,state.step,exp_dir)
            
            current_sim_state = sim_data.check_end_condition()
            
            if current_sim_state != 'interesting':
                sim_data.plot(state,exp_dir)
                print(f"Stopping criterion : {current_sim_state}")
                
                if state.step > 100000 :
                    
                    vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{chunk_idx}.mp4")
                    outputs_np = video_payload(outputs, stride=cfg.video_stride)
                    future = executor.submit(_video_worker, outputs_np, vid_path, 20, 5, cfg.resources)
                    pending_futures[future] = chunk_idx
                    
                return state, outputs, exp_dir, current_sim_state,chunks_survived
            
            ## Genealogy and MCRA
            sim_data.update_genealogy(outputs,state,exp_dir)
            sim_data.update_mrca_and_plot(outputs,exp_dir)
            

            ## TEST AGENTS IN LAB ENV ##
            phase = (chunk_idx) % cfg.cycle_period
            if ((chunk_idx) % cfg.pca_save_freq == 0) or (phase in cfg.lab_after_shuffle):
                subkey_lab,subkey_sim = random.split(subkey_lab)
                def submit_video(outputs_np, vid_path, *args, label=None):
                    os.makedirs(os.path.dirname(vid_path), exist_ok=True)  # avant submit
                    fut = executor.submit(_video_worker, outputs_np, vid_path, *args)
                    pending_futures[fut] = label if label is not None else vid_path
                    return fut
                sim_data.launch_env(state = state,key_env = subkey_env_lab,subkey_sim = subkey_sim,model = model_pour(cfg, model),exp_dir = exp_dir,n=50,submit_video=submit_video)

            ## EVALUABILITE ##
            if cfg.evolvability_freq > 0 and int(state.step) % cfg.evolvability_freq == 0:
                subkey_lab, subkey_evo = random.split(subkey_lab)
                sim_data.launch_evolvability(state=state, key_env=subkey_env_lab,
                                             subkey_sim=subkey_evo,
                                             model=model_pour(cfg, model),
                                             exp_dir=exp_dir)

            chunks_survived+=1
            # --- Checkpoint (synchrone) ---
            if (chunk_idx) % cfg.checkpoint_freq == 0:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"state_chunk_{chunk_idx}.pkl")
                save_checkpoint(state, ckpt_path)

            # --- Vidéo (asynchrone) ---
            if (chunk_idx) % cfg.video_freq == 0 or chunk_idx==start_chunk or (phase in [1,2,3]) :
                vid_path = os.path.join(exp_dir, "videos", f"video_chunk_{chunk_idx}.mp4")
                outputs_np = video_payload(outputs, stride=cfg.video_stride)
                future = executor.submit(_video_worker, outputs_np, vid_path, 20, 5, cfg.resources)
                pending_futures[future] = chunk_idx


            ## SHUFFLE RESOURCES ##
            if (chunk_idx) % cfg.cycle_period == 0 and chunk_idx >10 :
                old_resources = cfg.resources
                if cfg.shuffle_version == "v1":
                    # permute BASE_RESOURCES avec la cle du CHUNK, identite exclue.
                    # BASE_RESOURCES et non cfg.resources : c'est ce que faisait le
                    # code d'origine, et v1 n'existe que pour rejouer ces runs-la.
                    # Avec un autre nombre de ressources il rendrait une liste de
                    # taille differente, qui desaccorderait le reseau en silence.
                    if len(cfg.resources) != len(BASE_RESOURCES):
                        raise ValueError(
                            f"--shuffle v1 suppose les {len(BASE_RESOURCES)} "
                            f"ressources de BASE_RESOURCES, la config en a "
                            f"{len(cfg.resources)}. Utiliser --shuffle v2.")
                    new_resources = shuffle_resources_v1(BASE_RESOURCES, subkey)
                else:
                    key, subkey_shuffle = random.split(key)
                    new_resources = shuffle_resources(old_resources, subkey_shuffle)

                print("----------------- Change config at step : ---------------", state.step)
                for k, (old, new) in enumerate(zip(old_resources, new_resources)):
                    flag = "" if old.id == new.id else "  <-- changé"
                    print(f"  canal {k} : {LABELS[old.id]:>7} -> {LABELS[new.id]:<7}{flag}")

                log_resource_shuffle(exp_dir, chunk_idx, int(state.step),
                                     old_resources, new_resources)

                cfg = cfg._replace(resources=new_resources)
                sim_data.cfg = cfg
                
        print("Simulation terminée. Attente des vidéos en cours...")
        for f in as_completed(pending_futures):
            cidx = pending_futures[f]
            try:
                print(f"  [video] chunk {cidx} finalisé : {f.result()}")
            except Exception as e:
                print(f"  [video] ERREUR chunk {cidx} : {e}")


    sim_data.save_mrca_sim(data_dir)
    delta_sim = time.time()-start_time_sim
    delta_min,delta_sec = sec_to_minutes(delta_sim)
    print(f"Time to compute the simulation : {delta_min} min and {delta_sec:.1f} s")
    return state, outputs, exp_dir,current_sim_state,chunks_survived



CLI_PARAMS = [
    (("-t", "--temp"),    "temperature",                    float),
    (("-d", "--decay"),   "energy_decay",                   float),
    (("-f", "--factor"),  "factor_energy_decay_not_moving", float),
    (("-p", "--pmut"),    "param_mutate",                   float),
    (("-c", "--cycle"),   "cycle_period",                   int),
    (("-n", "--chunks"),  "num_chunks",                     int),
    (("-a", "--agents"),  "n_agents_max",                   int),
    (("-e", "--emax"),    "energy_max",                     float),
    (("--mvar",),         "mutation_var",                   float),
    (("--pregrow",),      "pre_growth_step",                int),
    (("--lab",),          "lab_time_steps",                 int),
    (("--video-freq",),   "video_freq",                     int),
    (("--crowd-start",),  "crowd_start",                    int),
    (("--crowd-limit",),  "crowd_limit",                    int),
    (("--view",),         "agent_view",                     int),
    (("--invasion",),     "invasion_start",                 int),
    (("--invasion-frac",),"invasion_frac",                  float),
    (("--forget-bias",),  "lstm_forget_bias",               float),
    (("--evo",),          "evolvability_freq",              int),
    (("--evo-agents",),   "evolvability_agents",            int),
    (("--evo-children",), "evolvability_children",          int),
    (("--replay-n",),     "replay_top_n",                   int),
    (("--replay-keys",),  "replay_keys",                    int),
    (("--replay-video-frac",), "replay_video_min_frac",     float),
    (("--inner-lr",),     "inner_lr",                       float),
    (("--inner-window",), "inner_window",                   int),
    (("--inner-every",),  "inner_every",                    int),
    (("--inner-clip",),   "inner_clip",                     float),
    (("--vpred-gain",),   "vpred_gain",                     float),
    (("--inner-policy-lr",), "inner_policy_lr",             float),
    (("--inner-policy-seuil",), "inner_policy_seuil",       float),
    (("--inner-policy-cout",),  "inner_policy_cout",        float),
]

# booleens : --dumb / --no-dumb, defaut pris sur le Config
CLI_FLAGS = [
    (("--dumb",),    "dumb_agent"),
    (("--wall",),    "letal_wall"),
    (("--logobs",),  "log_obs"),
    (("--randpos",), "random_pos_offspring"),
    (("--mem-ablation",), "lab_memory_ablation"),
    (("--weights",), "track_weights"),
    (("--inner",),   "inner_loop"),
    (("--vpred-oracle",), "vpred_oracle"),
    (("--value-map",),   "value_map"),
    (("--inner-policy",), "inner_policy"),
]


ABLATIONS = {
    "m": "ablate_memory",
    "r": "ablate_recurrence",
    "i": "ablate_interoception",
    "f": "ablate_feedback",
}


def parse_cli(cfg):
    # --from est lu en PREMIER : la config chargee devient la base, et les
    # autres flags s'appliquent par dessus. Sans ca ils seraient ecrases.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--from", dest="config_exp", default=None, metavar="DIR",
                     help="reprendre la config ET les graines d'un run, mais repartir de zero")
    pre.add_argument("-r", "--resume", default=None, metavar="DIR",
                     help="dossier d'experience a reprendre (avec --chunk-id)")
    connus, _ = pre.parse_known_args()
    subkeys = None
    source = connus.config_exp or connus.resume
    if source:
        cfg, subkeys = load_config(source)
        cfg = resolve_model(cfg)
        print(f"[cli] config et graines reprises de {source}")

    p = argparse.ArgumentParser(
        parents=[pre],
        description="Simulation eco-evo. Sans argument, les valeurs du Config de run.py.")
    for flags, champ, typ in CLI_PARAMS:
        p.add_argument(*flags, dest=champ, type=typ, default=getattr(cfg, champ),
                       metavar=typ.__name__.upper()[0], help=f"{champ} (defaut %(default)s)")
    for flags, champ in CLI_FLAGS:
        p.add_argument(*flags, dest=champ, action=argparse.BooleanOptionalAction,
                       default=getattr(cfg, champ), help=f"{champ} (defaut %(default)s)")
    p.add_argument("--inner-policy-portee", dest="inner_policy_portee",
                   choices=["case", "vue"], default=cfg.inner_policy_portee,
                   help="portee de la loss de politique : 'case' = la seule case "
                        "devant, 'vue' = la meilleure case visible escomptee par "
                        "la distance (defaut %(default)s)")
    p.add_argument("--inner-optim", dest="inner_optim",
                   choices=["sgd", "adam"], default=cfg.inner_optim,
                   help="optimiseur de la boucle interne (defaut %(default)s). "
                        "adam coute 2 moments par agent")
    p.add_argument("--inner-target", dest="inner_target",
                   choices=["reward", "delta"], default=cfg.inner_target,
                   help="cible de la loss de valeur : 'reward' = ce qui est mange, "
                        "'delta' = la variation d'energie, cout metabolique et "
                        "satiete compris (defaut %(default)s)")
    p.add_argument("--init", dest="init_scale", choices=["constant", "lecun"],
                   default=cfg.init_scale,
                   help="echelle des poids initiaux (defaut %(default)s)")
    p.add_argument("--shuffle", dest="shuffle_version", choices=["v1", "v2"],
                   default=cfg.shuffle_version,
                   help="version du shuffle (defaut %(default)s) ; v1 = avant le 18 aout")
    p.add_argument("-m", "--model", dest="model_version",
                   choices=sorted(MODEL_VERSIONS) + ["custom"],
                   default=cfg.model_version, help="version du reseau (defaut %(default)s)")
    p.add_argument("-s", "--seed",    type=int, default=None,
                   help="graine (defaut 105). Avec --from, la passer ignore les "
                        "graines chargees et en regenere de neuves")
    p.add_argument("-w", "--workers", type=int, default=4,    help="process video (defaut %(default)s)")
    p.add_argument("--chunk-id",      type=int, default=1,    help="chunk de reprise (defaut %(default)s)")
    p.add_argument("-x", "--ablate", default="", metavar="LETTRES",
                   help="ablations, lettres cumulables : "
                        + " ".join(f"{k}={v[7:]}" for k, v in ABLATIONS.items())
                        + " (ex: -x ri)")
    args = p.parse_args()

    maj = {c: getattr(args, c) for _, c, _ in CLI_PARAMS}
    maj.update({c: getattr(args, c) for _, c in CLI_FLAGS})
    maj["model_version"] = args.model_version
    maj["shuffle_version"] = args.shuffle_version
    maj["init_scale"] = args.init_scale
    maj["inner_target"] = args.inner_target
    maj["inner_optim"] = args.inner_optim
    maj["inner_policy_portee"] = args.inner_policy_portee

    lettres = args.ablate.lower()
    inconnues = sorted(set(lettres) - set(ABLATIONS))
    if inconnues:
        p.error(f"lettre(s) d'ablation inconnue(s) : {''.join(inconnues)} — attendu "
                + " ".join(f"{k}={v[7:]}" for k, v in ABLATIONS.items()))
    for lettre in lettres:
        maj[ABLATIONS[lettre]] = True

    # apres les ablations : -x m les pose ici, les tester plus haut les raterait
    if maj.get("inner_policy"):
        if not maj.get("inner_loop"):
            p.error("--inner-policy demande --inner : la loss de politique vise "
                    "la croyance produite par la boucle interne.")
        if maj.get("vpred_oracle"):
            print("[attention] --inner-policy avec --vpred-oracle : la valeur est "
                  "DONNEE et la politique\n            entrainee a s'en servir. "
                  "Rien n'emerge -- c'est le PLAFOND de la loss de\n"
                  "            politique : si elle n'apprend pas l'evitement avec "
                  "une croyance parfaite,\n            elle ne l'apprendra pas "
                  "avec une croyance apprise.")
    if maj.get("value_map") and not (maj.get("inner_loop") or maj.get("vpred_oracle")):
        p.error("--value-map demande une tete de valeur : ajouter --inner ou "
                "--vpred-oracle.")
    if maj.get("vpred_oracle") and maj["model_version"] != "v2":
        p.error("--vpred-oracle demande -m v2 : la tete de valeur n'existe "
                "qu'en memoire separee.")
    if maj.get("inner_loop"):
        if maj["model_version"] != "v2":
            p.error("--inner demande -m v2 : en v1 le LSTM ne recoit pas "
                    "last_eaten, il n'y a rien a predire.")
        if maj.get("ablate_memory") or maj.get("ablate_recurrence"):
            p.error("--inner et l'ablation de la recurrence s'annulent : le "
                    "carry est remis a zero a chaque pas, rien ne s'accumule.")

    # Une graine posee a la main l'emporte sur les graines chargees par --from,
    # sinon elle ne changerait que l'etat initial et pas les cles de chunk.
    if args.seed is None:
        args.seed = 105
    elif subkeys is not None:
        subkeys = None
        print(f"[cli] --seed {args.seed} : graines de --from ignorees, "
              f"nouveau tirage")

    neuf = cfg._replace(**maj)
    diff = {c: getattr(neuf, c) for c in Config._fields
            if getattr(neuf, c) != getattr(cfg, c)}
    if diff:
        print("[cli] surcharges :", ", ".join(f"{k}={v}" for k, v in diff.items()))
    return neuf, args, subkeys


if __name__ == '__main__':
    cfg = Config(
            grid_length=200,

            ### Simulation computation :
            chunk_size = 1000,
            num_chunks = 1000,
            checkpoint_freq = 50,
            video_freq = 50,
            pca_save_freq=50,

            ### AGENTS :
            n_agents_max=2000,
            n_agents_init=20,
            agent_view = 5,
            temperature=1/40,

            #Physiologie
            energy_decay=0.07/8,
            factor_energy_decay_not_moving = 0.3,
            energy_max = 8.0,

            time_to_die=100*4,
            time_above_repr = 80*4,
            min_energy_repr = 6.,
            starting_energy= 1.5,

            # Mutation parameters
            mutation_var = 0.02,
            param_mutate = 0.99,

            # INIT RESOURCES MAP
            pre_growth_step = 500,
            random_pos_offspring = False,
            dumb_agent = False,
            letal_wall=False,
            cycle_period = 200,
            log_obs = False,

            ablate_memory = False ,        # coupe les trois canaux ci-dessous
            ablate_recurrence = False ,     # lstm_h / lstm_c
            ablate_interoception = False ,  # energie
            ablate_feedback = False ,

            lab_time_steps = 2000,
            
            model_version = "v1", 
              
        )
    
    
    
    cfg, args, subkeys_init = parse_cli(cfg)

    # Sanity check :
    a = cfg.starting_energy - cfg.energy_decay * cfg.time_above_repr
    b = cfg.min_energy_repr
    assert(  b > a)

    key = random.PRNGKey(args.seed)
    print(jax.devices())

    state_final, output, exp_dir,_,_ = launch_simulation_chunked(
        key, cfg, resume_exp=args.resume, n_video_workers=args.workers,
        chunk_id=args.chunk_id, subkeys_init=subkeys_init)
