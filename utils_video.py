""" This script contains general utilities.
"""


from moviepy import VideoFileClip

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np

def save_chunk_video(outputs, filename, fps=20.0, scale=16, max_age=200):
    grids = outputs.grid
    agents = outputs.agents

    with VideoWriter(filename, fps=fps) as vid:
        for i in range(grids.shape[0]):

            grid_res = grids[i, 0]

            # --- base grid image ---
            img = np.ones((grid_res.shape[0], grid_res.shape[1], 3), dtype=np.float32)

            mask_res = grid_res > 0
            img[mask_res] = np.array([0.0, 1.0, 0.0])  # vert ressources

            # --- agents overlay ---
            pos = np.array(agents.position[i])      # (n_agents, 2)
            born = np.array(agents.born_step[i])    # (n_agents,)
            alive = np.array(agents.alive[i])       # (n_agents,)
            step = int(outputs.step[i])

            age = step - born
            age = np.clip(age, 0, max_age)

            # Échelle linéaire de l'intensité du rouge : 
            # Jeune (age=0) -> 1.0 (Rouge clair)
            # Vieux (age=max_age) -> 0.2 (Rouge foncé)
            intensity = 1.0 - 0.8 * (age / max_age)
            intensity = np.clip(intensity, 0.2, 1.0)

            # positions valides
            pos = pos[alive > 0]
            intensity = intensity[alive > 0]

            H, W = grid_res.shape

            # Masque booléen pour effacer le fond là où se trouvent les agents
            # Évite que le vert et le rouge se mélangent (jaune)
            agent_mask = np.zeros((H, W), dtype=bool)

            xs = pos[:, 0].astype(np.int32)
            ys = pos[:, 1].astype(np.int32)

            valid = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)
            xs, ys = xs[valid], ys[valid]
            intensity = intensity[valid]

            agent_mask[xs, ys] = True

            # On efface le fond (R, G, B) sous les agents pour éviter la superposition de couleurs
            img[agent_mask] = np.array([0.0, 0.0, 0.0])

            # Écriture directe des intensités de rouge
            img[xs, ys, 0] = intensity  # Rouge variable
            img[xs, ys, 1] = 0.0        # Vert éteint
            img[xs, ys, 2] = 0.0        # Bleu éteint

            # --- upscale ---
            img_upscaled = np.kron(
                img,
                np.ones((scale, scale, 1), dtype=np.float32)
            )

            vid.add(img_upscaled)
            

def merge_videos(directory, num_gens):
    """ Merge multiple videos into a single one.

    Attributes
    ----------
    directory: str
        name of directory where videos are saved

    num_gens: int
        last generation
    """
    gens = range(0, num_gens, 50) # loading every 50 generations
    L = []

    for gen in gens:
        file_path = "projects/" + directory + "/train/media/gen_" + str(gen) + ".mp4"
        video = VideoFileClip(file_path)
        L.append(video)

    final_clip = concatenate_videoclips(L)
    final_clip.to_videofile("projects/" + directory + "/total_training.mp4", fps=24, remove_temp=False)



class VideoWriter:
    """ Class for saving videos.
    """
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))

