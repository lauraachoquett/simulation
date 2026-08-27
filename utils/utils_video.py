""" This script contains general utilities.
"""


from moviepy import VideoFileClip

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np
import matplotlib.colors as mcolors
from simulation.data_class import COLOR_BY_ID

def save_chunk_video(outputs, filename, fps=20.0, scale=16, max_age=200, resources=None):
    grids = outputs.grid
    # Alpha IGNORE. COLOR_BY_ID[1] vaut "#3933F35E", soit 37% d'opacite : melange
    # au fond blanc il donne #b6b4fb, invisible a l'ecran. Le medium etant la
    # ressource qui pousse le plus vite, c'etait la plus abondante qui ne se
    # voyait pas. Une video de diagnostic doit montrer ce qui est la.
    colors = np.array([mcolors.to_rgb(COLOR_BY_ID[r.id][:7]) for r in resources],
                      dtype=np.float32)               # (n_types, 3)
    n_types = colors.shape[0]                    # déduit du tableau de couleurs

    
    with VideoWriter(filename, fps=fps) as vid:
        for i in range(grids.shape[0]):
            res = np.asarray(grids[i, :n_types])          # (n_types, L, L)
            H, W = res.shape[1], res.shape[2]

            present  = res.sum(axis=0) > 0
            argmax_k = res.argmax(axis=0)

            img = np.ones((H, W, 3), dtype=np.float32)
            for k in range(n_types):
                mask_k = present & (argmax_k == k)
                img[mask_k] = colors[k]                    # <-- couleur par identité

            # --- agents overlay (inchangé) ---
            pos   = np.array(outputs.position[i])
            born  = np.array(outputs.born_step[i])
            alive = np.array(outputs.alive[i])
            step  = int(outputs.step[i])

            age = np.clip(step - born, 0, max_age)
            intensity = np.clip(1.0 - 0.8 * (age / max_age), 0.2, 1.0)

            pos, intensity = pos[alive > 0], intensity[alive > 0]
            xs = pos[:, 0].astype(np.int32)
            ys = pos[:, 1].astype(np.int32)
            valid = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)
            xs, ys, intensity = xs[valid], ys[valid], intensity[valid]

            img[xs, ys] = 0.0
            img[xs, ys, 0] = intensity

            img_upscaled = np.kron(img, np.ones((scale, scale, 1), dtype=np.float32))
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

