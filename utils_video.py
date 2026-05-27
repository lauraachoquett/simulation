""" This script contains general utilities.
"""


from moviepy import VideoFileClip

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np

def save_chunk_video(outputs, filename, fps=20.0, scale=16):
    """
    scale : nombre de pixels par cellule de grille.
            Une grille 20x20 avec scale=16 → vidéo 320x320 px.
    """
    grids = outputs.grid  # Shape: (chunk_size, 2, n, n)

    with VideoWriter(filename, fps=fps) as vid:
        for i in range(grids.shape[0]):
            grid_res    = grids[i, 0]
            grid_agents = grids[i, 1]

            img = np.ones((grid_res.shape[0], grid_res.shape[1], 3), dtype=np.float32)

            mask_agents = grid_agents > 0
            mask_res    = grid_res > 0

            # Cellules avec agents : rouge pur → (1, 0, 0)
            img[mask_agents, 1] = 0.0
            img[mask_agents, 2] = 0.0

            # Cellules avec ressources : vert pur → (0, 1, 0)
            img[mask_res, 0] = 0.0
            img[mask_res, 2] = 0.0


            # Upscaling nearest-neighbor : répète chaque pixel `scale` fois
            # sur les deux axes spatiaux, sans toucher au canal couleur
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

