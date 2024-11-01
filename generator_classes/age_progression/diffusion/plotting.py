import os

import matplotlib.pyplot as plt
import numpy as np
from age_progression.diffusion.diffusion_utils import norm_all
from matplotlib.animation import FuncAnimation, ImageMagickWriter, PillowWriter


def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn, w, save=False):
    ncols = n_sample // nrows
    sx_gen_store = np.moveaxis(x_gen_store, 2, 4)  # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0],
                             n_sample)  # unity norm to put in range [0,1] for np.imshow

    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols, nrows))

    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i, (row * ncols) + col]))
        return plots

    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store], interval=200, blit=False, repeat=True,
                        frames=nsx_gen_store.shape[0])

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path_static = save_dir + f"{fn}_w{w}.gif"
        save_path_anim = save_dir + 'animation_ddpm.mp4'
        ani.save(save_path_static, dpi=100, writer=PillowWriter(fps=5))
        print(f'saved gif at {save_dir}')
    plt.close()

    return ani

def show_images(imgs, nrow=2):
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4,2 ))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.show()
