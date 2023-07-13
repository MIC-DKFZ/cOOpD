import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from numpy.lib.arraysetops import isin

v_data = (-1.5, 1.5)
v_seg = (0, 1)

def visualize_lung_data(x:dict ,visualize:str='data', cb=True, indices=None, title=None, title_dict=None, function=None, cmap=None):

    x_vis = np.copy(x[visualize])
    if function is not None:
        x_vis = function(x_vis)
    if indices is None:
        indices = np.arange(64)

    x_vis = x_vis[indices[:len(x_vis)]]
    if cb:
        if visualize == 'data':
            v_min, v_max = v_data
            # cmap="viridis"
            if cmap is None:
                cmap='gray'
        elif visualize == 'seg':
            v_min, v_max = v_seg
            if cmap is None:
                cmap="Reds"
        else:
            # v_min, v_max = None, None
            if cmap is None:
                cmap="viridis"
            if isinstance(x_vis, np.ndarray):
                v_min, v_max = np.min(x_vis), np.max(x_vis)
            elif torch.is_tensor(x_vis):
                v_min, v_max = torch.min(x_vis), torch.max(x_vis)
            else:
                v_min, v_max = None, None
    else:
        v_min, v_max = None, None
        if cmap is None:
            cmap="viridis"

    print(v_min, v_max)


    num_samples = len(x_vis)
    fig, axs = plt.subplots(8, 8)
    if num_samples > 0:
        for i in range(8):
            for j in range (8):
                b_idx = i*8+j
                # ind_idx = indices[b_idx]
                ind_idx = b_idx
                axs[i][j].set_axis_off()
                if b_idx >= num_samples:
                    continue
                if title is None:
                    axs[i][j].set_title(f"Slice: {x['slice_idxs'][ind_idx].item()}") #, b_idx{b_idx}")
                elif title_dict is not None:
                    axs[i][j].set_title("{}: {}".format(
                        title,
                        int(title_dict[title][indices[b_idx]].item())
                    )) #, b_idx{b_idx}")
                else:
                    pass
                im = axs[i][j].imshow(x_vis[ind_idx,0], cmap=cmap,vmin=v_min, vmax=v_max)

        
        fig.subplots_adjust(right=0.8)
        if cb:
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cbar_ax, )
    return fig, axs

def visualize_lung_seg_data(x:dict, indices=None):
    newcmp = get_seg_cm()

    if indices is None:
        indices = np.arange(64)

    num_samples = len(x['data'])
    fig, axs = plt.subplots(8, 8)
    cmap='gray'
    if num_samples > 0:
        for i in range(8):
            for j in range (8):
                b_idx = i*8+j
                ind_idx = indices[b_idx]
                axs[i][j].set_axis_off()
                if b_idx >= len(x['data']):
                    continue
                axs[i][j].set_title(f"Slice: {x['slice_idxs'][ind_idx].item()}") #, b_idx{b_idx}")
                im = axs[i][j].imshow(x['data'][ind_idx,0], vmin=v_data[0], vmax=v_data[1], cmap=cmap)
                axs[i][j].imshow(x['seg'][ind_idx,0], cmap=newcmp, vmin=v_seg[0], vmax=v_seg[1], interpolation='none')
                

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cbar_ax, )
    return fig, axs

def get_seg_cm():
    red = cm.get_cmap('Reds', 256)
    newcolors = red(np.linspace(0, 1, 256))
    newcolors[0:,-1]= 0
    newcolors[1:,-1]= 0.8
    newcmp = ListedColormap(newcolors)
    return newcmp

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in inches, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'LLNCS':
        width_in = 5.9
    # elif width == 'beamer':
    #     width_pt = 307.28987
    else:
        width_in = width

    # Width of figure (in pts)
    fig_width_in = width_in * fraction
    # Convert from pt to inches
    # inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    # fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def grouped_barplot(df, cat,subcat, val , err):
    ## This function is from Stackoverflow - Sadly I forgot the exact issue....
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width, 
                label="{} {}".format(subcat, gr), yerr=dfg[err].values)
#     plt.xlabel(cat)
#     plt.ylabel(val)
    plt.xticks(x, u)
#     plt.legend()
#     plt.show()
    fig, ax = plt.gcf(), plt.gca()
    return fig, ax