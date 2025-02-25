from datetime import datetime

import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np

from utils.utils_path import RESULT_PATH

#  Result parameters
VERSION = str(datetime.now()).split('.')[0][5:]
for s in [' ', ':', '-']:
    VERSION = VERSION.replace(s, '_')
VERSION = ''.join(VERSION)


def compute_axis_lim(values: np.ndarray) -> tuple[float, float]:
    lower_lim, upper_lim = np.min(values), np.max(values)
    delta = 0.01 * (upper_lim - lower_lim)
    return lower_lim - delta, upper_lim + delta


def subplots_custom(nrows, ncols, sharex=False, sharey=False, hspace=None, wspace=None):
    default_figsize = (6.4, 4.8)
    adjusted_figsize = (default_figsize[0] * ncols, default_figsize[1] * nrows)
    fig = plt.figure(figsize=adjusted_figsize)
    gs = fig.add_gridspec(nrows, ncols, hspace=hspace, wspace=wspace)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    return fig, axs


def show_or_save_plot(plot_name: str, show: bool=False):
    plt.tight_layout()
    plt.show() if show else save_plot(plot_name)

def save_plot(plot_name: str):
    for character in ['\n', ' ']:
        plot_name = plot_name.replace(character, '_')
    filepath = op.join(RESULT_PATH, VERSION, plot_name)
    for i, f in enumerate(['png', 'pdf'][:]):
        filepath_with_format = filepath + '.' + f
        if i >= 1:
            filepath_with_format = op.join(op.dirname(filepath_with_format), f, op.basename(filepath_with_format))
        path = op.dirname(filepath_with_format)
        if not op.exists(path):
            os.makedirs(path)
        transparent = True if f == 'svg' else False
        plt.savefig(filepath_with_format, format=f, bbox_inches="tight",
                    transparent=transparent)
    plt.close()
