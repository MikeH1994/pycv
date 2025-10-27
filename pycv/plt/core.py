import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.image import AxesImage


def save_current_fig(fpath, verbose=False, dpi=120, tight_layout=True, size_inches=(8,6)):
    if tight_layout:
        plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(*size_inches)
    parent_dir = os.path.dirname(fpath)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
        if verbose:
            print(f"Created {parent_dir}")
    plt.savefig(fpath, dpi=dpi)
    if verbose:
        print(f"saved {fpath}")


def set_labels_and_legend(title, xlabel, ylabel, title_fontsize=20, label_fontsize=18, legend_fontsize=16, tick_fontsize=16):
    if title is not None:
        plt.title(title, fontsize=title_fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > 0:
        plt.legend(loc=0, prop={'size': legend_fontsize})

def set_colorbar(tick_position = None, label=None, label_fontsize=18, tick_fontsize=16):
    ax = plt.gca()

    images = [child for child in ax.get_children() if isinstance(child, AxesImage)]
    cbar = plt.colorbar(images[-1], ax=ax)
    if tick_position is not None:
        cbar.set_ticks(tick_position)
    if label is not None:
        cbar.set_label(label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

def invert_x_axis():
    pass

def invert_y_axis():
    pass