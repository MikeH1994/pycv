import matplotlib.pyplot as plt
import numpy as np
import os

def save_current_fig(fpath, verbose=False):
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    parent_dir = os.path.dirname(fpath)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
        if verbose:
            print(f"Created {parent_dir}")
    plt.savefig(fpath, dpi=120)
    if verbose:
        print(f"saved {fpath}")


def set_labels_and_legend(title, xlabel, ylabel, title_fontsize=20, label_fontsize=18, legend_fontsize=16):
    if title is not None:
        plt.title(title, fontsize=title_fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(fontsize=legend_fontsize)
    plt.yticks(fontsize=legend_fontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > 0:
        plt.legend(loc=0, prop={'size': legend_fontsize})

def set_colorbar(tick_position = None, label=None, label_fontsize=18, tick_fontsize=16):
    cbar = plt.colorbar()
    if tick_position is not None:
        cbar.set_ticks(tick_position)
    if label is not None:
        cbar.set_label(label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

def invert_x_axis():
    pass

def invert_y_axis():
    pass