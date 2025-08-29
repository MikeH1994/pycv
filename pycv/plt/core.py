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
        print(f"Created {parent_dir}")
    plt.savefig(fpath, dpi=120)
    if verbose:
        print(f"saved {fpath}")


def set_labels_and_legend(title, xlabel, ylabel, title_fontsize=20, label_fontsize=18, legend_fontsize=16):
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(fontsize=legend_fontsize)
    plt.yticks(fontsize=legend_fontsize)
    plt.legend(loc=0, prop={'size': legend_fontsize})

