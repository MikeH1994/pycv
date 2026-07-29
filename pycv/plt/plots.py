import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def intensity_scatterplot(x, y, vals, cmap='viridis', point_size=20, show=True):
    norm = Normalize(vmin=np.min(vals), vmax=np.max(vals))
    sc = plt.scatter(x, y, c=vals, s=point_size, cmap=cmap, norm=norm)
    plt.axis("equal")
    if show:
        plt.show()