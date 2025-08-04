from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from dataclasses import dataclass




def updating_plot(data: Dict[str, List[np.ndarray | List]], delay_ms = 50):
    if len(data.values()) == 0:
        return

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y)

    # setting labelss
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Updating plot...")

    n_elems = len(list(data.values())[0])
    imshow = data["imshow"] if "imshow" in data else [None for _ in range(n_elems)]
    scatter = data["scatter"] if "scatter" in data else [None for _ in range(n_elems)]
    plot = data["plot"] if "plot" in data else [None for _ in range(n_elems)]


    # looping
    for _ in range(50):
        # updating the value of x and y
        line1.set_xdata(x * _)
        line1.set_ydata(y)

        # re-drawing the figure
        fig.canvas.draw()

        # to flush the GUI events
        fig.canvas.flush_events()
        plt.pause(0.05)
