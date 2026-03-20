import pycv.utils.windows as windows
import matplotlib.pyplot as plt
import numpy as np


def run():
    x1 = np.arange(50)
    win1 = windows.ahamming(50, 24.5)
    x2 = np.linspace(-60, 60, 1000)
    win2 = windows.ahamming_fn(50, 24.5)(x2)
    plt.scatter(x1, win1)
    plt.plot(x2, win2)
    plt.show()

    x2 = np.linspace(-50, 50, 100)
    plt.plot(x2, windows.ahamming_fn(50, 0.0)(x2))
    plt.show()

    x1 = np.arange(50)
    win1 = windows.tukey2(50, 1.0, 24.5)
    x2 = np.linspace(-60, 60, 1000)
    win2 = windows.tukey_fn(50, 1.0, 24.5)(x2)
    plt.scatter(x1, win1)
    plt.plot(x2, win2)
    plt.show()

    x2 = np.linspace(-50, 50, 100)
    plt.plot(x2, windows.tukey_fn(50, 1.0, 0.0)(x2))
    plt.show()

if __name__ == "__main__":
    run()
