import numpy as np
import matplotlib.pyplot as plt

def plot_optimal_value_bounds(bounds, fig_name=None):
    bounds[1] = np.where(bounds[1] >= 0, bounds[1], np.nan)
    plt.figure(figsize=(8, 3))
    plt.plot(bounds[0], bounds[2], lw=2, ls='-', label="best upper bound")
    plt.plot(
        [bounds[0, 0], bounds[0, -1]],
        [bounds[2, -1], bounds[2, -1]],
        lw=2, ls='--', label="optimal value")
    plt.plot(bounds[0], bounds[1], lw=2, ls=':', label="best lower bound")
    plt.xlim([bounds[0, 0], bounds[0, -1]])
    plt.xlabel("solver time (s)")
    plt.ylabel("objective value")
    plt.grid()
    plt.legend()
    plt.xlim(xmin=0)
    if fig_name is not None:
        plt.savefig(fig_name + ".pdf", bbox_inches="tight")
    plt.show()