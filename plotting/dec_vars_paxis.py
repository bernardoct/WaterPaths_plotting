import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_dec_vars_paxis(dec_vars, grid_size, max_mins):
    n_dvs = len(dec_vars)

    fig, axes = plt.subplots(grid_size[0], grid_size[1])

    for dv_name, ax in zip(dec_vars.keys(), axes.ravel()):
        utils_names = dec_vars[dv_name].keys()
        dv_data = []
        for u_name in utils_names:
            dv_data.append(dec_vars[dv_name][u_name])

        dv_data = np.array(dv_data).T

        # If JLA, normalize data if needed
        if dv_name == 'Jordan Lake\nAllocation':
            for d in dv_data.T:
                total_alloc = np.sum(d)
                if total_alloc > 1.:
                    d /= total_alloc

        ax.set_ylim(max_mins[dv_name])
        for data, c in zip(dv_data, cm.get_cmap('Accent').colors):
            ax.plot(data, c=c)
        ax.set_xticks(range(len(utils_names)))
        ax.set_xticklabels(utils_names)
        ax.set_title(dv_name)

    plt.show()
