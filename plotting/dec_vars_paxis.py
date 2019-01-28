import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_dec_vars_paxis(dec_vars, max_mins, axes, c, decvars_order, label):
    n_dvs = len(dec_vars)

    for dv_name, ax in zip(decvars_order, axes.ravel()):
        utils_names = dec_vars[dv_name].keys()
        dv_data = []
        for u_name in utils_names:
            dv_data.append(dec_vars[dv_name][u_name])

        dv_data = np.array(dv_data).T

        # If JLA, normalize data if needed
        if dv_name == 'Jordan Lake Allocation':
            total_alloc = np.sum(dv_data)
            if total_alloc > 1.:
                for i in range(len(dv_data)):
                    dv_data[i] /= total_alloc

        ax.set_yticks([])
        ax.set_ylim(max_mins[dv_name])
        ax.plot(dv_data, c=c, label=label)
        ax.set_xticks(range(len(utils_names)))
        ax.set_xticklabels(utils_names,
                           {'fontname': 'Open Sans Condensed', 'size': 11})
        ax.set_title(dv_name, {'fontname': 'Gill Sans MT', 'size': 12})

    axes[1, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -.4),
                      ncol=2,
                      prop={'family': 'Open Sans Condensed', 'size': 12})
