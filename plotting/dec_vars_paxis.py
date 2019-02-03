import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_dec_vars_paxis(dec_vars, max_mins, axes, c, decvars_order, label, utilities):
    for dv_name, ax in zip(decvars_order, axes.ravel()):
        utilities_to_plot = []
        dv_data = []
        for u_name in utilities:
            if u_name in dec_vars[dv_name]:
                utilities_to_plot.append(u_name)
                dv_data.append(dec_vars[dv_name][u_name])

        # ax.set_yticks([])
        ax.set_ylim(max_mins[dv_name])
        ax.plot(dv_data, c=c, label=label)
        ax.set_xticks(range(len(utilities_to_plot)))
        ax.set_xticklabels(utilities_to_plot,
                           {'fontname': 'Open Sans Condensed', 'size': 11})
        ax.set_title(dv_name, {'fontname': 'Gill Sans MT', 'size': 12})

    axes[-1, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -.4),
                      ncol=2,
                      prop={'family': 'Open Sans Condensed', 'size': 12})
