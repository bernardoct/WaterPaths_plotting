import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def pseudo_robustness_plot(utilities, robustnesses, colors,
                           files_root_directory, nwcu=259, beta=False,
                           plot_du=True, highlight_sols=()):
    nutils = len(utilities)
    nsols = len(robustnesses[0])
    ndu = nsols - nwcu
    bar_width = 0.5
    rob_col = 3 if beta else 1
    highlight_sols = (highlight_sols) if isinstance(highlight_sols, int) \
        else highlight_sols

    fig, axes = plt.subplots(nutils, 1, figsize=(8, 6))
    plt.subplots_adjust(left=0.2)

    bars_wcu, bars_du = 0, 0
    comp_bars = []
    for utility, robustness, axis in zip(utilities, robustnesses, axes):
        axis.set_ylabel(utility + '\n\nApproximate\nRobustness [-]',
                        **{'fontname':'CMU Bright', 'size' : 13})
        axis.set_ylim(0, 1)
        axis.set_xlim(-1., ndu + 0.25)
        du_ix = robustness[:, rob_col - 1].astype(int) >= nwcu
        wcu_ix = robustness[:, rob_col - 1].astype(int) < nwcu
        bars_wcu = axis.bar(np.arange(ndu) + bar_width + 1,
                            robustness[wcu_ix, rob_col][:ndu], bar_width,
                            color=colors[0], label='WCU Optimization')
        if plot_du:
            bars_du = axis.bar(np.arange(ndu) + 1,
                               robustness[du_ix, rob_col], bar_width,
                               color=colors[1], label='DU Optimization')
        axis.set_xticks([], [])
        axis.set_xticklabels(axis.get_xticks(),
                             {'fontname':'CMU Bright', 'size' : 13})
        axis.set_yticks([0, 0.5, 1.0])
        axis.set_yticklabels([0, 0.5, 1.0],
                             {'fontname':'CMU Bright', 'size' : 13})

        if len(highlight_sols) > 0:
            comp_bars = []
            for s, c in zip(highlight_sols, cm.get_cmap('Accent').colors):
                if s < nwcu:
                    ix = np.where(robustness[wcu_ix, rob_col - 1] == s)[0][0]
                    bars_wcu[ix].set_color(c)
                    comp_bars.append(bars_wcu[ix])
                else:
                    if plot_du:
                        ix = np.where(robustness[du_ix, rob_col - 1] == s)[0][0]
                        bars_du[ix].set_color(c)
                        comp_bars.append(bars_wcu[ix])

        # axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        # axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Approximate Robustness-ranked Solution',
                        **{'fontname':'CMU Bright', 'size' : 13})

    lines = ((bars_wcu, bars_du) if type(bars_du) is not int else (bars_wcu))
    lines_labels = (('WCU', 'DU') if type(bars_du) is not int else ('WCU'))

    if plot_du:
        lines += tuple(comp_bars)
        comp_labels = ['Compromise\nSolution {}'.format(i) for i in range(1, len(comp_bars) + 1)]
        lines_labels += tuple(comp_labels)

    legend = plt.figlegend(lines, lines_labels, 'lower center', bbox_to_anchor=(0.5, -0.01), ncol=len(lines))
    plt.setp(legend.texts, family='CMU Bright')

    # axes[legend_in_axis].legend()
    plt.savefig(files_root_directory + 'robustness_rank_bars{}{}.svg'.format(
        '' if plot_du else '_no_du', '' if len(highlight_sols) > 0 else '_compromise')
    )

    plt.show()


def important_factors_multiple_solutions_plot(most_influential_factors_all,
                                              lr_coef_all, nfactors, labels,
                                              files_root_directory):
    title_font = 'Gill Sans MT'
    everything_else_font = 'CMU Bright'
    # everything_else_font = 'Ubuntu'
    nsols = len(most_influential_factors_all)

    factors_to_plot = np.unique(
        np.array(most_influential_factors_all)[:, -nfactors:].ravel()
    )
    nfactors_to_plot = len(factors_to_plot)

    fig, axes = plt.subplots(1, nsols, sharey=True, sharex=True,
                             figsize=(4.6, 3.5))
    axes = axes if isinstance(axes, list) else [axes]

    plt.subplots_adjust(left=0.45, bottom=0.2, top=0.85, right=0.96)

    for coefs, axis, c, s in zip(lr_coef_all, axes,
                                 cm.get_cmap('Accent').colors,
                                 range(nsols)):
        axis.barh(range(nfactors_to_plot),
                  np.abs(coefs[factors_to_plot]) / np.sum(np.abs(coefs)),
                  color=c)
        axis.set_yticks(range(nfactors_to_plot))
        axis.set_yticklabels(np.array(labels)[factors_to_plot],
                             **{'fontname': everything_else_font, 'size': 13})
        axis.set_xlabel('Relative Importance\n(logistic regression coefficients)',
                        **{'fontname': everything_else_font, 'size': 13})
        axis.set_xticks(axis.get_xlim())
        axis.set_xticklabels(['Low', 'High'], **{'fontname': everything_else_font})
        axis.set_title('Compromise\nSolution {}'.format(s + 1),
                       **{'fontname': title_font, 'size': 15})

    plt.savefig(files_root_directory + 'important_factors.svg')
