import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def add_bubble_highlight(xy, color, text, ax, bar):
    (x, y) = xy
    kwargs = {'fontname' : 'Gill Sans MT', 'color' : 'white'}#, 'fontweight' : 'heavy'}
    ax.annotate(text, xy=xy, xytext=(x, y + 0.25),
                size=12, va="center",
                bbox=dict(boxstyle="round", fc=color, ec="none"),
                arrowprops=dict(arrowstyle="wedge, tail_width=1.",
                                fc=color, ec="none", relpos=(0.5, 0.2),
                                  patchA=None), **kwargs)

def highlight_bar(robustness, du_wcu_ix, rob_col, s, c, l, bars,
                  comp_bars, axis):
    ix = np.where(robustness[du_wcu_ix, rob_col - 1] == s)[0][0]
    bar = bars[ix]
    bar.set_color(c)
    comp_bars.append(bar)
    (x, y) = bar.xy
    add_bubble_highlight((x, y + bar._height), c, l, axis, bar)

def pseudo_robustness_plot(utilities, robustnesses, colors,
                           files_root_directory, nwcu=259, beta=False,
                           plot_du=True, highlight_sols=None):
    if highlight_sols is None:
        highlight_sols = {'ids': [], 'labels': []}

    nutils = len(utilities)
    nsols = len(robustnesses[0])
    ndu = nsols - nwcu
    bar_width = 0.5
    rob_col = 3 if beta else 1

    fig, axes = plt.subplots(nutils, 1, figsize=(10, 6))
    plt.subplots_adjust(left=0.2)

    bars_wcu, bars_du = 0, 0
    comp_bars = []
    for utility, robustness, axis in zip(utilities, robustnesses, axes):
        axis.set_ylabel(utility + '\n\nApproximate\nRobustness [-]',
                        **{'fontname':'CMU Bright', 'size' : 13})
        du_ix = robustness[:, rob_col - 1].astype(int) >= nwcu
        wcu_ix = robustness[:, rob_col - 1].astype(int) < nwcu
        bars_wcu = axis.bar(np.arange(nwcu) + bar_width + 1,
                            robustness[wcu_ix, rob_col], bar_width,
                            color=colors[0], label='WCU Optimization')
        if plot_du:
            bars_du = axis.bar(np.arange(ndu) + 1,
                               robustness[du_ix, rob_col], bar_width,
                               color=colors[1], label='DU Optimization')
        axis.set_xticks([], [])
        axis.set_xticklabels(axis.get_xticks(),
                             {'fontname':'CMU Bright', 'size': 13})
        axis.set_yticks([0, 0.5, 1.0])
        axis.set_yticklabels(['0%', '50%', '100%'],
                             {'fontname':'CMU Bright', 'size': 13})

        if len(highlight_sols['ids']) > 0:
            for s, c, l in zip(highlight_sols['ids'],
                               highlight_sols['colors'],
                               highlight_sols['labels']):
                if s < nwcu:
                    highlight_bar(robustness, wcu_ix, rob_col, s, c, l,
                                  bars_wcu, comp_bars, axis)
                else:
                    if plot_du:
                        highlight_bar(robustness, du_ix, rob_col, s, c, l,
                                      bars_du, comp_bars, axis)

        # axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        # axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)

    x_upper_lim = 0
    for bar in comp_bars:
        x_upper_lim = max(x_upper_lim, bar.xy[0])

    for axis in axes:
        axis.set_ylim(0, 1)
        axis.set_xlim(-1., x_upper_lim + 0.5)

    axes[-1].set_xlabel('Approximate Robustness-ranked Solution',
                        **{'fontname':'CMU Bright', 'size' : 13})

    lines = ((bars_wcu[-1], bars_du[-1])
             if type(bars_du) is not int else (bars_wcu))
    lines_labels = (('WCU\nOptimization', 'DU\nOptimization')
                    if type(bars_du) is not int else ('WCU'))

    lines += tuple([comp_bars[i] for i in range(len(highlight_sols['labels']))])
    if plot_du:
        comp_labels = ['High Robustness\n{}'.format(l) for l in highlight_sols['labels']]
        lines_labels += tuple(comp_labels)

    legend = plt.figlegend(lines, lines_labels, 'lower center',
                           bbox_to_anchor=(0.5, -0.01), ncol=len(lines))
    plt.setp(legend.texts, family='CMU Bright')

    # axes[legend_in_axis].legend()
    plt.savefig(files_root_directory + 'robustness_rank_bars{}{}.svg'.format(
        '' if plot_du else '_no_du', ''
        if len(['ids']) > 0 else '_compromise')
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
