import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def add_bubble_highlight(xy, color, text, ax):
    (x, y) = xy
    kwargs = {'fontname' : 'Gill Sans MT', 'color' : 'white'}#, 'fontweight' : 'heavy'}

    # dy = np.ptp(ax.get_ylim()) / 15.
    dy = 0.25

    ax.annotate(text, xy=xy, xytext=(x, y + dy),
                size=10, va="center",
                bbox=dict(boxstyle="round", fc=color, ec="none"),
                arrowprops=dict(arrowstyle="wedge, tail_width=1.",
                                fc=color, ec="none", relpos=(0.2, 1.1),
                                  patchA=None), **kwargs)

def _highlight_bar(robustness, du_wcu_ix, rob_col, s, c, l, bars,
                   comp_bars, axis):
    ix = np.where(robustness[du_wcu_ix, rob_col - 1] == s)[0][0]
    bar = bars[ix]
    bar.set_color(c)
    comp_bars.append(bar)
    (x, y) = bar.xy
    add_bubble_highlight((x, y + bar._height), c, l, axis)

def pseudo_robustness_plot(utilities, robustnesses, colors,
                           files_root_directory, nwcu, beta=False,
                           plot_du=True, highlight_solutions=None, upper_xlim=-1):
    if highlight_solutions is None:
        highlight_sols = [{'ids': [], 'labels': [], 'colors': []}] * \
                         max(2, len(robustnesses))
    else:
        highlight_sols = highlight_solutions

    nutils = len(utilities)
    nsols = len(robustnesses[0])
    ndu = nsols - nwcu
    bar_width = 0.5
    rob_col = 3 if beta else 1

    fig, axes = plt.subplots(nutils, 1, figsize=(10, 6))
    plt.subplots_adjust(left=0.2)

    bars_wcu, bars_du = 0, 0
    comp_bars = []
    for utility, robustness, ax in \
            zip(utilities, robustnesses, axes):
        ax.set_ylabel(utility + '\n\nRobustness [-]',
                        **{'fontname':'Open Sans Condensed', 'size' : 12})
        du_ix = robustness[:, rob_col - 1].astype(int) >= nwcu
        wcu_ix = robustness[:, rob_col - 1].astype(int) < nwcu
        bars_wcu = ax.bar(np.arange(nwcu) + bar_width + 1,
                            robustness[wcu_ix, rob_col], bar_width,
                            color=colors[0], label='WCU Optimization')
        if plot_du:
            bars_du = ax.bar(np.arange(ndu) + 1,
                               robustness[du_ix, rob_col], bar_width,
                               color=colors[1], label='DU Optimization')
        ax.set_xticks([], [])
        ax.set_xticklabels(ax.get_xticks(),
                             {'fontname':'Open Sans Condensed', 'size': 12})
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0%', '50%', '100%'],
                             {'fontname':'Open Sans Condensed', 'size': 12})

        if len(highlight_sols[0]) > 0:
            for s, c, l in zip(highlight_sols[0]['ids'],
                               highlight_sols[0]['colors'],
                               highlight_sols[0]['labels']):
                _highlight_bar(robustness, wcu_ix, rob_col, s, c, l,
                                bars_wcu, comp_bars, ax)

        if plot_du and len(highlight_sols[1]) > 0:
            for s, c, l in zip(highlight_sols[1]['ids'],
                               highlight_sols[1]['colors'],
                               highlight_sols[1]['labels']):
                _highlight_bar(robustness, du_ix, rob_col, s + nwcu, c, l,
                                bars_du, comp_bars, ax)

        # axis.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # axis.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if upper_xlim < 0:
        if highlight_solutions == None:
            upper_xlim = max([ndu, nwcu])
        else:
            upper_xlim = 0
            for bar in comp_bars:
                upper_xlim = max(upper_xlim, bar.xy[0])

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlim(-1., upper_xlim + 0.5)

    axes[-1].set_xticks([1, int(upper_xlim) + 1])
    axes[-1].set_xticklabels(['1$^{st}$', str(int(upper_xlim)) + '$^{th}$'],
                       {'fontname': 'Open Sans Condensed', 'size': 12})

    axes[-1].set_xlabel('Robustness-ranked Solution', labelpad=-5,
                        **{'fontname': 'Open Sans Condensed', 'size': 12})

    lines = ((bars_wcu[-1], bars_du[-1])
             if type(bars_du) is not int else (bars_wcu))
    lines_labels = tuple(['WCU Optimization', 'DU Optimization']
                    if type(bars_du) is not int else ['WCU'])

    lines += tuple(comp_bars)
    if highlight_solutions != None:
        for h in highlight_sols:
            if len(h) > 0:
                lines_labels += tuple(h['labels'])

    legend = plt.figlegend(lines, lines_labels, 'lower center',
                           bbox_to_anchor=(0.55, -0.025), ncol=len(lines_labels))
    plt.setp(legend.texts, family='Open Sans Condensed')

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
    everything_else_font = 'Open Sans Condensed'
    # everything_else_font = 'Ubuntu'
    nsols = len(most_influential_factors_all)

    factors_to_plot = np.unique(
        np.array(most_influential_factors_all)[:, -nfactors:].ravel()
    )
    nfactors_to_plot = len(factors_to_plot)

    fig, axes = plt.subplots(1, nsols, sharey=True, sharex=True,
                             figsize=(4.6, 3.5))
    # axes = axes if isinstance(axes, list) else [axes]

    plt.subplots_adjust(left=0.45, bottom=0.2, top=0.85, right=0.96)

    for coefs, axis, c, s in zip(lr_coef_all, axes,
                                 [cm.get_cmap('tab10').colors[cc] for cc in range(3, 8)],
                                 range(nsols)):
        axis.barh(range(nfactors_to_plot),
                  np.abs(coefs[factors_to_plot]) / np.sum(np.abs(coefs)),
                  color=c)
        axis.set_yticks(range(nfactors_to_plot))
        axis.set_yticklabels(np.array(labels)[factors_to_plot],
                             **{'fontname': everything_else_font, 'size': 12})
        axis.set_xlabel('Relative Importance\n(logistic regression coefficients)',
                        **{'fontname': everything_else_font, 'size': 12})
        axis.set_xticks(axis.get_xlim())
        axis.set_xticklabels(['Low', 'High'], **{'fontname': everything_else_font})
        axis.set_title('Durham/Buyer-preferred 1',
                       **{'fontname': title_font, 'size': 15})

    plt.savefig(files_root_directory + 'important_factors.svg')
