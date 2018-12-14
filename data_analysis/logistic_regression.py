from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from data_transformation.process_rdm_objectives import create_labels_list
from sklearn.linear_model import LogisticRegression

CRASHED_OBJ_VALUE = 10000

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    color_scale = np.concatenate(
        (np.linspace(minval, 0.5, n / 2), np.linspace(0.5, maxval, n / 2)),
        axis=None
    )
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0., b=1.),
        cmap(color_scale)
    )
    return new_cmap


def prepare_data_for_classification(objectives_by_solution, rdm_factors,
                                    performance_criteria):
    # Load one solution and check which RDM re-evaluations did not crash
    non_crashed_sols = objectives_by_solution[:, 0] != CRASHED_OBJ_VALUE
    objectives_by_solution = objectives_by_solution[non_crashed_sols]
    nobjs = len(performance_criteria)
    nrdms = rdm_factors.shape[1]

    # Label solutions as pass and fail
    pass_fail = objectives_by_solution[:, 0] > performance_criteria[0]
    for i in range(1, nobjs):
        pass_fail *= (objectives_by_solution[:, i] < performance_criteria[i])
    pass_fail = pass_fail == True

    non_crashed_rdm = rdm_factors[non_crashed_sols]

    return non_crashed_rdm, pass_fail, nrdms


def logistic_regression_classification(objectives_by_solution, rdm_factors,
                                       sol_number, ax,
                                       performance_criteria, plot=False,
                                       files_root_directory=''):
    non_crashed_rdm, pass_fail, nrdms = prepare_data_for_classification(
        objectives_by_solution, rdm_factors,
        performance_criteria)

    if len(np.unique(pass_fail)) == 2:
        # Perform logistic regression on rdm factors and pass/fail labels
        lr = LogisticRegression()
        lr.fit(non_crashed_rdm, pass_fail)

        # get most influential pair of factors
        most_influential_factors = np.argsort(np.abs(lr.coef_))[0]

        if plot:
            factor_mapping_plot(most_influential_factors, pass_fail,
                                non_crashed_rdm, sol_number, lr, ax,
                                files_root_directory=files_root_directory)
        return most_influential_factors, pass_fail, non_crashed_rdm, lr.coef_[0]
    else:
        return -np.ones(nrdms, dtype=int), pass_fail, \
               np.array([[False] * nrdms] * 2000), np.zeros(nrdms)


def boosted_trees_classification(objectives_by_solution, rdm_factors,
                                 sol_number, performance_criteria, ax,
                                 plot=False, n_trees=100, tree_depth=3,
                                 files_root_directory='', name_suffix='',
                                 cmap=cm.get_cmap('coolwarm'),
                                 dist_between_pass_fail_colors=0.7,
                                 region_alpha=0.2, shift_colors=0):
    non_crashed_rdm, pass_fail, nrdms = \
        prepare_data_for_classification(objectives_by_solution, rdm_factors,
                                        performance_criteria)

    if len(np.unique(pass_fail)) == 2:
        # Perform logistic regression on rdm factors and pass/fail labels
        gbc = GradientBoostingClassifier(n_estimators=n_trees,
                                         learning_rate=0.1,
                                         max_depth=tree_depth)
        # gbc = RandomForestClassifier(n_estimators=n_trees, n_jobs=2)
        gbc.fit(non_crashed_rdm, pass_fail)

        # get most influential pair of factors
        most_influential_factors = np.argsort(gbc.feature_importances_)[::-1]

        if plot:
            cmap_mod, feature_importances = \
                factor_mapping_plot(most_influential_factors, pass_fail,
                                     non_crashed_rdm, sol_number, gbc, ax,
                                     files_root_directory=files_root_directory,
                                     name_suffix=name_suffix, cmap=cmap,
                                     dist_between_pass_fail_colors=dist_between_pass_fail_colors,
                                     region_alpha=region_alpha,
                                     shift_colors=shift_colors)
        return most_influential_factors, pass_fail, non_crashed_rdm, \
               feature_importances, cmap_mod
    else:
        return -np.ones(nrdms, dtype=int), pass_fail, \
               [False] * nrdms, np.zeros(nrdms), None


def factor_mapping_plot(most_influential_factors, pass_fail,
                        non_crashed_rdm, sol_number, classifier, ax,
                        files_root_directory='', name_suffix='',
                        cmap=cm.get_cmap('coolwarm'),
                        dist_between_pass_fail_colors=0.7,
                        region_alpha=0.2, shift_colors=0):
    most_influential_pair = most_influential_factors[:2]

    labels = create_labels_list()

    # fig, ax = plt.subplots(figsize=(5, 4))
    feature_importances = deepcopy(classifier.feature_importances_)
    feature_importances /= np.sum(feature_importances)

    ax.set_xlabel('{} ({:.2f}%)'.format(
        labels[most_influential_pair[0]],
        feature_importances[most_influential_pair[0]] * 100),
        {'fontname': 'Open Sans Condensed', 'size': 12}
    )
    ax.set_ylabel('{} ({:.2f}%)'.format(
        labels[most_influential_pair[1]],
        feature_importances[most_influential_pair[1]] * 100),
        {'fontname': 'Open Sans Condensed', 'size': 12}
    )
    ax.set_title('RDM for Solution {} {}'.format(sol_number, name_suffix),
                 {'fontname': 'Gill Sans MT', 'size': 16})

    # Add legend and shrink current axis by 20% so that legend is not
    # outside plot.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    x_data = non_crashed_rdm[:, most_influential_pair[0]]
    y_data = non_crashed_rdm[:, most_influential_pair[1]]

    x_min, x_max = (x_data.min(), x_data.max())
    y_min, y_max = (y_data.min(), y_data.max())

    xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                         np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))

    dummy_points = np.ones((len(xx.ravel()), len(most_influential_factors)))
    dummy_points[:, most_influential_pair[0]] = xx.ravel()
    dummy_points[:, most_influential_pair[1]] = yy.ravel()

    z = classifier.predict_proba(dummy_points)[:, 1]
    z[z < 0] = 0.
    z = z.reshape(xx.shape)

    pass_color = 0.5 * (1. + dist_between_pass_fail_colors) + shift_colors
    fail_color = 0.5 * (1. - dist_between_pass_fail_colors) + shift_colors

    cmap_mod = truncate_colormap(cmap, minval=fail_color, maxval=pass_color)

    ax.contourf(xx, yy, z, 2, cmap=cmap_mod, alpha=region_alpha,
                     vmin=0, vmmax=1)
    ax.scatter(x_data[pass_fail], y_data[pass_fail], linewidths=0.5,
               c='none', edgecolor=cmap_mod(1.), label='Pass', s=3)
    ax.scatter(x_data[pass_fail == False], y_data[pass_fail == False],
               linewidths=1, c='none', edgecolor=cmap_mod(0.),
               label='Fail', s=3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              prop={'family': 'Open Sans Condensed', 'size': 12})

    xlims = np.array([np.around(xx.min(), 2), np.around(xx.max(), 2)])
    ylims = np.array([np.around(yy.min(), 2), np.around(yy.max(), 2)])
    # ax.set_xlim(xlims)
    # ax.set_ylim(ylims)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ndivs = 4
    xticks = [xlims[0] + xlims.ptp() * 1. / ndivs * p for p in range(ndivs)] \
             + [xlims[1]]
    xtick_labels = ['{:.1f}%'.format(p * 100) for p in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels,
                       {'fontname': 'Open Sans Condensed', 'size': 11})

    ndivs = 6
    yticks = [ylims[0] + ylims.ptp() * 1. / ndivs * p for p in range(ndivs)] \
             + [ylims[1]]
    ytick_labels = ['{:.1f}%'.format(p * 100) for p in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels,
                       {'fontname': 'Open Sans Condensed', 'size': 11})

    # # plt.show()
    # if len(files_root_directory) > 0:
    #     plt.savefig('{}/scenario_discovery_solution_{}_{}.png'.format(
    #         files_root_directory, sol_number, name_suffix))
    #     # plt.clf()
    #     # plt.close()
    # # else:
    # #     plt.show()
    # #     # plt.clf()
    # #     # plt.close()

    return cmap_mod, feature_importances
