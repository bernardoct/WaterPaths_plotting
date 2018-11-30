import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.ensemble import GradientBoostingClassifier

from data_transformation.process_rdm_objectives import create_labels_list
from sklearn.linear_model import LogisticRegression

CRASHED_OBJ_VALUE = 10000


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


def logistic_regression_classification(objectives_by_solution, rdm_factors, sol_number,
                                       performance_criteria, plot=False, files_root_directory=''):
    non_crashed_rdm, pass_fail, nrdms = prepare_data_for_classification(objectives_by_solution, rdm_factors,
                                                                        performance_criteria)

    if len(np.unique(pass_fail)) == 2:
        # Perform logistic regression on rdm factors and pass/fail labels
        lr = LogisticRegression()
        lr.fit(non_crashed_rdm, pass_fail)

        # get most influential pair of factors
        most_influential_factors = np.argsort(np.abs(lr.coef_))[0]

        if plot:
            logistic_regression_plot(most_influential_factors, pass_fail,
                                     non_crashed_rdm, sol_number, lr, files_root_directory=files_root_directory)
        return most_influential_factors, pass_fail, non_crashed_rdm, lr.coef_[0]
    else:
        return -np.ones(nrdms, dtype=int), pass_fail, \
               np.array([[False] * nrdms] * 2000), np.zeros(nrdms)


def boosted_trees_classification(objectives_by_solution, rdm_factors, sol_number,
                                 performance_criteria, plot=False,
                                 n_trees=100, tree_depth=3,
                                 files_root_directory='', name_suffix= ''):
    non_crashed_rdm, pass_fail, nrdms = \
        prepare_data_for_classification(objectives_by_solution, rdm_factors,
                                        performance_criteria)

    if len(np.unique(pass_fail)) == 2:
        # Perform logistic regression on rdm factors and pass/fail labels
        gbc = GradientBoostingClassifier(n_estimators=n_trees,
                                         learning_rate=0.1,
                                         max_depth=tree_depth)
        gbc.fit(non_crashed_rdm, pass_fail)

        # get most influential pair of factors
        most_influential_factors = np.argsort(gbc.feature_importances_)

        if plot:
            logistic_regression_plot(most_influential_factors, pass_fail,
                                     non_crashed_rdm, sol_number, gbc, files_root_directory=files_root_directory,
                                     name_suffix=name_suffix)
        return most_influential_factors, pass_fail, non_crashed_rdm, gbc.feature_importances_
    else:
        return -np.ones(nrdms, dtype=int), pass_fail, \
               [False] * nrdms, np.zeros(nrdms)


def logistic_regression_plot(most_influential_factors, pass_fail,
                             non_crashed_rdm, sol_number, classifier,
                             cmap=cm.get_cmap('coolwarm'), from_middle=0.35, files_root_directory='', name_suffix=''):
    most_influential_pair = most_influential_factors[-2:]

    # plot logistic regression
    labels = create_labels_list()
    # print 'Most influencial RDM factors: \n\t{}\n\t{}'.format(
    #     labels[most_influential_pair[0]], labels[most_influential_pair[1]]
    # )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(labels[most_influential_pair[0]],
                       {'fontname': 'Open Sans Condensed', 'size': 12})
    ax.set_ylabel(labels[most_influential_pair[1]],
                       {'fontname': 'Open Sans Condensed', 'size': 12})
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
    z = z.reshape(xx.shape)
    # cs = ax.contourf(xx, yy, 1. - z, 3, cmap=cmap, alpha=.7)
    cs = ax.contourf(xx, yy, 1. - z, 2,
                     colors=[cmap(0.10), cmap(0.5), cmap(0.9)], #cmap(0.32), cmap(0.78), cmap(0.9)],
                     alpha=.5)
    # ax.contour(cs, levels=[0, 0.5, 1.], colors='r')
    ax.scatter(x_data[pass_fail], y_data[pass_fail],
               c='none', edgecolor=cmap(0.5 - from_middle), label='Pass', s=2)
    ax.scatter(x_data[pass_fail == False], y_data[pass_fail == False],
               c='none', edgecolor=cmap(0.5 + from_middle), label='Fail', s=2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              prop={'family': 'Open Sans Condensed', 'size': 12})

    xlims = np.array([np.around(xx.min(), 2), np.around(xx.max(), 2)])
    ylims = np.array([np.around(yy.min(), 2), np.around(yy.max(), 2)])
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ndivs = 4
    xticks = [xlims[0] + xlims.ptp() * 1. / ndivs * p for p in range(ndivs)] \
             + [xlims[1]]
    xtick_labels = ['{}%'.format(p * 100) for p in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels,
                       {'fontname': 'Open Sans Condensed', 'size': 11})

    ndivs = 6
    yticks = [ylims[0] + ylims.ptp() * 1. / ndivs * p for p in range(ndivs)] \
             + [ylims[1]]
    ytick_labels = ['{}%'.format(p * 100) for p in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels,
                       {'fontname': 'Open Sans Condensed', 'size': 11})

    # plt.show()

    if len(files_root_directory) > 0:
        plt.savefig('{}/scenario_discovery_solution_{}_{}.png'.format(files_root_directory, sol_number, name_suffix))
        plt.clf()
        plt.close()
    else:
        plt.show()
        # plt.clf()
        # plt.close()
