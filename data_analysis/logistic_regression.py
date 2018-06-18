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
                                       performance_criteria, plot=False):

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
                                     non_crashed_rdm, sol_number)
        return most_influential_factors, pass_fail, non_crashed_rdm, lr.coef_[0]
    else:
        return -np.ones(nrdms, dtype=int), pass_fail, \
               [False] * nrdms, np.zeros(nrdms)


def boosted_trees_classification(objectives_by_solution, rdm_factors, sol_number,
                                 performance_criteria, plot=False,
                                 n_trees=100, tree_depth=3):

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
                                     non_crashed_rdm, sol_number, gbc)
        return most_influential_factors, pass_fail, non_crashed_rdm, gbc.feature_importances_
    else:
        return -np.ones(nrdms, dtype=int), pass_fail, \
               [False] * nrdms, np.zeros(nrdms)


def logistic_regression_plot(most_influential_factors, pass_fail,
                             non_crashed_rdm, sol_number, classifier,
                             cmap=cm.get_cmap('coolwarm'), from_middle=0.35):
    most_influential_pair = most_influential_factors[-2:]

    # plot logistic regression
    labels = create_labels_list()
    # print 'Most influencial RDM factors: \n\t{}\n\t{}'.format(
    #     labels[most_influential_pair[0]], labels[most_influential_pair[1]]
    # )

    ax = plt.subplot(111)
    ax.scatter(non_crashed_rdm[pass_fail, most_influential_pair[0]],
               non_crashed_rdm[pass_fail, most_influential_pair[1]],
               c=cmap(0.5 - from_middle), label='Pass')
    ax.scatter(non_crashed_rdm[pass_fail == False, most_influential_pair[0]],
               non_crashed_rdm[pass_fail == False, most_influential_pair[1]],
               c=cmap(0.5 + from_middle), label='Fail')
    ax.set_xlabel(labels[most_influential_pair[0]])
    ax.set_ylabel(labels[most_influential_pair[1]])
    ax.set_title('RDM for solution {}'.format(sol_number))

    # Add legend and shrink current axis by 20% so that legend is not
    # outside plot.
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # x_min, x_max = ax.get_xlim()
    # y_min, y_max = ax.get_ylim()
    #
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
    #                      np.arange(y_min, y_max, (y_max - y_min) / 100))

    z = classifier.predict_proba(non_crashed_rdm)[:, 1]
    z = z.reshape((len(non_crashed_rdm), len(non_crashed_rdm)))
    ax.contourf(non_crashed_rdm[:, most_influential_pair[0]],
                non_crashed_rdm[:, most_influential_pair[1]],
                z, cmap=cmap, alpha=.8)

    plt.show()
