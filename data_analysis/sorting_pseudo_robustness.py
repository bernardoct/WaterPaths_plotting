import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib import cm

from data_analysis.logistic_regression import logistic_regression_classification, boosted_trees_classification
from data_transformation.process_rdm_objectives import group_objectives, \
    create_labels_list

def calculate_pseudo_robustness_uniform(pass_fail):
    return 1.* sum(pass_fail * 1) / len(pass_fail)


def plot_beta(x, p, pass_fail, ax, cmap=cm.get_cmap('cool'), from_middle=0.35):
    xu = np.linspace(0., 1., 200)

    ax.scatter(x[pass_fail], p[pass_fail], c=cmap(0.5 - from_middle), s=0.01)
    ax.scatter(x[pass_fail == False], p[pass_fail == False],
               c=cmap(0.5 + from_middle), s=0.01)


def calculate_pseudo_robustness_beta(pass_fail, rdm_factors, base_parameter,
                                     lows, highs, means, plot=False):

    if not np.array_equal(rdm_factors.shape, (len(pass_fail), len(means))):
        raise Exception(('rdm_factors shape is {} but pass_fail and means have '
               + 'lengths of {} and {}').format(rdm_factors.shape,
                                               len(pass_fail), len(means)))

    if plot:
        fig, ax = plt.subplots()

    pseudo_robustness = 0.
    for rdm_factor, mean, low, high in zip(rdm_factors.T, means, lows, highs):
        rdm_ptp = high - low
        m = (mean - low) / rdm_ptp

        if max(rdm_factor) <= m:
            a = base_parameter
            b = 1. / m * (a - 1 - m * a + 2 * m)
        else:
            b = base_parameter
            a = ((b - 2.) * m + 1) / (1. - m)

        rdm_factor_norm = (rdm_factor - low) / rdm_ptp
        p_rdm_beta = beta.pdf(rdm_factor_norm, a, b)

        pseudo_robustness += sum(p_rdm_beta[pass_fail]) / sum(p_rdm_beta)

        if plot:
            plot_beta(rdm_factor_norm, p_rdm_beta, pass_fail, ax)

    if plot:
        plt.show()

    return pseudo_robustness / len(means)


def get_influential_rdm_factors_logistic_regression(objectives_by_solution, non_crashed_by_solution,
                                                    performance_criteria, files_root_directory,
                                                    apply_criteria_on_objs, rdm_factors,
                                                    not_group_objectives=False, solutions=(), plot=False):

    nsols = len(objectives_by_solution)
    if len(solutions) == 0:
        solutions = range(nsols)

    most_influential_factors_all, pass_fail_all, non_crashed_rdm_all, \
    lr_coef_all = [], [], [], []

    for sol_number in solutions:
        print 'Performing scenario discovery for solution {}'.format(sol_number)

        if not_group_objectives:
            objectives = objectives_by_solution[sol_number][:, apply_criteria_on_objs]
        else:
            objectives = group_objectives(
                        objectives_by_solution[sol_number],
                        ['max', 'min', 'min', 'min',
                         'min', 'min']
                    )[:, apply_criteria_on_objs]

        # objectives_normalized = (objectives - objectives.min(axis=0)) / objectives.ptp(axis=0)

        most_influential_factors, pass_fail, non_crashed_rdm, lr_coef = \
            logistic_regression_classification(
                objectives,
                # objectives_normalized,
                rdm_factors[non_crashed_by_solution[sol_number]],
                sol_number, performance_criteria,
                plot=plot,
                files_root_directory=files_root_directory if plot else ''
            )

        most_influential_factors_all.append(most_influential_factors)
        pass_fail_all.append(pass_fail)
        non_crashed_rdm_all.append(non_crashed_rdm)
        lr_coef_all.append(lr_coef)

    return most_influential_factors_all, pass_fail_all, \
           non_crashed_rdm_all, lr_coef_all


def get_influential_rdm_factors_boosted_trees(objectives_by_solution, non_crashed_by_solution,
                                              performance_criteria,
                                              apply_criteria_on_objs, rdm_factors,
                                              not_group_objectives=False, solutions=(),
                                              n_trees=100, tree_depth=2, plot=False, name_suffix='',
                                              files_root_directory=''):

    nsols = len(objectives_by_solution)
    if len(solutions) == 0:
        solutions = range(nsols)

    most_influential_factors_all, pass_fail_all, non_crashed_rdm_all, \
    lr_coef_all = [], [], [], []

    for sol_number in solutions:
        # Load RDM files in a single table

        print 'Performing scenario discovery for solution {}{}'.format(sol_number, name_suffix)

        if not_group_objectives:
            objectives = objectives_by_solution[sol_number][:, apply_criteria_on_objs]
        else:
            objectives = group_objectives(
                        objectives_by_solution[sol_number],
                        ['max', 'min', 'min', 'min',
                         'min', 'min']
                    )[:, apply_criteria_on_objs]

        # objectives_normalized = (objectives - objectives.min(axis=0)) / objectives.ptp(axis=0)

        most_influential_factors, pass_fail, non_crashed_rdm, lr_coef, ax = \
            boosted_trees_classification(
                objectives,
                # objectives_normalized,
                rdm_factors[non_crashed_by_solution[sol_number]],
                sol_number, performance_criteria,
                plot=plot, n_trees=n_trees, tree_depth=tree_depth,
                files_root_directory=files_root_directory if plot else '',
                name_suffix=name_suffix
            )

        most_influential_factors_all.append(most_influential_factors)
        pass_fail_all.append(pass_fail)
        non_crashed_rdm_all.append(non_crashed_rdm)
        lr_coef_all.append(lr_coef)

    return most_influential_factors_all, pass_fail_all, \
           non_crashed_rdm_all, lr_coef_all, ax


def influential_factors_plot(objectives_by_solution, non_crashed_by_solution,
                                performance_criteria, files_root_directory,
                                apply_criteria_on_objs, rdm_factors):
    most_influential_factors_all, pass_fail_all, \
    non_crashed_rdm_all, lr_coef_all = \
        get_influential_rdm_factors_logistic_regression(objectives_by_solution,
                                                        non_crashed_by_solution,
                                                        performance_criteria,
                                                        files_root_directory,
                                                        apply_criteria_on_objs,
                                                        rdm_factors)

    all_pass = []
    plot = []
    for s in range(len(objectives_by_solution)):
        unique = np.unique(pass_fail_all[s])
        if len(unique) == 1 and unique[0]:
            all_pass.append(s)

        if len(unique) == 1 and not unique[0]:
            plot.append(False)
        else:
            plot.append(True)

    fig, ax = plt.subplots(figsize=(9, 6))
    factor_data = np.abs(np.vstack(lr_coef_all))[plot]
    ax.imshow(factor_data[np.argsort(factor_data[:, 0])],
              cmap='OrRd',
              interpolation='nearest',
              aspect='auto')
    ax.set_xlabel('Factor', **{'fontname':'CMU Bright', 'size' : 13})
    ax.set_ylabel('Solution', **{'fontname':'CMU Bright', 'size' : 13})

    importances = factor_data.sum(axis=0)
    importance_order = np.argsort(importances)[::-1]

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    bars_range = np.arange(len(importance_order), dtype=float)
    ax2.bar(bars_range, importances[importance_order] / importances[0])
    ax2.set_xticks(bars_range)
    ax2.xaxis.set_tick_params(rotation=90)
    labels = np.array(create_labels_list())[importance_order]
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Factor', **{'fontname':'CMU Bright', 'size' : 13})
    ax2.set_ylabel('Overall Importance', **{'fontname':'CMU Bright', 'size' : 13})

    plt.show()


def calculate_pseudo_robustness(objectives_by_solution, non_crashed_by_solution,
                                performance_criteria, files_root_directory,
                                apply_criteria_on_objs, utility, rdm_factors,
                                not_group_objectives=False):
    pseudo_robustness_uniform = []
    pseudo_robustness_beta = []

    most_influential_factors_all, pass_fail_all, \
    non_crashed_rdm_all, lr_coef_all = \
        get_influential_rdm_factors_logistic_regression(objectives_by_solution,
                                                        non_crashed_by_solution,
                                                        performance_criteria,
                                                        files_root_directory,
                                                        apply_criteria_on_objs,
                                                        rdm_factors,
                                                        not_group_objectives=
                                                        not_group_objectives)

    for most_influential_factors, pass_fail, non_crashed_rdm, lr_coef in \
            zip(most_influential_factors_all, pass_fail_all,
                non_crashed_rdm_all, lr_coef_all):

        # Number of important uncertainty factors -- scenario discovery
        n_factors = np.sum(np.sort(np.abs(lr_coef) + 1e-6) / np.sum(np.abs(lr_coef) + 1e-6)
                           > 0.1)

        # Parameters to create beta
        lows = np.array(
            [0.5, 1.0, 0.6, 0.6] + [0.9] * 4 + [0.8] + [0.75, 1.0] * 18)
        highs = np.array(
            [2., 1.2, 1.0, 1.4] + [1.1] * 4 + [1.2] + [1.5, 1.2] * 18)
        means = np.array(
            [1., 1.0, 1.0, 1.0] + [1.0] * 4 + [1.0] + [1.0, 1.0] * 18)

        # Calculate robustness if there was at least one pass
        r1 = calculate_pseudo_robustness_uniform(pass_fail)
        if r1 != 0 and r1 != 1:
            r2 = calculate_pseudo_robustness_beta(
                pass_fail,
                non_crashed_rdm[:, most_influential_factors[-n_factors:]],
                2.5,
                lows[most_influential_factors[-n_factors:]],
                highs[most_influential_factors[-n_factors:]],
                means[most_influential_factors[-n_factors:]]#, plot=True
            )
        else:
            r2 = r1

        pseudo_robustness_uniform.append(r1)
        pseudo_robustness_beta.append(r2)

    sorted_uniform = np.argsort(pseudo_robustness_uniform)[::-1]
    sorted_beta = np.argsort(pseudo_robustness_beta)[::-1]
    robustness = np.vstack((
        sorted_uniform,
        np.array(pseudo_robustness_uniform)[sorted_uniform],
        sorted_beta,
        np.array(pseudo_robustness_beta)[sorted_beta])
    )

    np.savetxt(files_root_directory + 'robustness_{}.csv'.format(utility),
               robustness.T, header='uniform rank, uniform, beta rank, beta',
               delimiter=',')


def get_robust_compromise_solutions(robustnesses, percentile, beta=False):
    robustness_col = 3 if beta else 1
    percentile_sols = np.array([r[0, robustness_col] * percentile for r in robustnesses])
    robustnesses_uniform = np.array([np.array(
        sorted(r, key=lambda x: x[0]))[:, robustness_col] for r in robustnesses]).T

    robust_for_all_utilities = []
    for ru in robustnesses_uniform:
        if (ru > percentile_sols).all():
            robust_for_all_utilities.append(True)
        else:
            robust_for_all_utilities.append(False)

    print 'Found {} solutions in the top {} percentile for all utilities.'\
        .format(sum(robust_for_all_utilities), percentile * 100)

    return np.where(robust_for_all_utilities)[0], robustnesses_uniform


