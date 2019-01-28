from matplotlib.colors import LinearSegmentedColormap
from data_analysis.sorting_pseudo_robustness import \
    calculate_pseudo_robustness, get_robust_compromise_solutions, \
    get_influential_rdm_factors_boosted_trees, \
    get_influential_rdm_factors_logistic_regression
from data_transformation.process_decvars import process_decvars_inverse, \
    check_repeated
from data_transformation.process_rdm_objectives import *
from plotting.dec_vars_paxis import plot_dec_vars_paxis
from plotting.diagnostics import plot_utility_comparison_parallel_figures
from plotting.parallel_axis import __calculate_alphas
from plotting.pathways_plotting import *
from plotting.parallel_axis import *
import os.path
from copy import deepcopy
from multiprocessing import Pool
from functools import partial

bu_cy = LinearSegmentedColormap.from_list('BuCy', [(0, 0, 1), (0, 1, 1)])
bu_cy_r = bu_cy.reversed()

blues_hc_colors = np.array([[33., 68., 120.], [95., 141., 211.],
                            [215., 227., 244.]]) / 256
oranges_hc_colors = np.array([[170, 68., 0.], [225., 127., 42.],
                              [255., 230., 213.]]) / 256
light_greys = np.array([[0.7, 0.7, 0.7], [0.9, 0.9, 0.9]])

oranges_hc = LinearSegmentedColormap.from_list('Oranges_hc', oranges_hc_colors)
blues_hc = LinearSegmentedColormap.from_list('Blues_hc', blues_hc_colors)
light_greys_hc = LinearSegmentedColormap.from_list('Light_greys_hc',
                                                   light_greys)

oranges_hc_r = LinearSegmentedColormap.from_list('Oranges_hc_r',
                                                 oranges_hc_colors[::-1])
blues_hc_r = LinearSegmentedColormap.from_list('Blues_hc_r',
                                               blues_hc_colors[::-1])
light_greys_hc_r = LinearSegmentedColormap.from_list('Light_greys_hc_r',
                                                     light_greys[::-1])

cmaps = [oranges_hc, blues_hc]


def plot_all_paxis(objective_on_du_grouped, objective_on_wcu_grouped,
                   objective_rdm_grouped, axis_labels, files_root_directory,
                   n_wcu, robustnesses_ordered_by_sol_id,
                   utilities, brushing_robustness, brush_criteria1,
                   highlight_solutions, axes_formating=[]):
    columns_to_plot = range(6)
    color_column = 0

    # Calculate ranges
    all_to_plot = np.vstack((objective_rdm_grouped,
                             objective_on_wcu_grouped,
                             objective_on_du_grouped))
    # ranges_all = np.vstack((all_to_plot.min(axis=0),
    #                         all_to_plot.max(axis=0))).T
    ranges_all = np.array([[0.85, 0., 77, 0., 0., 0.25],
                           [1., 0.6, 720, 0.6, 0.4, 0.69]]).T

    parallel_axis([robustnesses_ordered_by_sol_id[:n_wcu],
                   robustnesses_ordered_by_sol_id[n_wcu:]], range(4), 1,
                  [oranges_hc_r, blues_hc_r], utilities, 'Robustness',
                  ['WCU Optimization', 'DU Optimization'],
                  axis_ranges=[[0, 1]] * 4,
                  axis_number_formating=['{:.0%}'] * 4,
                  brush_criteria=brushing_robustness,
                  size=(8, 3.5),
                  file_name=files_root_directory + 'robustness_paxis.svg',
                  highlight_solutions=highlight_solutions,
                  lw=2.,
                  labels_after_col=1)

    parallel_axis(
        [objective_rdm_grouped[:n_wcu], objective_rdm_grouped[n_wcu:]],
        columns_to_plot, color_column,
        [cmaps[0], cmaps[1]],
        axis_labels,
        'Estimated Objective values for Policies from WCU '
        'and DU Optimization',
        ['WCU Optimization', 'DU Optimization'],
        axis_ranges=ranges_all,
        file_name=files_root_directory + 'du_and_wcu_on_reeval.svg',
        axis_to_invert=[0],
        lw=2.,
        size=(8, 3.5),
        axis_number_formating=axes_formating,
        cbar_same_scale=True,
        highlight_solutions=highlight_solutions,
        brush_criteria=brush_criteria1
    )

    # parallel_axis([robustnesses_ordered_by_sol_id[:n_wcu],
    #                robustnesses_ordered_by_sol_id[n_wcu:]], range(4), 1,
    #               [oranges_hc_r, blues_hc_r], utilities, 'Robustness',
    #               ['WCU Optimization', 'DU Optimization'],
    #               axis_ranges=[[0, 1]] * 4,
    #               axis_number_formating=['{:.0%}'] * 4,
    #               brush_criteria=brushing_robustness,
    #               size=(8, 3.5),
    #               file_name=files_root_directory + 'robustness_paxis.svg',
    #               highlight_solutions=highlight_solutions,
    #               lw=2.,
    #               labels_after_col=1)

    # parallel_axis(
    #     [objective_rdm_grouped[:n_wcu], objective_rdm_grouped[n_wcu:]],
    #     columns_to_plot, color_column,
    #     [cmaps[0], cmaps[1]],
    #     axis_labels,
    #     'Estimated Objective values for Policies from WCU '
    #     'and DU Optimization',
    #     ['WCU Optimization', 'DU Optimization'],
    #     axis_ranges=ranges_all,
    #     file_name=files_root_directory + 'du_and_wcu_on_reeval.svg',
    #     axis_to_invert=[0],
    #     lw=2.,
    #     size=(9, 5),
    #     axis_number_formating=axes_formating,
    #     cbar_same_scale=True,
    # )


def calculate_pseudo_robustnesses(performance_criteria, objectives_by_solution,
                                  non_crashed_by_solution, rdm_factors,
                                  files_root_directory, utilities):
    robustnesses = []
    # Robustness calculation
    for u in range(len(utilities)):
        if not os.path.isfile(files_root_directory + 'data/robustness_{}.csv'
                .format(utilities[u])):
            calculate_pseudo_robustness(objectives_by_solution,
                                        non_crashed_by_solution,
                                        performance_criteria,
                                        files_root_directory,
                                        (0 + u * 6, 1 + u * 6, 4 + u * 6),
                                        utilities[u], rdm_factors,
                                        not_group_objectives=True)

        robustnesses.append(
            np.loadtxt(files_root_directory + 'data/robustness_{}.csv'
                       .format(utilities[u]), delimiter=',')
        )

    return robustnesses


def plot_pathways_utility(pathways, source_colormap_id, solution,
                          utility_to_plot, name, rdm, n_existing_sources,
                          files_root_directory, sources, infra_built, ax, ax_cb,
                          cb_pos, suffix=''):
    # # Only low and high WJLWTP had permitting times.
    # important_factors_multiple_solutions_plot(most_influential_factors_all,
    #                                           lr_coef_all, 2,
    #                                           create_labels_list(),
    #                                           files_root_directory)

    # Plot pathways
    pathways_list_utility_high = \
        get_pathways_by_utility_realization(pathways[0])
    utility_pathways_high = pathways_list_utility_high[utility_to_plot]

    utility_pathways_high_copy = utility_pathways_high

    plot_colormap_pathways(utility_pathways_high_copy, 2400,
                                     source_colormap_id, solution,
                                     rdm, n_existing_sources, ax, ax_cb, cb_pos,
                                     savefig_directory=files_root_directory, # + 'Figures/',
                                     nrealizations=1000, sources=sources,
                                     utility_name=name, year0=2015,
                                     suffix=suffix)


def plot_decision_vars(dec_vars, solutions_info, files_root_directory):
    # dec_vars_raw = np.loadtxt(files_root_directory
    #                           + 'combined_reference_sets.set',
    #                           delimiter=',')

    # dec_vars_no_rep, ix = check_repeated(dec_vars_raw)

    # plot_decvars_radar(dec_vars, [[0] * 4, [1] * 4])
    max_mins = {'Restriction Trigger': [0, 1],
                'Transfer Trigger': [0, 1],
                'Jordan Lake Allocation': [0, 0.7],
                'Infrastructure\n(long-term) Trigger': [0, 1],
                'Insurance Trigger': [0, 1],
                'Annual Contingency\nFund Contribution': [0, 0.1]}

    fig, axes = plt.subplots(2, len(max_mins) / 2, figsize=(8.5, 5.5))#, sharey=True)
    colors = cm.get_cmap('Accent').colors
    plt.subplots_adjust(left=0.08, bottom=0.11, right=0.97,
                        top=0.94, hspace=0.35)

    for s, c, sol_name in zip(solutions_info['ids'], colors,
                                        solutions_info['labels']):
        dec_vars_processed = process_decvars_inverse(
            dec_vars[s], ['Durham', 'OWASA', 'Raleigh', 'Cary'],
            {'Restriction Trigger': 0, 'Transfer Trigger': 4,
             'Insurance Trigger': 15, 'Annual Contingency\nFund Contribution': 11, 'Infrastructure\n(long-term) Trigger': 23,
             'Jordan Lake Allocation': 7}
        )

        decvars_order = ['Restriction Trigger', 'Transfer Trigger',
                         'Jordan Lake Allocation', 'Infrastructure\n(long-term) Trigger',
                         'Insurance Trigger', 'Annual Contingency\nFund Contribution']

        plot_dec_vars_paxis(dec_vars_processed, max_mins, axes, c,
                            decvars_order, sol_name)

    # axes[0].set_ylabel(sol_name)
    for ax in axes.ravel():
        ax.set_ylim([0, 1])
    # axes[-1, -1].legend()

    jla_ylim = [0, 0.69]
    axes[0, 2].set_ylim(jla_ylim)
    axes[0, 2].set_yticks(jla_ylim)
    axes[0, 2].set_yticklabels(['{:.00%}'.format(x) for x in jla_ylim],
                               {'fontname': 'Open Sans Condensed', 'size': 11})
    acfc_ylim = [0, 0.1]
    axes[1, 2].set_ylim(acfc_ylim)
    axes[1, 2].set_yticks(acfc_ylim)
    axes[1, 2].set_yticklabels(['{:.00%}'.format(x) for x in acfc_ylim],
                               {'fontname': 'Open Sans Condensed', 'size': 11})
    rof_ylim = [0, 1]
    rof_ytick_labels = ['Low\n(High Usage)', 'High\n(Low Usage)']
    axes[0, 0].set_ylim(rof_ylim)
    axes[0, 0].set_yticks(rof_ylim)
    axes[0, 0].set_yticklabels(rof_ytick_labels,
                               {'fontname': 'Open Sans Condensed', 'size': 11})
    axes[1, 0].set_ylim(rof_ylim)
    axes[1, 0].set_yticks(rof_ylim)
    axes[1, 0].set_yticklabels(rof_ytick_labels,
                               {'fontname': 'Open Sans Condensed', 'size': 11})

    for i in [0, 1]:
        for j in [0, 2]:
            axes[i, j].tick_params(top='off', bottom='off', left='off',
                                   right='off',
                                   labelleft='on', labelbottom='on')

    # plt.show()
    plt.savefig(files_root_directory + 'decvars.svg')

    # gmm_cluster(dec_vars, files_root_directory)

    # ix_complimentary, new_passes = find_complimentary_solution(s, objectives_by_solution, non_crashed_by_solution,
    #                                                            performance_criteria, apply_criteria_on_objs,
    #                                                            np.array(['max', 'min', 'min', 'min', 'min', 'min']))
    #
    # most_influential_factors_all, pass_fail_all, non_crashed_rdm_all, \
    # lr_coef_all = get_influential_rdm_factors_boosted_trees(objectives_by_solution,
    #                                                         non_crashed_by_solution,
    #                                                         performance_criteria,
    #                                                         files_root_directory,
    #                                                         apply_criteria_on_objs, rdm_factors,
    #                                                         solutions=[ix_complimentary-1],
    #                                                         plot=True, n_trees=25, tree_depth=2)

    # # Only low and high WJLWTP had permitting times.
    # important_factors_multiple_solutions_plot(most_influential_factors_all,
    #                                           lr_coef_all, 2,
    #                                           create_labels_list(),
    #                                           files_root_directory)


def get_best_worse_rdm_objectives_based(objectives_original,
                                        performance_criteria,
                                        apply_criteria_on_objs, invert):
    objectives = np.multiply(objectives_original[:, apply_criteria_on_objs],
                             invert)
    objs_min, objs_ptp = objectives.min(axis=0), objectives.ptp(axis=0)

    performance_criteria_norm = \
        (np.array(performance_criteria) - objs_min) / (objs_ptp + 1e-6)
    objectives_norm_minus_criteria = \
        (objectives - objs_min) / (objs_ptp + 1e-6) - performance_criteria_norm

    objectives_norm_fail_by = np.multiply(objectives_norm_minus_criteria,
                                          objectives_norm_minus_criteria > 0)
    distance_from_criteria_fail = np.sqrt(np.multiply(objectives_norm_fail_by,
                                                      objectives_norm_fail_by).sum(
        axis=1))

    objectives_norm_pass_by = np.multiply(objectives_norm_minus_criteria,
                                          objectives_norm_minus_criteria < 0)
    distance_from_criteria_pass = np.sqrt(np.multiply(objectives_norm_pass_by,
                                                      objectives_norm_pass_by).sum(
        axis=1))

    best_rdm, worse_rdm = np.argmax(distance_from_criteria_pass), \
                          np.argmax(distance_from_criteria_fail)

    return best_rdm, worse_rdm


def get_best_worse_rdm_factors_based(rdm_factors, important_factors, pass_fail,
                                     feature_importances):
    n_pass_fail = len(pass_fail)
    corrected_factors = []
    for f in important_factors[:2]:
        sf = np.argsort(rdm_factors[:, f])
        if sum(pass_fail[sf][:int(n_pass_fail / 2)]) > \
                sum(pass_fail[sf][int(n_pass_fail / 2):]):
            cf = deepcopy(rdm_factors[:, f])
        else:
            cf = max(rdm_factors[:, f]) - deepcopy(rdm_factors[:, f])

        corrected_factors.append((cf - cf.min()) / cf.ptp())

    elip_norm_sq = (np.square((feature_importances[important_factors[0]] *
                               corrected_factors[0])) +
                    np.square((feature_importances[important_factors[1]] *
                               corrected_factors[1])))

    return np.argmin(elip_norm_sq), np.argmax(elip_norm_sq)


def get_rdm_closest_to(rdm_factors, features, values, feature_importances):
    feature_importances_partial = deepcopy(feature_importances) + 0.2
    feature_importances_partial[np.argsort(feature_importances)[:-4]] = 0
    rdm_factors_important = deepcopy(rdm_factors[:, features])
    normed_values = []
    for i in range(len(values)):
        col = rdm_factors[:, i]
        normed_values.append((values[i] - col.min()) / col.ptp())
        rdm_factors_important[:, i] = (col - col.min()) / col.ptp()

    dist = np.zeros(len(rdm_factors))
    for f, col, nv in zip(features, rdm_factors_important.T, normed_values):
        dist += np.square((col - nv)) * feature_importances_partial[f] ** 2

    return np.argmin(dist)


def plot_scenario_discovery_solution_utility(utility_name,
                                             objectives_by_solution,
                                             non_crashed_by_solution,
                                             performance_criteria,
                                             files_root_directory,
                                             rdm_factors,
                                             ntrees, tree_depth,
                                             apply_criteria_on_objs,
                                             su, ax, features_speficig_rdm=(),
                                             values_speficig_rdm=()):
    cmap = cm.get_cmap('RdGy')
    from_middle = 0.6
    dist_between_pass_fail_colors = 0.75
    region_alpha = 0.6
    shift_colors = -0.1

    s, u = su
    most_influential_factors, pass_fail, \
    non_crashed_rdm, feature_importances, cmap_mod = \
        get_influential_rdm_factors_boosted_trees(
            objectives_by_solution,
            non_crashed_by_solution,
            performance_criteria,
            np.array(apply_criteria_on_objs) + u * 6, rdm_factors, ax, s,
            not_group_objectives=True,
            plot=True,
            n_trees=ntrees,
            tree_depth=tree_depth,
            name_suffix=utility_name,
            cmap=cmap,
            dist_between_pass_fail_colors=dist_between_pass_fail_colors,
            region_alpha=region_alpha, shift_colors=shift_colors)
    # files_root_directory=files_root_directory)

    best_rdm, worse_rdm = get_best_worse_rdm_factors_based(
        non_crashed_rdm, most_influential_factors,
        pass_fail, feature_importances)

    specific_rdm = -1
    if len(features_speficig_rdm) > 0:
        specific_rdm = get_rdm_closest_to(non_crashed_rdm, features_speficig_rdm,
                           values_speficig_rdm, feature_importances)

    # best_rdm, worse_rdm = get_best_worse_rdm_objectives_based(
    #     objectives_by_solution[s], performance_criteria,
    #     np.array(apply_criteria_on_objs) + 6 * u, invert)

    x_best_rdm, y_best_rdm = non_crashed_rdm[
        best_rdm, [most_influential_factors[0],
                   most_influential_factors[1]]]
    x_worse_rdm, y_worse_rdm = non_crashed_rdm[
        worse_rdm, [most_influential_factors[0],
                    most_influential_factors[1]]]
    x_specific_rdm, y_specific_rdm = non_crashed_rdm[
        specific_rdm, [most_influential_factors[0],
                    most_influential_factors[1]]]

    if ax is not None:
        ax.scatter([x_best_rdm], [y_best_rdm],
                   c=[cmap_mod(0.5 + from_middle)], edgecolor='black',
                   linewidths=1.25, marker='*', s=150, label='Most Favorable',
                   clip_on=False)
        ax.scatter([x_worse_rdm], [y_worse_rdm],
                   c=[cmap_mod(0.5 - from_middle)], edgecolor='black',
                   linewidths=1.25, marker='*', s=150, label='Most Unfavorable',
                   clip_on=False)
        ax.scatter([x_specific_rdm], [y_specific_rdm],
                   c='b', edgecolor='black',
                   linewidths=1.25, marker='*', s=150,
                   label='80% of demand \ngrowth projection',
                   clip_on=False)
        # pos = list(ax.get_position().bounds)
        # pos[2] = .225
        # ax.set_position(pos)
        # plt.tight_layout()
        # plt.savefig('{}/scenario_discovery_solution_{}_{}.png'.format(
        #     files_root_directory, s, u))
        # plt.close()

    return best_rdm, worse_rdm, specific_rdm


def load_pathways(non_crashed_by_solution, non_repeated_dec_var_ix, solution,
                  best_worse_rdm, files_root_directory):
    non_crashed_rdms = len(non_crashed_by_solution[solution])

    rdm_to_load = []
    for rdm in best_worse_rdm:
        rdm_to_load.append(
            np.arange(non_crashed_rdms)[non_crashed_by_solution[solution]][
                rdm]
        )

    pathways = load_pathways_solution(
        files_root_directory + '../re-evaluation_against_du/pathways/',
        non_repeated_dec_var_ix[solution],
        rdm_to_load
    )

    return pathways


def plot_pathways_solutions_rdms(infra_built, utilities, n_existing_sources,
                                 source_colormap_id, files_root_directory,
                                 sources, cb_pos, pathways_solution_rdm_ax):

    utility = pathways_solution_rdm_ax[3]
    plot_pathways_utility([pathways_solution_rdm_ax[0]],
                                    source_colormap_id,
                                    pathways_solution_rdm_ax[1],
                                    utility, utilities[utility],
                                    pathways_solution_rdm_ax[2],
                                    n_existing_sources,
                                    files_root_directory, sources, infra_built,
                                    pathways_solution_rdm_ax[5],
                                    pathways_solution_rdm_ax[6], cb_pos,
                                    suffix=pathways_solution_rdm_ax[4])

    # plot_pathways_utility([pathways[1]], solution, utility, utility_name,
    #                       best_worse_rdm[1], n_existing_sources,
    #                       files_root_directory, sources,
    #                       suffix='worse_rdm')


def plot_scenario_discovery_pathways(utilities, objectives_by_solution,
                                     source_colormap_id,
                                     non_crashed_by_solution,
                                     performance_criteria,
                                     files_root_directory, rdm_factors_orig,
                                     n_existing_sources, apply_criteria_on_objs,
                                     non_repeated_dec_var_ix,
                                     most_robust_for_each, sources,
                                     solution_names, sols_to_plot):
    rdm_factors = deepcopy(rdm_factors_orig)

    ntrees = 100
    tree_depth = 2
    utilitiy_ids = range(1, len(utilities))
    sols = np.array([most_robust_for_each[i] for i in sols_to_plot])
    sols_names = np.array([solution_names[i] for i in sols_to_plot])
    best_rdm_worse_rdm_all_sols = []
    fig, axes = plt.subplots(len(sols), len(utilitiy_ids), figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    for s, sn, axes_row in zip(sols, sols_names, axes):
        su_all = []
        for u in utilitiy_ids:
            su_all.append([s, u])

        best_rdm_worse_rdm = []
        rdm_specifics = range(rdm_factors.shape[1])
        rdm_specific_values = np.ones(len(rdm_specifics))
        rdm_specific_values[0] = 0.8

        for su, ax in zip(su_all, axes_row):
            best_rdm_worse_rdm.append(
              plot_scenario_discovery_solution_utility(
                  utilities[su[1]], objectives_by_solution,
                  non_crashed_by_solution, performance_criteria,
                  files_root_directory, rdm_factors, ntrees, tree_depth,
                  apply_criteria_on_objs, su, ax,
                  features_speficig_rdm=rdm_specifics,
                  values_speficig_rdm=rdm_specific_values
              )
            )

        best_rdm_worse_rdm_all_sols.append(best_rdm_worse_rdm)
        title0 = axes_row[0].get_ylabel()
        axes_row[0].set_ylabel('{}\n{}'.format(sn, title0),
                               {'fontname': 'Open Sans Condensed'})

    axes[1, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -1.),
                      ncol=5,
                      prop={'family': 'Open Sans Condensed', 'size': 12})

    for ax, u in zip(axes[0], np.array(utilities)[utilitiy_ids]):
        ax.set_title(u, {'fontname': 'Gill Sans MT', 'size': 12})

    # fig.suptitle('Scenario Discovery for the Two Buyer Preferred Solutions',
    #              **{'fontname': 'Gill Sans MT', 'size': 16})
    # plt.savefig(files_root_directory + '/scenario_discovery.svg')
    # plt.show()

    # Load pathways before plotting. Necessary to determine what infrastructure
    # is built across all realizations to have as few sources as possible in
    # the plots' colorbar.
    plt.close()

    utilities_ids_pathways = [1, 3]

    for s, sn, best_rdm_worse_rdm in zip(sols, solution_names[sols_to_plot],
                                         best_rdm_worse_rdm_all_sols):
        print "Working on pathways for solution {}".format(s)
        fig, axes = plt.subplots(len(utilities_ids_pathways) * 2 + 1, 3, figsize=(6, 8)) # 3 is the number of RDMs
        fig.set_size_inches(10, 12)
        fig.suptitle('Pathways for Solution \"{}\" for 3 SOWs'.format(sn),
                     **{'fontname': 'Gill Sans MT', 'size': 16})
        pathways = []
        pathways_solutions_rdm = []
        ax_key = axes[len(utilities_ids_pathways) * 2, 2]
        for bw, u, axes_row in zip(best_rdm_worse_rdm,
                                   utilities_ids_pathways * 2,
                                   axes[range(1, len(utilities_ids_pathways)*2, 2)]):
            print bw
            pathways_sol_rdm = load_pathways(non_crashed_by_solution,
                                             non_repeated_dec_var_ix, s, bw,
                                             files_root_directory)
            pathways += pathways_sol_rdm
            pathways_solutions_rdm.append([pathways_sol_rdm[0], s, bw[0],
                                           u, 'best_rdm', axes_row[0],
                                           ax_key])
            pathways_solutions_rdm.append([pathways_sol_rdm[2], s, bw[2],
                                           u, 'specific_rdm', axes_row[1],
                                           ax_key])
            pathways_solutions_rdm.append([pathways_sol_rdm[1], s, bw[1],
                                           u, 'worse_rdm', axes_row[2],
                                           ax_key])

        # for bw, axes_col in zip(best_rdm_worse_rdm,
        #                         utilities_ids_pathways * 2,
        #                         axes[range(0, len(utilities_ids_pathways) * 2,
        #                                      2)].T):


        infra_built = np.unique(np.vstack(pathways)[:, 3])
        cb_pos = [0.2, 0.23, 0.7, 0.01]
        axes[-1, 0].remove()
        axes[-1, 1].remove()

        for psr in pathways_solutions_rdm:
            plot_pathways_solutions_rdms(infra_built, utilities,
                                         n_existing_sources,
                                         source_colormap_id,
                                         files_root_directory,
                                         sources, cb_pos, psr)

        for ax in axes[:len(utilities_ids_pathways) * 2 - 1].ravel():
            ax.get_xaxis().set_visible(False)

        for ax in axes[:len(utilities_ids_pathways) * 2, 1:3].ravel():
            ax.get_yaxis().set_visible(False)

        for ax in axes[len(utilities_ids_pathways) * 2 - 1]:
            for l in ax.get_xticklabels():
                l.set_fontproperties('Open Sans Condensed')

        titles = ['Most Favorable\nSOW', '80% demand growth\nmultiplier SOW',
                  'Least Favorable\nSOW']
        for ax, t in zip(axes[0], titles):
            ax.set_title(t, {'fontname': 'Gill Sans MT', 'size': 12})

        for ax, u in zip(axes.T[0], np.array(utilities)[utilitiy_ids]):
            ax.set_ylabel('{}\nRealization'.format(u),
                          {'fontname': 'Open Sans Condensed',
                           'fontweight': 'semibold'})

        # plt.savefig(files_root_directory +
        #             '/complete_scenario_discovery_pathways{}.svg'.format(s))
        plt.show()
        # plt.close()

        print 'Done with pathways for solution {}'.format(s)


def plot_diagnostics():
    ''' Time interval to plot '''
    # dt = 40
    # w0 = 1
    # weeks = range(52 * w0, 13 + 52*(w0 + dt))
    weeks = range(2344)

    dec_vars = np.loadtxt('solutions_not_crashed.csv', delimiter=',')
    rest_triggers = [[]] * 4
    transfer_triggers = [[]] * 4
    inf_triggers = [[]] * 4
    ins_triggers = [[]] * 4
    rest_triggers[1] = dec_vars[:, 0];
    rest_triggers[0] = dec_vars[:, 1];
    rest_triggers[3] = dec_vars[:, 2];
    rest_triggers[2] = dec_vars[:, 3];
    transfer_triggers[1] = dec_vars[:, 4];
    transfer_triggers[0] = dec_vars[:, 5];
    transfer_triggers[3] = dec_vars[:, 6];
    transfer_triggers[2] = np.zeros(len(transfer_triggers[0]))
    ins_triggers[1] = dec_vars[:, 15]
    ins_triggers[0] = dec_vars[:, 16]
    ins_triggers[3] = dec_vars[:, 17]
    ins_triggers[2] = dec_vars[:, 18]
    inf_triggers[1] = dec_vars[:, 23];
    inf_triggers[0] = dec_vars[:, 24];
    inf_triggers[3] = dec_vars[:, 25];
    inf_triggers[2] = dec_vars[:, 26];

    # plot_utility_comparison_parallel_io(weeks, [270, 315], rest_triggers, transfer_triggers, ins_triggers, inf_triggers, output_directory='Diagnostics_buyer_solutions/', rdms=['', 1512, 1610, 1644])
    # plot_utility_comparison_parallel_io(weeks, [270, 315], rest_triggers, transfer_triggers, ins_triggers, inf_triggers, output_directory='Diagnostics_buyer_solutions/')
    plot_utility_comparison_parallel_figures(weeks, [270, 315], rest_triggers, transfer_triggers, ins_triggers,
        inf_triggers, output_directory='Diagnostics_buyer_solutions/', rdms=[1512, 1610, 1644], figsize=(4.9, 2), nprocs=2)

def create_plots():

    files_root_directory = 'F:/Dropbox/Bernardo/Research/WaterPaths_results/rdm_results/'
    # files_root_directory = '/media/DATA/Dropbox/Bernardo/Research/WaterPaths_results/rdm_results/'
    n_rdm_scenarios = 2000
    n_solutions = 368
    n_objectives = 20
    n_objectives_per_utility_plus_jla = 6
    n_utilities = 4
    sources = np.array(['Lake Michie & Little River Res. (Durham)',
                        'Falls Lake',
                        'Wheeler-Benson Lakes',
                        'Stone Quarry',
                        'Cane Creek Reservoir',
                        'University Lake',
                        'Jordan Lake',
                        'Little River Reservoir (Raleigh)',#7 0
                        'Richland Creek Quarry',#8 1
                        'Teer Quarry',#9 2
                        'Neuse River Intake',#10 3
                        'Dummy Node',#11 4
                        'Low StoneQuarry Expansion',#12 5
                        'High Stone Quarry Expansion',#13 6
                        'University Lake Expansion',#14 7
                        'Low Lake Michie Expansion',#15 8
                        'High Lake  Michie Expansion',#16 9
                        'Falls Lake  Reallocation',#17 10
                        'Low Reclaimed Water System',#18 11
                        'High Reclaimed Water System',#19 12
                        'Low WJLWTP',#20 13
                        'High WJLWTP',#21 14
                        'Cary WTP upgrade 1',#22 15
                        'Cary WTP upgrade 2',#23 16
                        'Cane Creek Reservoir Expansion',#24 17
                        'Status-quo']) #25 18

    # Load decision variables
    dec_vars_raw = np.loadtxt(files_root_directory
                              + 'data/combined_reference_sets.set',
                              delimiter=',')

    # Look for repeated solutions -- ix is index of non-repeated
    dec_vars, objectives_rdm, non_repeated_dec_var_ix = check_repeated(
        dec_vars_raw[:, :-6],
        dec_vars_raw[:, -6:])

    n_wcu = sum(non_repeated_dec_var_ix < 249)

    # Load objectives for each RDM scenario organized by solution (list
    # of matrices)
    objectives_by_solution, non_crashed_by_solution = \
        load_objectives(files_root_directory, n_solutions, n_rdm_scenarios,
                        n_objectives, n_utilities)  # , processed=False)

    objectives_by_solution = [objectives_by_solution[i] for i in
                              non_repeated_dec_var_ix]
    non_crashed_by_solution = [non_crashed_by_solution[i] for i in
                               non_repeated_dec_var_ix]

    # Load RDM factors
    rdm_utilities = np.loadtxt(files_root_directory +
                               'data/rdm_utilities_reeval.csv', delimiter=',')
    rdm_dmp = np.loadtxt(files_root_directory + 'data/rdm_dmp_reeval.csv',
                         delimiter=',')
    rdm_sources_meaningful = [0] + range(15, 51)
    rdm_water_sources = np.loadtxt(files_root_directory +
                                   'data/rdm_water_sources_reeval.csv',
                                   delimiter=',',
                                   usecols=rdm_sources_meaningful)
    rdm_factors = np.hstack((rdm_utilities, rdm_dmp, rdm_water_sources))
    rdm_inflows = np.loadtxt(files_root_directory + 'data/rdm_inflows.csv')

    # Back-calculate objectives for each solution as if objectives had been
    # calculated with 1,000 * 2,000 = 2e6 fully specified worlds.
    objectives_rdm = back_calculate_objectives(objectives_by_solution,
                                               n_objectives_per_utility_plus_jla,
                                               n_utilities)

    # Load objectives on either DU or WCU space, as they came out of Borg
    objective_on_wcu = load_on_du_objectives(files_root_directory,
                                             on='wcu',
                                             ix=non_repeated_dec_var_ix)
    objective_on_du = load_on_du_objectives(files_root_directory,
                                            on='du', ix=non_repeated_dec_var_ix)

    axis_labels = ['Reliability\n(average)', 'Restriction Frequency\n(average)',
                   'Infrastructure Net\nPresent Value (average)',
                   'Total Annual Financial\nCost (average)',
                   'Annual Financial\nCost Fluctuation (average)',
                   'Jordan Lake Allocation\n(as decision variable)'] * n_utilities
    dataset_names = ('WCU Optimization', 'DU Optimization')

    # Three different brushing criteria relaxing reliability
    brush_criteria1 = {0: [0.99, 1.0], 1: [0.0, 0.2], 4: [0.0, 0.10]}
    brush_criteria2 = {0: [0.98, 1.0], 1: [0.0, 0.2], 4: [0.0, 0.10]}
    brush_criteria3 = {0: [0.97, 1.0], 1: [0.0, 0.2], 4: [0.0, 0.10]}

    # Parallel axis plot axes formatting
    axes_formating = ['{:.0%}', '{:.0%}', '${:,.0f}MM', '{:.0%}',
                      '{:.0%}', '{:.0%}']

    # Group solutions
    objective_rdm_grouped = group_objectives(objectives_rdm,
                                             ['max', 'min', 'min', 'min',
                                              'min', 'min'])
    objective_on_wcu_grouped = group_objectives(objective_on_wcu,
                                                ['max', 'min', 'min', 'min',
                                                 'min', 'min'])

    # Get rid of two really low reliability solutions that are shifting
    #       the blue and making all lines look dark except the corresponding few.
    objective_on_wcu_grouped = objective_on_wcu_grouped[
        objective_on_wcu_grouped[:, 0] > 0.92]

    objective_on_du_grouped = group_objectives(objective_on_du,
                                               ['max', 'min', 'min', 'min',
                                                'min', 'min'])

    # # Retrieve solutions that met brushing criteria
    alphas = __calculate_alphas(objective_rdm_grouped, brush_criteria1,
                                base_alpha=1.)
    good_sols = np.argwhere(alphas == 1.)
    print 'not brushed performance ', good_sols

    performance_criteria = (0.990, 0.2, 0.1)
    apply_criteria_on_objs = [0, 1, 4]
    utilities = ['OWASA', 'Durham', 'Cary', 'Raleigh']
    robustnesses = calculate_pseudo_robustnesses(performance_criteria,
                                                 objectives_by_solution,
                                                 non_crashed_by_solution,
                                                 rdm_factors,
                                                 files_root_directory,
                                                 utilities)

    rob_beta = True
    robust_for_all, robustnesses_ordered_by_sol_id = \
        get_robust_compromise_solutions(robustnesses, 0.75)

    np.savetxt(files_root_directory + 'robustnesses.csv',
               robustnesses_ordered_by_sol_id, delimiter=',',
               header='OWASA,Durham,Cary,Raleigh')

    hcm = cm.get_cmap('tab10').colors
    highlighted_colors = [hcm[9], hcm[3], hcm[8], hcm[2]]
    most_robust_for_each = np.argmax(robustnesses_ordered_by_sol_id, axis=0)
    highlighted_labels = ['O', 'D/B1', 'C/S', 'R/B2']

    # PLOT APPROX. ROBUSTNESS BAR CHART
    # pseudo_robustness_plot(utilities, robustnesses,
    #                        [oranges_hc(0.4), blues_hc(0.4)],
    #                        files_root_directory,
    #                        beta=rob_beta)
    # pseudo_robustness_plot(utilities, robustnesses,
    #                        [oranges_hc(0.4), blues_hc(0.4)],
    #                        files_root_directory,
    #                        plot_du=False, beta=rob_beta)
    most_robust_for_each[0] = 299
    most_robust_for_each[2] = 351
    most_robust_for_each_np = np.array(most_robust_for_each) - n_wcu
    print most_robust_for_each
    highlight_solutions = [{}, {'ids': most_robust_for_each_np,
                                'colors': highlighted_colors,
                                'labels': highlighted_labels}]
    # highlight_solutions_rob_compromise = [{}, {'ids' : [270],
    #                        'colors' : [highlighted_colors[3]],
    #                        'labels' : ['RC']}]

    # Plot objectives highlighted solutions
    columns_to_plot = range(6)
    color_column = 0

    # Calculate ranges
    all_to_plot = np.vstack((objective_rdm_grouped,
                             objective_on_wcu_grouped,
                             objective_on_du_grouped))
    # ranges_all = np.vstack((all_to_plot.min(axis=0),
    #                         all_to_plot.max(axis=0))).T
    ranges_all = np.array([[0.85, 0., 77, 0., 0., 0.25],
                           [1., 0.6, 720, 0.6, 0.4, 0.6]]).T

    brushing_robustness = dict(zip(range(4), zip([1.1] * 4,
                                                 robustnesses_ordered_by_sol_id[
                                                     robust_for_all[
                                                         0]] - 0.001)))

    # Plot all parallel axis plots
    # plot_all_paxis(objective_on_du_grouped, objective_on_wcu_grouped,
    #                objective_rdm_grouped, axis_labels, files_root_directory,
    #                n_wcu, robustnesses_ordered_by_sol_id, utilities, brushing_robustness,
    #                brush_criteria1, highlight_solutions, axes_formating)

    utilities_labels = deepcopy(utilities)
    utilities_labels[1] = 'Durham\n(Overall-performing solution)'
    # pseudo_robustness_plot(utilities, robustnesses,
    #                        [oranges_hc(0.4), blues_hc(0.4)],
    #                        files_root_directory, nwcu=n_wcu,
    #                        highlight_solutions=highlight_solutions,
    #                        upper_xlim=100,
    #                        plot_du=True)

    brushing_robustness = dict(zip(range(4), zip([1.1] * 4,
                                                 robustnesses_ordered_by_sol_id[
                                                     robust_for_all[
                                                         0]] - 0.001)))
    source_colormap_id = np.array([[14, 0],
                                    [20, 2],
                                    [21, 3],
                                    [9, 4],
                                    [15, 5],
                                    [16, 6],
                                    [18, 8],
                                    [19, 9],
                                    [22, 13],
                                    [23, 14],
                                    [len(sources) - 1, 15],
                                    [7, 16],
                                    [8, 17],
                                    [17, 18]])

    solution_names = np.array(['', 'Buyer Preferred I', 'Supplier Preferred',
                      'Buyer Preferred II'])

    plot_scenario_discovery_pathways(utilities, objectives_by_solution,
                                     source_colormap_id,
                                     non_crashed_by_solution,
                                     performance_criteria,
                                     files_root_directory,
                                     rdm_factors, 7, apply_criteria_on_objs,
                                     non_repeated_dec_var_ix,
                                     most_robust_for_each, sources,
                                     solution_names, [1, 3])

    # re-create dictionary to plot only buyer solutions
    solutions_info = {}
    sols_dv_to_plot = [1, 3]
    for k in highlight_solutions[1].keys():
        solutions_info[k] = [highlight_solutions[1][k][s] for s in sols_dv_to_plot]

    solutions_info['ids'] = [most_robust_for_each[s] for s in sols_dv_to_plot]

    plot_decision_vars(dec_vars, solutions_info, files_root_directory)
    np.savetxt(files_root_directory + 'solutions_not_crashed.csv', dec_vars,
               delimiter=',')

    # plot_diagnostics()


if __name__ == '__main__':
    create_plots()
