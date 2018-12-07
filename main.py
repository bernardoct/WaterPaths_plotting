from matplotlib.colors import LinearSegmentedColormap
from data_analysis.sorting_pseudo_robustness import \
    calculate_pseudo_robustness, get_robust_compromise_solutions, \
    get_influential_rdm_factors_boosted_trees, \
    get_influential_rdm_factors_logistic_regression
from data_transformation.process_decvars import process_decvars_inverse, \
    check_repeated
from data_transformation.process_rdm_objectives import *
from plotting.dec_vars_paxis import plot_dec_vars_paxis
from plotting.parallel_axis import __calculate_alphas
from plotting.pathways_plotting import *
from plotting.parallel_axis import *
import os.path
from copy import deepcopy

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
                   objective_rdm_grouped, axis_labels, dataset_names,
                   files_root_directory, n_wcu, axes_formating=[]):
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
        size=(9, 5),
        axis_number_formating=axes_formating,
        cbar_same_scale=True,
    )


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


def plot_pathways_utility(pathways, solution, utility_to_plot, name,
                          rdm, n_existing_sources, suffix=''):
    # # Only low and high WJLWTP had permitting times.
    # important_factors_multiple_solutions_plot(most_influential_factors_all,
    #                                           lr_coef_all, 2,
    #                                           create_labels_list(),
    #                                           files_root_directory)

    ninfra = np.array([0, 6, 0, 4])[utility_to_plot]

    # Plot pathways
    pathways_list_utility_high = \
        get_pathways_by_utility_realization(pathways[0])
    utility_pathways_high = pathways_list_utility_high[utility_to_plot]

    # replace infrastructure id by construction order
    construction_order = get_infra_order(utility_pathways_high)
    # utility_pathways_high_copy = \
    #     convert_pathways_from_source_id_to_construction_id(
    #         utility_pathways_high,
    #         construction_order)
    utility_pathways_high_copy = utility_pathways_high

    plot_colormap_pathways(utility_pathways_high_copy, 2400,
                           solution, rdm, n_existing_sources,
                           savefig_directory=files_root_directory,
                           # + 'Figures/',
                           nrealizations=1000,
                           ninfra=ninfra, sources=sources,
                           construction_order=construction_order,
                           utility_name=name, year0=2015, suffix=suffix)


def plot_decision_vars():
    # dec_vars_raw = np.loadtxt(files_root_directory
    #                           + 'combined_reference_sets.set',
    #                           delimiter=',')

    # dec_vars_no_rep, ix = check_repeated(dec_vars_raw)

    dec_vars_processed = process_decvars_inverse(dec_vars[robust_for_all],
                                                 ['Durham', 'OWASA', 'Raleigh',
                                                  'Cary'],
                                                 {'Restriction\nTrigger': 0,
                                                  'Transfer\nTrigger': 4,
                                                  'Insurance\nTrigger': 15,
                                                  'ACFC': 11,
                                                  'Long term\nROF': 23,
                                                  'Jordan Lake\nAllocation': 7})

    # plot_decvars_radar(dec_vars, [[0] * 4, [1] * 4])
    max_mins = {'Restriction\nTrigger': [0, 1],
                'Transfer\nTrigger': [0, 1],
                'Insurance\nTrigger': [0, 1],
                'ACFC': [0, 0.1],
                'Long term\nROF': [0, 1],
                'Jordan Lake\nAllocation': [0, 0.7]}

    plot_dec_vars_paxis(dec_vars_processed, (2, 3), max_mins)

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


def get_best_worse_rdm(objectives_original, performance_criteria, apply_criteria_on_objs, invert):
    objectives = np.multiply(objectives_original[:, apply_criteria_on_objs], invert)
    objs_min, objs_ptp = objectives.min(axis=0), objectives.ptp(axis=0)

    performance_criteria_norm = (np.array(performance_criteria) - objs_min) / (objs_ptp + 1e-6)
    objectives_norm_minus_criteria = (objectives - objs_min) / (objs_ptp + 1e-6) - performance_criteria_norm

    objectives_norm_fail_by = np.multiply(objectives_norm_minus_criteria, objectives_norm_minus_criteria > 0)
    distance_from_criteria_fail = np.sqrt(np.multiply(objectives_norm_fail_by, objectives_norm_fail_by).sum(axis=1))

    objectives_norm_pass_by = np.multiply(objectives_norm_minus_criteria, objectives_norm_minus_criteria < 0)
    distance_from_criteria_pass = np.sqrt(np.multiply(objectives_norm_pass_by, objectives_norm_pass_by).sum(axis=1))

    best_rdm, worse_rdm = np.argmax(distance_from_criteria_pass), np.argmax(distance_from_criteria_fail)

    return best_rdm, worse_rdm


def plot_scenario_discovery_pathways(utilities, objectives_by_solution, non_crashed_by_solution, performance_criteria,
                                     invert, files_root_directory, rdm_factors, n_existing_sources):

    ntrees = 100
    tree_depth = 4
    for s in [most_robust_for_each[1], most_robust_for_each[3]]:
        for u in range(1, len(utilities)):
            best_rdm, worse_rdm = get_best_worse_rdm(objectives_by_solution[s], performance_criteria,
                                                     np.array(apply_criteria_on_objs) + 6 * u, invert)

            most_influential_factors_all_u, pass_fail_all_u, non_crashed_rdm_all_u, lr_coef_all_u, ax = \
                get_influential_rdm_factors_boosted_trees(
                    objectives_by_solution,
                    non_crashed_by_solution,
                    performance_criteria,
                    np.array(apply_criteria_on_objs) + u * 6, rdm_factors,
                    not_group_objectives=True,
                    solutions=[s],
                    plot=True,
                    n_trees=ntrees,
                    tree_depth=tree_depth,
                    name_suffix=utilities[u],
                    files_root_directory=files_root_directory)

            x_best_rdm, y_best_rdm = rdm_factors[best_rdm, [most_influential_factors_all_u[0][-2],
                                                            most_influential_factors_all_u[0][-1]]]
            x_worse_rdm, y_worse_rdm = rdm_factors[worse_rdm, [most_influential_factors_all_u[0][-2],
                                                               most_influential_factors_all_u[0][-1]]]
            if ax is not None:
                ax.scatter([x_best_rdm, x_worse_rdm], [y_best_rdm, y_worse_rdm], c=cm.get_cmap('tab10').colors[2], s=50)

            best_rdm_original_numbering = \
                np.arange(len(non_crashed_by_solution[s]))[non_crashed_by_solution[s]][best_rdm]
            worse_rdm_original_numbering = \
                np.arange(len(non_crashed_by_solution[s]))[non_crashed_by_solution[s]][worse_rdm]
            pathways = load_pathways_solution(files_root_directory +
                                              '../re-evaluation_against_du/pathways/',
                                              non_repeated_dec_var_ix[s],
                                              [best_rdm_original_numbering, worse_rdm_original_numbering])

            plot_pathways_utility([pathways[0]], s, u, utilities[u], best_rdm, n_existing_sources, suffix='best_rdm')

            plot_pathways_utility([pathways[1]], s, u, utilities[u], worse_rdm, n_existing_sources, suffix='worse_rdm')


if __name__ == '__main__':
    # files_root_directory = 'F:/Dropbox/Bernardo/Research/WaterPaths_results/rdm_results/'
    files_root_directory = '/media/DATA/Dropbox/Bernardo/Research/WaterPaths_results/rdm_results/'
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

    # # Plot all parallel axis plots
    # plot_all_paxis(objective_on_du_grouped, objective_on_wcu_grouped,
    #                objective_rdm_grouped, axis_labels,
    #                dataset_names, files_root_directory, n_wcu, axes_formating)

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
    #                        beta=True)
    # pseudo_robustness_plot(utilities, robustnesses,
    #                        [oranges_hc(0.4), blues_hc(0.4)],
    #                        files_root_directory,
    #                        plot_du=False, beta=True)
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

    plot_scenario_discovery_pathways(utilities, objectives_by_solution,
                                     non_crashed_by_solution, performance_criteria, [-1, 1, 1], files_root_directory,
                                     rdm_factors, 7)
