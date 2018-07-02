from matplotlib.colors import LinearSegmentedColormap

from data_analysis.clustering import gmm_cluster
from data_analysis.complimentary_solutions import calc_pass_fail
from data_analysis.sorting_pseudo_robustness import \
    calculate_pseudo_robustness, get_robust_compromise_solutions, \
    get_influential_rdm_factors_boosted_trees, \
    get_influential_rdm_factors_logistic_regression
from data_transformation.process_decvars import process_decvars_inverse, check_repeated
from data_transformation.process_rdm_objectives import *
from plotting.dec_vars_paxis import plot_dec_vars_paxis
from plotting.parallel_axis import __calculate_alphas
from plotting.pathways_plotting import *
from plotting.parallel_axis import *
import os.path

from plotting.dec_vars_radar import plot_decvars_radar
from plotting.robustness_bar_chart import pseudo_robustness_plot, important_factors_multiple_solutions_plot
from sklearn import mixture

bu_cy = LinearSegmentedColormap.from_list('BuCy', [(0, 0, 1), (0, 1, 1)])
bu_cy_r = bu_cy.reversed()

blues_hc_colors = np.array([[33., 68., 120.], [95., 141., 211.],
                            [215., 227., 244.]]) / 256
oranges_hc_colors = np.array([[170, 68., 0.], [225., 127., 42.],
                              [255., 230., 213.]]) / 256
light_greys = np.array([[0.7, 0.7, 0.7], [0.9, 0.9, 0.9]])

oranges_hc = LinearSegmentedColormap.from_list('Oranges_hc', oranges_hc_colors)
blues_hc = LinearSegmentedColormap.from_list('Blues_hc', blues_hc_colors)
light_greys_hc = LinearSegmentedColormap.from_list('Light_greys_hc', light_greys)

cmaps = [oranges_hc, blues_hc]


def plot_all_paxis(objective_on_du_grouped, objective_on_wcu_grouped,
                   objective_rdm_grouped, axis_labels, dataset_names,
                   files_root_directory):
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

    # parallel_axis([objective_on_du_grouped[249:]],
    #               columns_to_plot,
    #               color_column,
    #               [cmaps[1]],
    #               axis_labels,
    #               'DU out of optimization',
    #               [dataset_names[1]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'du_on_du.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3)
    #               )
    # parallel_axis([objective_on_wcu_grouped[:249]],
    #               columns_to_plot, color_column,
    #               [cmaps[0]],
    #               axis_labels,
    #               'WCU out of optimization',
    #               [dataset_names[0]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'wcu_on_wcu.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3)
    #               )
    # parallel_axis([objective_rdm_grouped[249:]],
    #               columns_to_plot, color_column,
    #               [cmaps[1]],
    #               axis_labels,
    #               'DU on Re-evaluation Space',
    #               [dataset_names[1]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'du_on_reeval.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3)
    #               )
    # parallel_axis([objective_rdm_grouped[:249]],
    #               columns_to_plot, color_column,
    #               [cmaps[0]],
    #               axis_labels,
    #               'WCU on Re-evaluation Space',
    #               [dataset_names[0]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'wcu_on_reeval.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3)
    #               )
    parallel_axis([objective_on_du_grouped[249:], objective_rdm_grouped[249:]],
                  columns_to_plot, color_column,
                  [light_greys_hc, cmaps[1]],
                  axis_labels,
                  'DU on Re-evaluation Space',
                  ['DU Optimization', dataset_names[1]],
                  axis_ranges=ranges_all,
                  file_name=files_root_directory + 'du_on_reeval_back_du_on_du.svg',
                  axis_to_invert=[0],
                  lw=2.,
                  size=(9, 3)
                  )
    parallel_axis([objective_on_wcu_grouped[:249], objective_rdm_grouped[:249]],
                  columns_to_plot, color_column,
                  [light_greys_hc, cmaps[0]],
                  axis_labels,
                  'WCU on Re-evaluation Space',
                  ['WCU Optimization', dataset_names[0]],
                  axis_ranges=ranges_all,
                  file_name=files_root_directory + 'wcu_on_reeval_back_wcu_on_wcu.svg',
                  axis_to_invert=[0],
                  lw=2.,
                  size=(9, 3)
                  )
    # parallel_axis((objective_on_wcu_grouped[:249], objective_on_wcu_grouped[249:]),
    #               columns_to_plot, color_column,
    #               cmaps,
    #               axis_labels,
    #               'DU and WCU on WCU Space',
    #               dataset_names,
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'all_on_wcu.svg',
    #               axis_to_invert=[0],
    #               lw=2.
    #               )
    # parallel_axis((objective_on_du_grouped[:249], objective_on_du_grouped[249:]),
    #               columns_to_plot, color_column,
    #               cmaps,
    #               axis_labels,
    #               'DU and WCU on DU Space',
    #               dataset_names,
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'all_on_du.svg',
    #               axis_to_invert=[0],
    #               lw=2.
    #               )
    parallel_axis((objective_rdm_grouped[:249], objective_rdm_grouped[249:]),
                  columns_to_plot, color_column,
                  cmaps,
                  axis_labels,
                  'DU and WCU on Re-evaluation Space',
                  dataset_names,
                  axis_ranges=ranges_all,
                  file_name=files_root_directory + 'all_reeval_wcu_du.svg',
                  axis_to_invert=[0],
                  lw=2.
                  )
    # parallel_axis((objective_rdm_grouped[:249], objective_rdm_grouped[249:]),
    #               columns_to_plot, color_column,
    #               cmaps,
    #               axis_labels,
    #               'DU and WCU on Re-evaluation Space',
    #               dataset_names,
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory +
    #                      'all_reeval_wcu_du_brushed_99_20_10.svg',
    #               axis_to_invert=[0],
    #               brush_criteria=brush_criteria1,
    #               lw=2.
    #               )
    # parallel_axis((objective_rdm_grouped[:249], objective_rdm_grouped[249:]),
    #               columns_to_plot, color_column,
    #               cmaps,
    #               axis_labels,
    #               'DU and WCU on Re-evaluation Space',
    #               dataset_names,
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory +
    #                      'all_reeval_wcu_du_brushed_98_20_10.svg',
    #               axis_to_invert=[0],
    #               brush_criteria=brush_criteria2,
    #               lw=2.
    #               )
    # parallel_axis((objective_rdm_grouped[:249], objective_rdm_grouped[249:]),
    #               columns_to_plot, color_column,
    #               cmaps,
    #               axis_labels,
    #               'DU and WCU on Re-evaluation Space',
    #               dataset_names,
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory +
    #                      'all_reeval_wcu_du_brushed_97_20_10.svg',
    #               axis_to_invert=[0],
    #               brush_criteria=brush_criteria3,
    #               lw=2.
    #               )
    # parallel_axis([objective_rdm_grouped[249:]],
    #               columns_to_plot, color_column,
    #               [cmaps[1]],
    #               axis_labels,
    #               'DU on Re-evaluation Space',
    #               [dataset_names[1]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'du_on_reeval_brushed_99_20_10.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3),
    #               brush_criteria=brush_criteria1,
    #               )
    # parallel_axis([objective_rdm_grouped[249:]],
    #               columns_to_plot, color_column,
    #               [cmaps[1]],
    #               axis_labels,
    #               'DU on Re-evaluation Space',
    #               [dataset_names[1]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'du_on_reeval_brushed_98_20_10.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3),
    #               brush_criteria=brush_criteria2,
    #               )
    # parallel_axis([objective_rdm_grouped[249:]],
    #               columns_to_plot, color_column,
    #               [cmaps[1]],
    #               axis_labels,
    #               'DU on Re-evaluation Space',
    #               [dataset_names[1]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'du_on_reeval_brushed_97_20_10.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3),
    #               brush_criteria=brush_criteria3,
    #               )
    # parallel_axis([objective_rdm_grouped[:249]],
    #               columns_to_plot, color_column,
    #               [cmaps[0]],
    #               axis_labels,
    #               'WCU on Re-evaluation Space',
    #               [dataset_names[0]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'wcu_on_reeval_brushed_99_20_10.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3),
    #               brush_criteria=brush_criteria1,
    #               )
    # parallel_axis([objective_rdm_grouped[:249]],
    #               columns_to_plot, color_column,
    #               [cmaps[0]],
    #               axis_labels,
    #               'WCU on Re-evaluation Space',
    #               [dataset_names[0]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'wcu_on_reeval_brushed_98_20_10.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3),
    #               brush_criteria=brush_criteria2,
    #               )
    # parallel_axis([objective_rdm_grouped[:249]],
    #               columns_to_plot, color_column,
    #               [cmaps[0]],
    #               axis_labels,
    #               'WCU on Re-evaluation Space',
    #               [dataset_names[0]],
    #               axis_ranges=ranges_all,
    #               file_name=files_root_directory + 'wcu_on_reeval_brushed_97_20_10.svg',
    #               axis_to_invert=[0],
    #               lw=2.,
    #               size=(9, 3),
    #               brush_criteria=brush_criteria3,
    #               )


def calculate_pseudo_robustnesses(performance_criteria, objectives_by_solution,
                                  non_crashed_by_solution, rdm_factors,
                                  files_root_directory, utilities):
    robustnesses = []
    # Robustness calculation
    for u in range(len(utilities)):
        if not os.path.isfile(files_root_directory + 'robustness_{}.csv'
                .format(utilities[u])):
            calculate_pseudo_robustness(objectives_by_solution,
                                        non_crashed_by_solution,
                                        performance_criteria,
                                        files_root_directory,
                                        (0 + u * 6, 1 + u * 6, 4 + u * 6),
                                        utilities[u], rdm_factors,
                                        not_group_objectives=True)

        robustnesses.append(
            np.loadtxt(files_root_directory + 'robustness_{}.csv'
                       .format(utilities[u]), delimiter=',')
        )

    return robustnesses


if __name__ == '__main__':
    files_root_directory = 'F:/Dropbox/Bernardo/Research/WaterPaths_results/' \
                           'rdm_results/'
    # files_root_directory = '/media/DATA//Dropbox/Bernardo/Research/WaterPaths_results/' \
    #                        'rdm_results/'
    n_rdm_scenarios = 2000
    n_solutions = 368
    n_objectives = 20
    n_objectives_per_utility_plus_jla = 6
    n_utilities = 4
    sources = np.array(['Lake Michie & Little River Res. (Durham)',
                        'Falls Lake',
                        'Wheeler-Benson\nLakes',
                        'Stone Quarry',
                        'Cane Creek\nReservoir',
                        'University Lake',
                        'Jordan Lake',
                        'Little River\nReservoir (Raleigh)',
                        'Richland\nCreek Quarry',
                        'Teer Quarry',
                        'Neuse River\nIntake',
                        'Dummy Node',
                        'Low StoneQuarry\nExpansion',
                        'High Stone\nQuarry Expansion',
                        'University Lake\nExpansion',
                        'Low Lake Michie\nExpansion',
                        'High Lake \nMichie Expansion',
                        'Falls Lake \nReallocation',
                        'Low Reclaimed\nWater System',
                        'High Reclaimed\nWater System',
                        'Low WJLWTP',
                        'High WJLWTP',
                        'Cary WTP\nupgrade 1',
                        'Cary WTP\nupgrade 2',
                        'Cane Creek\nReservoir Expansion',
                        'Status-quo'])

    # Load decision variables
    dec_vars_raw = np.loadtxt(files_root_directory
                              + 'combined_reference_sets.set',
                              delimiter=',')

    # Look for repeated solutions -- ix is index of non-repeated
    dec_vars, objectives_rdm, ix = check_repeated(dec_vars_raw[:, :-6],
                                                  dec_vars_raw[:, -6:])

    # Load objectives for each RDM scenario organized by solution (list
    # of matrixes)
    objectives_by_solution, non_crashed_by_solution = \
        load_objectives(files_root_directory, n_solutions, n_rdm_scenarios,
                        n_objectives, n_utilities)  # , processed=False)

    objectives_by_solution = [objectives_by_solution[i] for i in ix]
    non_crashed_by_solution = [non_crashed_by_solution[i] for i in ix]

    # Load RDM factors
    rdm_utilities = np.loadtxt(files_root_directory +
                               'rdm_utilities_reeval.csv', delimiter=',')
    rdm_dmp = np.loadtxt(files_root_directory + 'rdm_dmp_reeval.csv',
                         delimiter=',')
    rdm_sources_meaningful = [0] + range(15, 51)
    rdm_water_sources = np.loadtxt(files_root_directory +
                                   'rdm_water_sources_reeval.csv',
                                   delimiter=',',
                                   usecols=rdm_sources_meaningful)
    rdm_factors = np.hstack((rdm_utilities, rdm_dmp, rdm_water_sources))
    rdm_inflows = np.loadtxt(files_root_directory + 'rdm_inflows.csv')

    # Back-calculate objectives for each solution as if objectives had been
    # calculated with 1,000 * 2,000 = 2e6 fully specified worlds.
    objectives_rdm = back_calculate_objectives(objectives_by_solution,
                                               n_objectives_per_utility_plus_jla,
                                               n_utilities)

    # Load objectives on either DU or WCU space, as they came out of Borg
    objective_on_wcu = load_on_du_objectives(files_root_directory,
                                             on='wcu', ix=ix)
    objective_on_du = load_on_du_objectives(files_root_directory,
                                            on='du', ix=ix)

    axis_labels = ['Reliability', 'Restriction\nFrequency',
                   'Infrastructure Net\nPresent Value', 'Financial Cost',
                   'Financial\nRisk', 'Jordan Lake\nAllocation'] * n_utilities
    dataset_names = ('WCU Optimization', 'DU Optimization')

    # Three different brushing criteria relaxing reliability
    brush_criteria1 = {0: [0.99, 1.0], 1: [0.0, 0.2], 4: [0.0, 0.10]}
    brush_criteria2 = {0: [0.98, 1.0], 1: [0.0, 0.2], 4: [0.0, 0.10]}
    brush_criteria3 = {0: [0.97, 1.0], 1: [0.0, 0.2], 4: [0.0, 0.10]}

    # Group solutions
    objective_rdm_grouped = group_objectives(objectives_rdm,
                                             ['max', 'min', 'min', 'min', 'min',
                                              'min'])
    objective_on_wcu_grouped = group_objectives(objective_on_wcu,
                                                ['max', 'min', 'min', 'min',
                                                 'min', 'min'])



    # Get rid of two really low reliability solutions that are shifting
    #       the blue and making all lines look dark except the corresponding few.
    # objective_on_wcu_grouped = objective_on_wcu_grouped[
    #     objective_on_wcu_grouped[:, 0] > 0.92]

    objective_on_du_grouped = group_objectives(objective_on_du,
                                               ['max', 'min', 'min', 'min',
                                                'min', 'min'])

    # # Plot all parallel axis plots
    # plot_all_paxis(objective_on_du_grouped, objective_on_wcu_grouped,
    #                objective_rdm_grouped, axis_labels,
    #                dataset_names, files_root_directory)

    # Retrieve solutions that met brushing criteria
    alphas = __calculate_alphas(objective_rdm_grouped, brush_criteria1,
                                base_alpha=1.)
    good_sols = np.argwhere(alphas - .01)

    performance_criteria = (0.990, 0.2, 0.1)
    utilities = ['OWASA', 'Durham', 'Cary', 'Raleigh']
    robustnesses = calculate_pseudo_robustnesses(performance_criteria,
                                                 objectives_by_solution,
                                                 non_crashed_by_solution,
                                                 rdm_factors,
                                                 files_root_directory, utilities)

    robust_for_all, robustnesses_ordered_by_sol_id = \
        get_robust_compromise_solutions(robustnesses, 0.75, beta=True)

    # PLOT APPROX. ROBUSTNESS BAR CHART
    # pseudo_robustness_plot(utilities, robustnesses,
    #                        [oranges_hc(0.4), blues_hc(0.4)],
    #                        files_root_directory, beta=True)
    # pseudo_robustness_plot(utilities, robustnesses,
    #                        [oranges_hc(0.4), blues_hc(0.4)],
    #                        files_root_directory, plot_du=False)
    # pseudo_robustness_plot(utilities, robustnesses,
    #                        [oranges_hc(0.4), blues_hc(0.4)],
    #                        files_root_directory,
    #                        highlight_sols=robust_for_all, beta=True)

    # most_influential_factors_all, pass_fail_all, non_crashed_rdm_all, \
    # lr_coef_all = get_influential_rdm_factors_logistic_regression(objectives_by_solution,
    #                                                               non_crashed_by_solution,
    #                                                               performance_criteria,
    #                                                               files_root_directory,
    #                                                               (0, 1, 4), rdm_factors,
    #                                                               solutions=robust_for_all,
    #                                                               plot=True)
    # most_influential_factors_all, pass_fail_all, non_crashed_rdm_all, \
    # lr_coef_all = get_influential_rdm_factors_boosted_trees(objectives_by_solution,
    #                                                         non_crashed_by_solution,
    #                                                         performance_criteria,
    #                                                         files_root_directory,
    #                                                         (0, 1, 4), rdm_factors,
    #                                                         solutions=robust_for_all,
    #                                                         plot=True, n_trees=25, tree_depth=2)

    # # Only low and high WJLWTP had permitting times.
    # important_factors_multiple_solutions_plot(most_influential_factors_all,
    #                                           lr_coef_all, 2,
    #                                           create_labels_list(),
    #                                           files_root_directory)

    s = robust_for_all[0]
    utility_to_plot = 1
    realization_to_plot = 13
    ninfra_mono = 5
    ninfra = np.array([0, 6, 0, 4])[utility_to_plot]
    name = np.array(utilities)[utility_to_plot]

    # most_influential_factors = most_influential_factors_all[0]
    # rdm_max = np.argmax(rdm_factors[:, most_influential_factors[-1]])
    # rdm_min = np.argmin(rdm_factors[:, most_influential_factors[-1]])
    # rdm_mono = rdm_max

    # pathways_all = load_pathways_solution(files_root_directory + '../re-evaluation_against_du/pathways/',
    #                                       s, [rdm_max, rdm_min])
    # pathways_all_mono = load_pathways_solution(files_root_directory,
    #                                            s, [rdm_max])
    #
    # # # SORT REALIATIONS BY INFLOWS, EVAPORATIONS AND DEMANDS. NOT FRUITFUL
    # # # PROBABLY BECAUSE OF THE COMPLEX SYSTEM DYNAMICS
    #
    # # lt_rof_mono = np.loadtxt(files_root_directory +
    # #                          'Utilities_s{}_RDM{}_r{}.csv'
    # #                          .format(s, rdm_max, realization_to_plot),
    # #                          delimiter=',',
    # #                          skiprows=1)[:, 4 + utility_to_plot * 15]
    # # skiprows=1)[:, [4 + utility_to_plot * 15, 13 + utility_to_plot * 15]]
    # # lt_rof_mono[:, 1] /= np.max(lt_rof_mono[:, 1]*4)
    #
    # # Plot pathways
    # pathways_list_utility_high = \
    #     get_pathways_by_utility_realization(pathways_all[0])
    # utility_pathways_high = pathways_list_utility_high[utility_to_plot]
    #
    # # replace infrastructure id by construction order
    # construction_order = get_infra_order(utility_pathways_high)
    # utility_pathways_high_copy = \
    #     convert_pathways_from_source_id_to_construction_id(utility_pathways_high,
    #                                                        construction_order)
    #
    # plot_colormap_pathways(utility_pathways_high_copy, 2400, s, rdm_max,
    #                        savefig_directory=files_root_directory,# + 'Figures/',
    #                        nrealizations=1000,
    #                        ninfra=ninfra, sources=sources,
    #                        construction_order=construction_order,
    #                        utility_name=name, year0=2015)
    # # plot_2d_pathways(utility_pathways_high_copy, 2400, s, rdm_max, sources,
    # #                  construction_order, ninfra=ninfra,
    # #                  savefig_directory=files_root_directory + 'Figures/',
    # #                  utility_name=name, year0=2015)
    # # plot_2d_pathways(utility_pathways_high_copy, 2400, s, rdm_max, sources,
    # #                  construction_order, ninfra=ninfra,
    # #                  savefig_directory=files_root_directory + 'Figures/',
    # #                  utility_name=name, year0=2015, monocromatic=True)
    #
    # pathways_list_utility_low = \
    #     get_pathways_by_utility_realization(pathways_all[1])
    # utility_pathways_low = pathways_list_utility_low[utility_to_plot]
    #
    # # replace infrastructure id by construction order
    # utility_pathways_low_copy = \
    #     convert_pathways_from_source_id_to_construction_id(utility_pathways_low,
    #                                                        construction_order)
    #
    # plot_colormap_pathways(utility_pathways_low_copy, 2400, s, rdm_min,
    #                        savefig_directory=files_root_directory,# + 'Figures/',
    #                        nrealizations=1000,
    #                        ninfra=ninfra, sources=sources,
    #                        construction_order=construction_order,
    #                        utility_name=name, year0=2015)
    # # plot_2d_pathways(utility_pathways_low_copy, 2400, s, rdm_min, sources,
    # #                  construction_order, ninfra=ninfra,
    # #                  savefig_directory=files_root_directory + 'Figures/',
    # #                  utility_name=name, year0=2015)
    # # plot_2d_pathways(utility_pathways_low_copy, 2400, s, rdm_min, sources,
    # #                  construction_order, ninfra=ninfra,
    # #                  savefig_directory=files_root_directory + 'Figures/',
    # #                  utility_name=name, year0=2015, monocromatic=True)
    #
    # # Plot pathways
    # pathways_list_utility_mono = \
    #     get_pathways_by_utility_realization(pathways_all_mono[0])
    # utility_pathways_mono = pathways_list_utility_mono[utility_to_plot]
    #
    # # replace infrastructure id by construction order
    # construction_order_mono = get_infra_order(utility_pathways_mono)
    # utility_pathways_mono_copy = \
    #     convert_pathways_from_source_id_to_construction_id(utility_pathways_mono,
    #                                                        construction_order_mono)

    # plot_2d_pathways([utility_pathways_mono_copy[13]], 2400, s, rdm_mono,
    #                  sources, construction_order_mono, ninfra=ninfra_mono,
    #                  savefig_directory=files_root_directory + 'Figures/',
    #                  utility_name=name, year0=2015, monocromatic=True,
    #                  lt_rof=lt_rof_mono)

    # plot_pathways_id(utility_pathways_high, s, rdm_max, sources,
    #                  construction_order, savefig_directory=files_root_directory,
    #                  ninfra=ninfra, utility_name=name, year0=2015)
    # plot_pathways_id(durham_pathways_low, s, rdm_min, sources,
    #                  construction_order, savefig_directory=files_root_directory,
    #                  sort_by=sort_pathways_by, ninfra=7, year0=2015)

    # dec_vars_raw = np.loadtxt(files_root_directory
    #                           + 'combined_reference_sets.set',
    #                           delimiter=',')

    # dec_vars_no_rep, ix = check_repeated(dec_vars_raw)

    dec_vars_processed = process_decvars_inverse(dec_vars[robust_for_all],
                                   ['Durham', 'OWASA', 'Raleigh', 'Cary'],
                                   {'Restriction\nTrigger': 0,
                                    'Transfer\nTrigger': 4,
                                    'Insurance\nTrigger': 15,
                                    'ACFC': 11,
                                    'Long term\nROF': 23,
                                    'Jordan Lake\nAllocation': 7})

    # plot_decvars_radar(dec_vars, [[0] * 4, [1] * 4])
    max_mins =  {'Restriction\nTrigger': [0, 1],
                 'Transfer\nTrigger': [0, 1],
                 'Insurance\nTrigger': [0, 1],
                 'ACFC': [0, 0.1],
                 'Long term\nROF': [0, 1],
                 'Jordan Lake\nAllocation': [0, 0.7]}
    # plot_dec_vars_paxis(dec_vars_processed, (2, 3), max_mins)

    # gmm_cluster(dec_vars, files_root_directory)

    calc_pass_fail(objectives_by_solution, criteria)