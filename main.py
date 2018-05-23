from matplotlib.colors import LinearSegmentedColormap

from data_analysis.logistic_regression import logistic_regression
from data_analysis.sorting_pseudo_robustness import \
    calculate_pseudo_robustness_uniform, calculate_pseudo_robustness_beta
from data_transformation.process_rdm_objectives import *
from plotting.pathways_plotting import *
from plotting.parallel_axis import *

bu_cy = LinearSegmentedColormap.from_list('BuCy', [(0, 0, 1), (0, 1, 1)])
bu_cy_r = bu_cy.reversed()

def group_objectives(objectives, max_min):
    nsols = len(objectives)
    nobjs_total = len(objectives[0])
    nobjs = len(max_min)
    grouped_objs = np.zeros((nsols, len(max_min)))
    for i in range(nobjs):
        objs = range(i, nobjs_total, nobjs)
        if max_min[i] == 'min':
            grouped_objs[:, i] = np.max(objectives[:, objs], axis=1)
        elif max_min[i] == 'max':
            grouped_objs[:, i] = np.min(objectives[:, objs], axis=1)
        else:
            raise Exception('You must specify either max or min')

    return grouped_objs

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
    sources = np.array(["Lake Michie & Little River Res. (Durham)",
                        "Falls Lake",
                        "Wheeler-Benson Lakes",
                        "Stone Quarry",
                        "Cane Creek Reservoir",
                        "University Lake",
                        "Jordan Lake",
                        "Little River Reservoir (Raleigh)",
                        "Richland Creek Quarry",
                        "Teer Quarry",
                        "Neuse River Intake",
                        "Dummy Node",
                        "Low Stone Quarry Expansion",
                        "High Stone Quarry Expansion",
                        "University Lake Expansion",
                        "Low Lake Michie Expansion",
                        "High Lake Michie Expansion",
                        "Falls Lake Reallocation",
                        "Low Reclaimed Water System",
                        "High Reclaimed Water System",
                        "Low WJLWTP",
                        "High WJLWTP",
                        "Cary WTP upgrade 1",
                        "Cary WTP upgrade 2",
                        "Cane Creek Reservoir Expansion",
                        "Status-quo"])

    # Load objectives for each RDM scenario organized by solution (list
    # of matrixes)
    objectives_by_solution = load_objectives(files_root_directory,
                                             n_solutions, n_rdm_scenarios,
                                             n_objectives, n_utilities)

    # Back-calculate objectives for each solution as if objectives had been
    # calculated with 1,000 * 2,000 = 2e6 fully specified worlds.
    objectives_rdm = back_calculate_objectives(objectives_by_solution,
                                               n_objectives_per_utility_plus_jla,
                                               n_utilities)

    # Load objectives on either DU or WCU space, as they came out of Borg
    objective_on_wcu = load_on_du_objectives(files_root_directory, on='wcu')
    objective_on_du = load_on_du_objectives(files_root_directory, on='du')

    ranges = [[0.8, 1], [0, 0.4], [0, 800], [0, 0.5], [0, 0.3], [0, 1.0]] * 4
    axis_labels = ['Reliability', 'Rest. Freq.', 'Infra NPV', 'Financial Cost',
                   'Worse\nCase Cost', 'Jordan Lake\nAllocation'] * n_utilities
    dataset_names = ('WCU Optimization', 'DU Optimization')
    # u = 1
    # color_column = n_objectives_per_utility_plus_jla * u
    # columns_to_plot = np.arange(n_objectives_per_utility_plus_jla * u,
    #                             n_objectives_per_utility_plus_jla * u + 6)
    columns_to_plot = range(6)
    color_column = 1

    brush_criteria = {0: [0.99, 1.0], 1: [0.0, 0.2], 4: [0.0, 0.10]}
    objective_rdm_grouped = group_objectives(objectives_rdm,
                                             ['max', 'min', 'min', 'min', 'min',
                                              'min'])
    objective_on_wcu_grouped = group_objectives(objective_on_wcu,
                                             ['max', 'min', 'min', 'min', 'min',
                                              'min'])
    objective_on_du_grouped = group_objectives(objective_on_du,
                                             ['max', 'min', 'min', 'min', 'min',
                                              'min'])
    ranges_du = np.vstack((objective_rdm_grouped.min(axis=0),
                           objective_rdm_grouped.max(axis=0))).T
    ranges_du[0] = np.array([ranges_du[0, 1], ranges_du[0, 0]])
    # cmap_wcu = bu_cy_r
    # cmap_du = cm.get_cmap('autumn_r')
    cmap_wcu = bu_cy
    cmap_du = cm.get_cmap('autumn')
    paxis_plot([objective_rdm_grouped[249:]],
               columns_to_plot, color_column,
               [cmap_du],
               axis_labels,
               'DU on Re-eval Space',
               dataset_names[1],
               axis_ranges=ranges_du,
               file_name=files_root_directory + 'du_on_reeval.png',
               axis_to_invert=[0]#,#range(0, 20, 6),
               # brush_criteria=brush_criteria
               )
    paxis_plot([objective_rdm_grouped[:249]],
               columns_to_plot, color_column,
               [cmap_wcu],
               axis_labels,
               'WCU on Re-eval Space',
               dataset_names[0],
               axis_ranges=ranges_du,
               file_name=files_root_directory + 'wcu_on_reeval.png',
               axis_to_invert=[0]#,#range(0, 20, 6),
               # brush_criteria=brush_criteria
               )
    paxis_plot([objective_on_du_grouped[249:]],
               columns_to_plot, color_column,
               [cmap_du],
               axis_labels,
               'DU on Re-eval Space',
               dataset_names[1],
               axis_ranges=ranges_du,
               file_name=files_root_directory + 'du_on_du.png',
               axis_to_invert=[0]#,#range(0, 20, 6),
               # brush_criteria=brush_criteria
               )
    paxis_plot([objective_on_wcu_grouped[:249]],
               columns_to_plot, color_column,
               [cmap_wcu],
               axis_labels,
               'WCU on Re-eval Space',
               dataset_names[0],
               axis_ranges=ranges_du,
               file_name=files_root_directory + 'wcu_on_wcu.png',
               axis_to_invert=[0]#,#range(0, 20, 6),
               # brush_criteria=brush_criteria
               )
    paxis_plot((objective_rdm_grouped[:249], objective_rdm_grouped[249:]),
               columns_to_plot, color_column,
               [cmap_wcu, cmap_du],
               axis_labels,
               'DU and WCU on Re-eval Space',
               dataset_names,
               axis_ranges=ranges_du,
               file_name=files_root_directory + 'all_reeval_wcu_du.png',
               axis_to_invert=[0]#,#range(0, 20, 6),
               # brush_criteria=brush_criteria
               )
    paxis_plot((objective_rdm_grouped[:249], objective_rdm_grouped[249:]),
               columns_to_plot, color_column,
               [cmap_wcu, cmap_du],
               axis_labels,
               'DU and WCU on Re-eval Space',
               dataset_names,
               axis_ranges=ranges_du,
               file_name=files_root_directory + 'all_reeval_wcu_du_brushed.png',
               axis_to_invert=[0],#range(0, 20, 6),
               brush_criteria=brush_criteria
               )



    sol_number = 273
    # pathways_all = load_pathways_solution('F:/Dropbox/Bernardo/Research/WaterPaths_results/re-evaluation_against_du/', sol_number, 2)
    #
    # plot_pathways_id(pathways_all, sol_number, 1, sources, files_root_directory)

    # Load RDM files in a single table
    rdm_utilities = np.loadtxt(files_root_directory
                               + 'rdm_utilities_reeval.csv',
                               delimiter=',')
    rdm_dmp = np.loadtxt(files_root_directory + 'rdm_dmp_reeval.csv',
                         delimiter=',')
    rdm_sources_meaningful = [0] + range(15, 51)
    rdm_water_sources = np.loadtxt(files_root_directory
                                   + 'rdm_water_sources_reeval.csv',
                                   delimiter=',')[:, rdm_sources_meaningful]


    rdm_shape = (len(rdm_dmp), rdm_utilities.shape[1] + rdm_dmp.shape[1]
                 + rdm_water_sources.shape[1])
    rdm_factors = np.hstack((rdm_utilities, rdm_dmp, rdm_water_sources))

    lows = np.array(
        [0.5, 1.0, 0.6, 0.6] + [0.9] * 4 + [0.8] + [0.75, 1.0] * 18)
    highs = np.array(
        [2., 1.2, 1.0, 1.4] + [1.1] * 4 + [1.2] + [1.5, 1.2] * 18)
    means = np.array(
        [1., 1.0, 1.0, 1.0] + [1.0] * 4 + [1.0] + [1.0, 1.0] * 18)

    performance_criteria = (0.990, 0.2, 1e5, 1e5, 0.1, 1e5)

    most_influential_factors, pass_fail, non_crashed_rdm, lr_coef = \
                logistic_regression(
                group_objectives(objectives_by_solution[sol_number],
                                 ['max', 'min', 'min', 'min', 'min', 'min']),
                rdm_factors, sol_number, performance_criteria, plot=True
            )
    #
    # pseudo_robustness_uniform = []
    # pseudo_robustness_beta = []
    # for sol_number in range(368):
    #     print 'Calculating robustness for solution {}'.format(sol_number)
    #
    #     most_influential_factors, pass_fail, non_crashed_rdm, lr_coef = \
    #         logistic_regression(
    #             group_objectives(objectives_by_solution[sol_number],
    #                              ['max', 'min', 'min', 'min', 'min', 'min']),
    #             rdm_factors, sol_number, performance_criteria#, plot=True
    #         )
    #
    #     # Number of important uncertainty factors -- scenario discovery
    #     n_factors = np.sum(np.sort(np.abs(lr_coef)) / np.sum(np.abs(lr_coef))
    #                        > 0.1)
    #     print n_factors
    #
    #     if sum(pass_fail) != 0:
    #         r1 = calculate_pseudo_robustness_uniform(pass_fail)
    #         r2 = calculate_pseudo_robustness_beta(
    #             pass_fail,
    #             non_crashed_rdm[:, most_influential_factors[-n_factors:]],
    #             2.5,
    #             lows[most_influential_factors[-n_factors:]],
    #             highs[most_influential_factors[-n_factors:]],
    #             means[most_influential_factors[-n_factors:]]#, plot=True
    #         )
    #     else:
    #         r1, r2 = 0, 0
    #
    #     pseudo_robustness_uniform.append(r1)
    #     pseudo_robustness_beta.append(r2)
    #
    # sorted_uniform = np.argsort(pseudo_robustness_uniform)[::-1]
    # sorted_beta = np.argsort(pseudo_robustness_beta)[::-1]
    # robustness = np.vstack((
    #     sorted_uniform,
    #     np.array(pseudo_robustness_uniform)[sorted_uniform],
    #     sorted_beta,
    #     np.array(pseudo_robustness_beta)[sorted_beta])
    # )
    #
    # np.savetxt(files_root_directory + 'robustness.csv', robustness.T,
    #            header='uniform rank, uniform, beta rank, beta',
    #            delimiter=',')
