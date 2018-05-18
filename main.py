from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from data_transformation.process_rdm_objectives import *
from plotting.pathways_plotting import *
from plotting.parallel_axis import *

bu_cy = LinearSegmentedColormap.from_list('BuCy', [(0, 0, 1), (0, 1, 1)])
bu_cy_r = bu_cy.reversed()

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

    objectives_by_solution = load_objectives(files_root_directory,
                                             n_solutions, n_rdm_scenarios,
                                             n_objectives, n_utilities)

    objectives_rdm = back_calculate_objectives(objectives_by_solution,
                                               n_objectives_per_utility_plus_jla,
                                               n_utilities)

    objective_on_du = load_on_du_objectives(files_root_directory, on='wcu')

    ranges = [[0.8, 1], [0, 0.4], [0, 800], [0, 0.5], [0, 0.3], [0, 1.0]] * 4
    axis_labels = ['Reliability', 'Rest. Freq.', 'Infra NPV', 'Financial Cost',
                   'Worse\nCase Cost', 'Jordan Lake\nAllocation'] * n_utilities
    dataset_names = ('WCU Optimization', 'DU Optimization')
    u = 1
    color_column = n_objectives_per_utility_plus_jla * u
    columns_to_plot = np.arange(n_objectives_per_utility_plus_jla * u,
                                n_objectives_per_utility_plus_jla * u + 6)

    ranges_du = np.vstack((objective_on_du.min(axis=0),
                           objective_on_du.max(axis=0))).T
    ranges_du[0] = np.array([ranges_du[0, 1], ranges_du[0, 0]])
    brush_criteria = {6: [0.99, 1.0], 7: [0.0, 0.2], 10: [0.0, 0.05]}
    paxis_matplotlib_hack([objective_on_du[249:]],
                          columns_to_plot, color_column,
                          [cm.get_cmap('autumn_r')],
                          axis_labels,
                          'DU on Re-eval Space',
                          dataset_names[1],
                          ranges=ranges_du, file_name='all_reeval_du.png',
                          invert_axes=range(0, 20, 6),
                          brush_criteria=brush_criteria)
    paxis_matplotlib_hack([objective_on_du[:249]],
                          columns_to_plot, color_column,
                          [bu_cy_r],
                          axis_labels,
                          'WCU on Re-eval Space',
                          dataset_names[0],
                          ranges=ranges_du, file_name='all_reeval_wcu.png',
                          invert_axes=range(0, 20, 6),
                          brush_criteria=brush_criteria)
    paxis_matplotlib_hack((objective_on_du[:249], objective_on_du[249:]),
                          columns_to_plot, color_column,
                          [bu_cy_r, cm.get_cmap('autumn_r')],
                          axis_labels,
                          'DU and WCU on Re-eval Space',
                          dataset_names,
                          ranges=ranges_du, file_name='all_reeval_du_wcu.png',
                          invert_axes=range(0, 20, 6),
                          brush_criteria=brush_criteria)

    # pathways_all = load_pathways_solution(files_root_directory, 0, 2)
    #
    # plot_pathways_id(pathways_all, 0, 1, sources, files_root_directory)
