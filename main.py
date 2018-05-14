from matplotlib.colors import LinearSegmentedColormap
from data_transformation.process_rdm_objectives import *
from plotting.pathways_plotting import plot_pathways_id

bu_cy = LinearSegmentedColormap.from_list('BuCy', [(0, 0, 1), (0, 1, 1)])

if __name__ == '__main__':
    # files_root_directory = 'F:/Dropbox/Bernardo/Research/WaterPaths_results/' \
    #                        'rdm_results/'
    files_root_directory = '/media/DATA//Dropbox/Bernardo/Research/WaterPaths_results/' \
                           'rdm_results/'
    n_rdm_scenarios = 2000
    n_solutions = 368
    n_objectives = 20
    n_objectives_per_utility_plus_jla = 6
    n_utilities = 4

    objectives_by_solution = load_objectives(files_root_directory,
                                             n_solutions, n_rdm_scenarios,
                                             n_objectives, n_utilities)

    objectives_rdm = back_calculate_objectives(objectives_by_solution,
                                               n_objectives_per_utility_plus_jla,
                                               n_utilities)

    objective_on_du = load_on_du_objectives(files_root_directory, on='wcu')

    ranges = [[0.8, 1], [0, 0.4], [0, 500], [0, 0.3], [0, 0.3], [0, 0.1]] * 4
    axis_labels = ['Reliability', 'Rest. Freq.', 'Infra NPV', 'Financial Cost',
                    'Worse Case Cost', 'JLA'] * n_utilities
    title = 'All on WCU space | cool colors WCU, warm colors DU'
    u = 3
    color_column = n_objectives_per_utility_plus_jla * u - 1
    columns_to_plot = np.arange(n_objectives_per_utility_plus_jla * u - 1,
                                n_objectives_per_utility_plus_jla * u + 5)
    # paxis_plot_hack((objectives_rdm[:249], objectives_rdm[249:]),
    #                 columns_to_plot, color_column,
    #                 [bu_cy, cm.autumn],
    #                 axis_labels,
    #                 axis_labels[color_column],
    #                 title,
    #                 ranges)
    # paxis_plot_hack((objective_on_du[:249], objective_on_du[249:]),
    #                 columns_to_plot, color_column,
    #                 [bu_cy, cm.autumn],
    #                 axis_labels,
    #                 axis_labels[color_column],
    #                 title,
    #                 ranges)

    pathways_all = load_pathways_solution(files_root_directory, 0, 2)

    plot_pathways_id(pathways_all, 0, 1, files_root_directory)
