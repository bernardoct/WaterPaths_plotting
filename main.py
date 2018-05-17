from matplotlib.colors import LinearSegmentedColormap
from data_transformation.process_rdm_objectives import *
from plotting.pathways_plotting import *
from plotting.parallel_axis import *

bu_cy = LinearSegmentedColormap.from_list('BuCy', [(0, 0, 1), (0, 1, 1)])

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
    title = 'All on WCU space'
    u = 3
    color_column = n_objectives_per_utility_plus_jla * u
    columns_to_plot = np.arange(n_objectives_per_utility_plus_jla * u,
                                n_objectives_per_utility_plus_jla * u + 6)

    paxis_matplotlib_hack((objective_on_du[:249], objective_on_du[249:]),
                    columns_to_plot, color_column,
                    [bu_cy, cm.get_cmap('autumn')],
                    axis_labels,
                    title,
                    dataset_names,
                    ranges=())

    # pathways_all = load_pathways_solution(files_root_directory, 0, 2)
    #
    # plot_pathways_id(pathways_all, 0, 1, sources, files_root_directory)
