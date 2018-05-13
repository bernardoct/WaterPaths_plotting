import numpy as np
from multiprocessing import Pool
from functools import partial


def read_rdm_file(files_root_directory, i):
    file = files_root_directory + "Objectives_by_rdm/" \
                                  "Objectives_RDM{}_sols0_to_368.csv".format(i)
    print 'Reading objectives rdm {}'.format(i)
    return np.loadtxt(file, delimiter=',')


def to_objectives_by_solution(objectives_rdm, nsols, nobjs,
                              nutils, files_root_directory):
    objectives_by_solution = []
    nobjs_per_utility = nobjs / nutils
    nrdm = len(objectives_rdm)
    jla = np.loadtxt(files_root_directory + 'combined_reference_sets.set',
                     delimiter=',')
    for s in range(nsols):
        jla_sol = jla[s, 7:11]
        objectives_sol = np.ones((nrdm, nobjs)) * 10000
        print 'Compiling solution {}'.format(s)

        for o in range(nrdm):
            if s < len(objectives_rdm[o]):
                objectives_sol[o] = objectives_rdm[o][s]

        objectives_sol_jla = np.zeros((objectives_sol.shape[0],
                                       objectives_sol.shape[1] + nutils))
        for u in range(nutils):
            objectives_sol_jla[:, nobjs_per_utility * u + u:
                                  nobjs_per_utility * (u + 1) + u] = \
                objectives_sol[:, nobjs_per_utility * u:
                                  nobjs_per_utility * (u + 1)]
            objectives_sol_jla[:, (nobjs_per_utility + 1) * (u + 1) - 1] \
                = jla_sol[u]

        non_crashed_rdm = objectives_sol_jla[:, 0] < 1.1

        objectives_sol_jla = objectives_sol_jla[non_crashed_rdm]

        np.save(files_root_directory + 'Objectives_by_solution/'
                                       'Objectives_s{}.npy'.format(s),
                objectives_sol_jla)
        np.savetxt(files_root_directory + 'Objectives_by_solution/'
                                          'Objectives_s{}.csv'.format(s),
                   objectives_sol_jla, delimiter=',')

        objectives_by_solution.append(objectives_sol_jla)

    return objectives_by_solution


def load_objs_by_solution(files_root_directory, nsols):
    objectives_by_solution = []
    for i in range(nsols):
        objs = np.load(files_root_directory + 'Objectives_by_solution/'
                                              'Objectives_s{}.npy'.format(i))
        objectives_by_solution.append(objs)

    return objectives_by_solution


def load_objectives(files_root_directory, nsols, n_rdm_scenarios,
                    nobjs, nutils, processed=True):
    if not processed:
        # Read objectives from RDM files.
        objectives_rdm = Pool(4).map(partial(read_rdm_file,
                                             files_root_directory),
                                     range(n_rdm_scenarios))

        # Organize objectives in one matrix per solution and save them.
        return to_objectives_by_solution(objectives_rdm, nsols, nobjs,
                                         nutils, files_root_directory)
    else:
        return load_objs_by_solution(files_root_directory, nsols)


def load_on_du_objectives(files_root_directory, on):
    if on == 'du':
        objectives = np.loadtxt(
            files_root_directory + 'Objectives_in_wcu_du/objs_all_on_du.csv',
            delimiter=',')
    elif on == 'wcu':
        objectives = np.loadtxt(
            files_root_directory + 'Objectives_in_wcu_du/objs_all_on_wcu.csv',
            delimiter=',')
    else:
        raise ValueError('When reading objective files you have to specify '
                         'either \'wcu\' or \'du\'.')

    jla = np.loadtxt(files_root_directory + 'combined_reference_sets.set',
                     delimiter=',')[:, 7:11]
    overallocation = np.sum(jla, axis=1) > 1.
    jla[overallocation, :] /= \
        np.tile(np.sum(jla, axis=1)[overallocation], (4, 1)).T
    objectives = np.hstack((objectives, jla))

    indexing = np.array(range(20) + [23])
    indexing = np.insert(indexing, 5, 20)
    indexing = np.insert(indexing, 11, 21)
    indexing = np.insert(indexing, 17, 22)

    return objectives[:, indexing]


def back_calculate_objectives(objectives_by_solution, nobjs, nutils):
    nsols = len(objectives_by_solution)
    objectives = np.zeros((nsols, nobjs * nutils))
    for s in range(nsols):
        objectives[s, :] = np.mean(objectives_by_solution[s], axis=0)
        nrdms = len(objectives_by_solution[s])
        for u in range(nutils):
            worse_case_cost = np.sort(objectives_by_solution[s]
                                      [:, (u + 1) * nobjs - 1])[-nrdms / 100]
            objectives[s, (u + 1) * nobjs - 1] = worse_case_cost

    return objectives


def load_pathways_solution(files_root_directory, s, n_rdms):
    pathways = []
    for rdm in range(n_rdms):
        pathways.append(
            np.loadtxt(
                files_root_directory
                + 'Pathways/Pathways_s{}_RDM{}.out'.format(s, rdm),
                delimiter='\t',
                comments='R',
                dtype=int)
        )

    return pathways
