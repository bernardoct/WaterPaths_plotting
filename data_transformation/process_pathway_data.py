from copy import deepcopy

import numpy as np


def reorder_pathways(pathways_utility):

    # score = []
    # for p in pathways_utility:
    #     # score.append(np.abs(np.sum(2400 - p[1])))
    #     score.append(np.abs(np.sum(p[2])))
    #
    # ix = np.argsort(np.array(score))
    #
    # return [pathways_utility[i] for i in ix]

    # return sorted(pathways_utility,
    #               key=lambda x: (len(x[1]),
    #                              -np.sum(x[1]),
    #                              x[2, -1]))
    # return sorted(pathways_utility,
    #               key=lambda x: [len(x[1]),
    #                              np.prod(x[2]),
    #                              -np.sum(x[1])])

    sorted_pathways = sorted(pathways_utility,
                  key=lambda x: list(x[2]) + [1e5] * (10 - len(x[2])))
    sorted_pathways_no_empty = []
    count = 0
    for sp in sorted_pathways:
        if sp != []:
            sorted_pathways_no_empty.append(sp)
            print count, sp[0]

        count += 1
    return sorted_pathways_no_empty


def convert_pathways_from_source_id_to_construction_id(pathways_all_rdms,
                                                       construction_order):
    construction_order_expanded = np.zeros(max(construction_order) + 1)
    for i in range(len(construction_order)):
        construction_order_expanded[construction_order[i]] = i

    pathways_all_rdms_copy = deepcopy(pathways_all_rdms)
    for p in pathways_all_rdms_copy:
        p[2] = construction_order_expanded[p[2]]

    return pathways_all_rdms_copy


def create_fixed_length_pathways_array(weeks_vector,
                                       infra_option_or_npv, length):

    fixed_length_array = np.ones(length) * -1

    for i in range(1, len(weeks_vector)):
        fixed_length_array[weeks_vector[i - 1] : weeks_vector[i]] = \
            infra_option_or_npv[i - 1]

    fixed_length_array[weeks_vector[-1] : -1] = infra_option_or_npv[-1]

    return fixed_length_array


def get_mesh_pathways(pathways_utility, nweeks, nrealizations=-1):
    if nrealizations == -1:
        nrealizations = len(pathways_utility)

    x, y = np.meshgrid(range(nrealizations), range(nweeks))

    z = np.ones((nrealizations, nweeks)) * -1
    for p in pathways_utility:
        r = p[0][0]
        z[r] = create_fixed_length_pathways_array(p[1], p[2], nweeks)

    return x, y, z


def get_infra_order(pathways_utility_rdm):

    build_infra_all_reals = np.array([])
    for p in pathways_utility_rdm:
        build_infra_all_reals = np.hstack((build_infra_all_reals, p[2]))
    built_infra = np.unique(build_infra_all_reals)

    return built_infra.astype(int)


def get_pathways_by_utility_realization(pathways_sol):

    # Reformat utility data
    pathways_list_utility = []
    for u in range(int(max(pathways_sol[:, 1])) + 1):
        pathways_list = []
        for r in range(int(max(pathways_sol[:, 0])) + 1):
            ur = (pathways_sol[:, [0, 1]] == np.array([r, u])).all(axis=1)
            if np.sum(ur) > 0:
                pathways_list.append(pathways_sol[ur][:, [0, 2, 3]].T)
        pathways_list_utility.append(pathways_list)

    return pathways_list_utility
