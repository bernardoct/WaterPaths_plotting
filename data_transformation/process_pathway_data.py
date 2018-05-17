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
    return sorted(pathways_utility,
                  key=lambda x: list(x[1]) + [1e5] * (10 - len(x[1])))


def create_fixed_length_pathways_array(weeks_vector,
                                       infra_option_or_npv, length):
    fixed_length_array = np.ones(length) * -1

    for i in range(1, len(weeks_vector)):
        fixed_length_array[weeks_vector[i-1]:weeks_vector[i]] = \
            infra_option_or_npv[i - 1]

    fixed_length_array[weeks_vector[-1]:-1] = infra_option_or_npv[-1]

    return fixed_length_array


def get_mesh_pathways(pathways_utility, nweeks):
    x, y = np.meshgrid(range(len(pathways_utility)), range(nweeks))
    z = np.array([create_fixed_length_pathways_array(p[1], p[2], nweeks)
                  for p in pathways_utility]).T

    return x, y, z


def get_infra_order(pathways_utility_rdm):

    lengths = []
    for p in pathways_utility_rdm:
        lengths.append(len(p[2]))
    length = max(lengths)

    built_list = []
    for p in pathways_utility_rdm:
        if len(p[2]) == length:
            built_list.append(p[2])

    # return max(set(built_list), key=built_list.count)
    return built_list[-1]

def get_pathways_by_utility_realization(pathways_rdm):

    # Reformat utility data
    pathways_list_utility = []
    for u in range(int(max(pathways_rdm[:, 1])) + 1):
        pathways_list = []
        for r in range(int(max(pathways_rdm[:, 0])) + 1):
            ur = (pathways_rdm[:, [0, 1]] == np.array([r, u])).all(axis=1)
            pathways_list.append(pathways_rdm[ur][:, [0, 2, 3]].T)
        pathways_list_utility.append(pathways_list)

    return pathways_list_utility
