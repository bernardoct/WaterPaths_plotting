import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


def reorder_pathways(pathways_utility):
    score = []

    for p in pathways_utility:
        score.append(np.abs(np.sum(1e6 - p[1])))

    ix = np.argsort(np.array(score))

    return [pathways_utility[i] for i in ix]


def create_fixed_length_array(weeks_vector, infra_option_or_npv, length):
    fixed_length_array = np.zeros(length)

    for i in range(1, len(weeks_vector)):
        fixed_length_array[weeks_vector[i]:weeks_vector[i-1]] = infra_option_or_npv[i - 1]

    fixed_length_array[weeks_vector[-1]:-1] = infra_option_or_npv[-1]

    return fixed_length_array


def plot_construction_moment(pathways_utility):
    x, y = [], []
    for i in range(len(pathways_utility)):
        x = np.hstack((x, pathways_utility[i][0]))
        y = np.hstack((y, pathways_utility[i][1]))

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def plot_3d_pathway_plot(pathways_utility, nweeks):

    x, y = np.meshgrid(range(len(pathways_utility)), range(nweeks))




def plot_pathways_id(pathways_all_rdms, s, rdm, savefig_directory=''):
    pathways_rdm = pathways_all_rdms[rdm]

    # Reformat utility data
    pathways_list_utility = []
    for u in range(int(max(pathways_rdm[:, 1])) + 1):
        pathways_list = []
        for r in range(int(max(pathways_rdm[:, 0])) + 1):
            ur = (pathways_rdm[:, [0, 1]] == np.array([r, u])).all(axis=1)
            pathways_list.append(pathways_rdm[ur][:, [0, 2, 3]].T)
        pathways_list_utility.append(pathways_list)

    durham_pathways_reordered = reorder_pathways(pathways_list_utility[1])

    # plot_construction_moment(durham_pathways_reordered)

    if savefig_directory == '':
        plt.show()
    else:
        plt.savefig(savefig_directory + 'Pathways_s{}_RDM{}.png'.format(s, rdm))