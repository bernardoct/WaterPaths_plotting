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

def create_fixed_length_array(weeks, length):


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
    x, y = [], []
    for i in range(len(durham_pathways_reordered)):
        x = np.hstack((x, durham_pathways_reordered[i][0]))
        y = np.hstack((y, durham_pathways_reordered[i][1]))

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

    if savefig_directory == '':
        plt.show()
    else:
        plt.savefig(savefig_directory + 'Pathways_s{}_RDM{}.png'.format(s, rdm))