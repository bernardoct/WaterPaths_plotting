import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from data_transformation.process_pathway_data import *
import seaborn as sns
import pandas as pd


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


def plot_3d_pathways(pathways_utility, nweeks, s, rdm, savefig_directory=''):

    x, y, z = get_mesh_pathways(pathways_utility, nweeks)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False,
                    rstride=10, cstride=100)
    # cset = ax.contourf(x, y, z, zdir='x', offset=-500, cmap=cm.coolwarm)
    # cset = ax.contourf(x, y, z, zdir='y', offset=3000, cmap=cm.coolwarm)
    # cset = ax.contourf(x, y, z, zdir='z', offset=-40, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if savefig_directory == '':
        plt.show()
    else:
        plt.savefig(savefig_directory + 'Pathways_s{}_RDM{}.png'.format(s, rdm))


def plot_colormap_pathways(pathways_utility, nweeks, s, rdm, savefig_directory=''):

    x, y, z = get_mesh_pathways(pathways_utility, nweeks)

    plt.imshow(z.T, origin='lower', cmap=cm.jet)

    if savefig_directory == '':
        plt.show()
    else:
        plt.savefig(savefig_directory + 'Pathways_s{}_RDM{}.png'.format(s, rdm))


def plot_2d_pathways(pathways_utility, nweeks, s, rdm, savefig_directory=''):

    pathways = np.array([create_fixed_length_pathways_array(p[1], p[2], nweeks)
                        for p in pathways_utility])[:, :-2]

    for p in pathways:
        plt.plot(p, c='r', alpha=0.01, lw=4)

    if savefig_directory == '':
        plt.show()
    else:
        plt.savefig(savefig_directory + 'Pathways_s{}_RDM{}.png'.format(s, rdm))


def plot_joyplot_pathways(pathways_utility, construction_order, nweeks, s, rdm,
                          savefig_directory=''):

    weeks = []
    infra_built = []
    for p in pathways_utility:
        weeks += list(p[1])
        infra_built += list(p[2])
    x = weeks
    g = infra_built

    # Create the data
    rs = np.random.RandomState(1979)
    # x = rs.randn(500)
    # g = np.tile(list("ABCDEFGHIJ"), 50)
    df = pd.DataFrame(dict(x=x, g=g))
    # m = df.g.map(ord)
    # df["x"] += m

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "x")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play will with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)


def plot_pathways_id(pathways_all_rdms, s, rdm, savefig_directory=''):
    pathways_rdm = pathways_all_rdms[rdm]
    pathways_list_utility = get_pathways_by_utility_realization(pathways_rdm)

    # re-ordet pathways to smooth out plots
    durham_pathways_reordered = reorder_pathways(pathways_list_utility[1])

    # replace infrastructure id by construction order
    construction_order = get_infra_order(durham_pathways_reordered)
    construction_order_expanded = np.zeros(max(construction_order) + 1)
    for i in range(len(construction_order)):
        construction_order_expanded[construction_order[i]] = i

    for p in durham_pathways_reordered:
        p[2] = construction_order_expanded[p[2]]

    # plot_construction_moment(durham_pathways_reordered)
    # plot_3d_pathway_plot(durham_pathways_reordered, 2400, s, rdm,
    #                      savefig_directory)
    # pathway_colormap_plot(durham_pathways_reordered, 2400, s,
    #                       rdm, savefig_directory=savefig_directory)
    # plot_2d_pathways(durham_pathways_reordered, 2400, s,
    #                  rdm, savefig_directory=savefig_directory)
    plot_joyplot_pathways(durham_pathways_reordered, construction_order, 2400, s,
                          rdm, savefig_directory=savefig_directory)


