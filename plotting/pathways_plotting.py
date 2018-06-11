import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from data_transformation.process_pathway_data import *
import seaborn as sns
import pandas as pd
from copy import deepcopy
from matplotlib.patches import Ellipse

tick_font_size = 12

def plot_construction_moment(pathways_utility, s, rdm,
                             savefig_directory=''):
    x, y = [], []
    for i in range(len(pathways_utility)):
        x = np.hstack((x, pathways_utility[i][1]))
        y = np.hstack((y, pathways_utility[i][0]))

    x[0] = 0.

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=250)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.xlim([0, 2240])
    plt.xlabel('Weeks')
    plt.ylabel('Realization')

    if savefig_directory == '':
        plt.show()
    else:
        plt.savefig(savefig_directory +
                    'Pathways_s{}_RDM{}_construction_density.svg'.format(s, rdm))


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
        plt.savefig(savefig_directory + 'Pathways_s{}_RDM{}.svg'.format(s, rdm))


def create_cmap(pathways, ninfra):
    cmap = cm.get_cmap('jet')
    if ninfra == 0:
        ninfra = np.max(pathways)

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, ninfra, ninfra + 1)
    normalize = BoundaryNorm(bounds, cmap.N)

    return cmap, normalize, bounds


def plot_colormap_pathways(pathways_utility, nweeks, s, rdm, #cmap,
                           savefig_directory='', nrealizations=1000,
                           sort_by=(), ninfra=0, sources=(),
                           construction_order=(),
                           utility_name='', year0=0):

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y, pathways = get_mesh_pathways(pathways_utility, nweeks,
                                nrealizations=nrealizations)
    if len(sort_by) == 0:
        pathways = np.array(sorted(pathways, key=lambda x : sum(x)))
    else:
        pathways = np.array(pathways)[sort_by]

    cmap, normalize, bounds = create_cmap(pathways, ninfra+1)

    ax.imshow(pathways + 1.1, origin='lower', cmap=cmap, norm=normalize,
              aspect='auto')
    ax.set_xlabel('Year', **{'fontname':'CMU Bright', 'size' : tick_font_size})
    ax.set_ylabel('Realization', **{'fontname':'CMU Bright', 'size' : tick_font_size})

    ax.grid(False)
    ax.set_yticks([0, nrealizations])
    ax.set_yticklabels(['Little and late\nnew infrastructure', 'Significant '
                       'and early\nnew infrastructure'],
                       {'fontname':'CMU Bright', 'size' : tick_font_size})
    xticks_at = np.arange(0, nweeks, 52.1 * 5)
    ax.set_xticks(xticks_at)
    ax.set_xticklabels((xticks_at / 52.1).astype(int) + year0,
                       {'fontname':'CMU Bright', 'size' : tick_font_size})

    if len(construction_order) > 0:
        sources = np.hstack((['Status-quo'], sources))
        construction_order = np.hstack(([0], construction_order + 1))
        ax2 = fig.add_axes([0.75, 0.1, 0.03, 0.8])
        pos = ax.get_position()
        new_pos = [pos.x0, 0.1, 0.7 - pos.x0, 0.9]
        ax.set_position(new_pos)
        cb = ColorbarBase(ax2, cmap=cmap, norm=normalize,
                                       spacing='proportional', ticks=bounds,
                                       boundaries=bounds, format='%1i')

        # cb.ax.set_title('Infrastructure\nOptions', **{'fontname':'CMU Bright',
        #                                            'size' : tick_font_size})
        # ax2.set_ylabel('Very custom cbar [-]', size=12)
        cb.set_ticks(bounds[:-1] + 0.5)
        cb.set_ticklabels(sources[construction_order.astype(int)],
                          {'fontname':'CMU Bright', 'size' : tick_font_size})
        cb.ax.tick_params(labelsize=tick_font_size)

    if savefig_directory == '':
        plt.show()
    else:
        plt.savefig(savefig_directory +
                    'Pathways_s{}_RDM{}_{}_colormap.svg'.format(s, rdm,
                                                                utility_name))


def plot_2d_pathways(pathways_utility, nweeks, s, rdm, sources,
                     construction_order, savefig_directory='', ninfra=0,
                     utility_name='', year0=0, monocromatic=False,
                     plot_unique_pathways=False, lt_rof=()):
    nrealizations = len(pathways_utility)

    if len(lt_rof) == 0:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 5),
                                      gridspec_kw={'height_ratios' : [3, 1],
                                                   'hspace' : 0.3})
        ax2.plot(lt_rof)
        ax2.set_xlabel('Year', **{'fontname':'CMU Bright', 'size' : tick_font_size})
        ax2.set_ylabel('Long-term ROF [-]', **{'fontname':'CMU Bright', 'size' : tick_font_size})
        xticks_at = np.arange(0, nweeks, 52.1 * 5)
        ax2.set_xticks(xticks_at)
        ax2.set_xticklabels((xticks_at / 52.1).astype(int) + year0,
                            **{'fontname':'CMU Bright', 'size' : tick_font_size})
        ax2.set_ylim(0, 0.2)
        # ax2.set_yticklabels(list(ax2.get_yticklabels()),
        #                     **{'fontname':'CMU Bright', 'size' : tick_font_size})

    pathways = np.array([create_fixed_length_pathways_array(p[1], p[2], nweeks)
                        for p in pathways_utility], dtype=float)[:, :-2]
    cmap, normalize, bounds = create_cmap(pathways, ninfra+1)

    horizontal_lines = []
    vertical_lines = []
    horizontal_lines_colors=[]
    npath = 50

    pathways_to_plot = [pathways_utility[i]
                        for i in np.random.permutation(
            range(nrealizations))[:npath]]
    # pathways_to_plot = pathways_utility

    # Create lines for pathways chart
    longest_pathway = 0
    for p in pathways_to_plot:
        weeks = p[1]
        infras = p[2] + 1
        nconstructions = len(weeks)
        if nconstructions > longest_pathway:
            longest_pathway = nconstructions
            print p[0][0]
        # Create horizontal lines from pathways weeks and infra id
        horizontal_lines.append([[0, weeks[0]],
                                 [0, 0]])

        for i in range(nconstructions - 1):
            horizontal_lines.append([[weeks[i], weeks[i+1]],
                                     [infras[i], infras[i]]])
        horizontal_lines.append([[weeks[-1], nweeks],
                                 [infras[-1], infras[-1]]])

        # Create vertical grey lines connecting horizontal lines
        vertical_lines.append([[weeks[0], weeks[0]], [0, infras[0]]])
        for i in range(1, nconstructions):
            vertical_lines.append([[weeks[i], weeks[i]],
                                   [infras[i - 1], infras[i]]])


    # if plot_unique_pathways:
    #     horizontal_lines = [[x[:len(x) / 2], x[len(x) / 2:]] for x in
    #                         set(tuple(x[0] + x[1]) for x in horizontal_lines)]
    #     vertical_lines =   [[x[:len(x) / 2], x[len(x) / 2:]] for x in
    #                         set(tuple(x[0] + x[1]) for x in vertical_lines)]

    # Assign correct line color
    for hl in horizontal_lines:
        horizontal_lines_colors.append('grey' if monocromatic else
                                       cmap(normalize(hl[1][0])))

    count = 0
    nlines = len(horizontal_lines)
    alpha = max(0.002, 1. / min(npath, len(pathways_to_plot)))
    # Plot vertical lines first, so that they're underneath horizontal ones
    for line in vertical_lines:
        # count += 1
        # print '{}/{}'.format(count, nlines)
        ax.plot(line[0], line[1], c='gray', alpha=alpha, lw=1)

    count = 0
    # Plot horizontal lines
    for line, color in zip(horizontal_lines, horizontal_lines_colors):
        # count += 1
        # print '{}/{}'.format(count, nlines)
        ax.plot(line[0], line[1], c=color, alpha=alpha, lw=3)

    labels = [sources[-1]] + list(sources[construction_order])
    ax.set_yticks(range(0, len(labels)))
    ax.set_ylim(-0.1, 0.1 + ninfra)
    ax.set_yticklabels(labels, **{'fontname':'CMU Bright', 'size' : tick_font_size})
    ax.set_xlabel('Year', **{'fontname':'CMU Bright', 'size' : tick_font_size})
    ax.set_ylabel('Infrastructure Option', **{'fontname':'CMU Bright', 'size' : tick_font_size})
    xticks_at = np.arange(0, nweeks, 52.1 * 5)
    ax.set_xticks(xticks_at)
    ax.set_xticklabels((xticks_at / 52.1).astype(int) + year0, {'fontname':'CMU Bright', 'size' : tick_font_size})
    ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    fig.subplots_adjust(left=0.4)

    if savefig_directory == '':
        plt.show()
        pass
    else:
        plt.savefig(savefig_directory +
                    'Pathways_s{}_RDM{}_{}{}{}_2d.svg'
                    .format(s, rdm, utility_name,
                            '_mono' if monocromatic else '',
                            '_unique' if len(lt_rof) > 0 else ''))


def plot_joyplot_pathways(pathways_utility, construction_order, nweeks, s, rdm,
                          savefig_directory=''):

    weeks = []
    infra_built = []
    for p in pathways_utility:
        weeks += list((p[1] + np.random.normal(loc=0.0, scale=5.0, size=len(p[1]))))
        infra_built += list(-p[2])
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
    # g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=.0, bw=.5)
    g.map(sns.rugplot, "x", clip_on=False, alpha=1, lw=2.)
    # g.map(sns.kdeplot, "x", clip_on=False, color="b", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=0.2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "x")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)
    g.fig.set_size_inches(7.5, 7.5)

    # Remove axes details that don't play will with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.show()


def plot_pathways_id(pathways_all_rdms, s, rdm, sources, construction_order,
                     savefig_directory='',
                     sort_by=(), ninfra=0, utility_name='', year0=0):

    pathways_all_rdms_copy = \
        convert_pathways_from_source_id_to_construction_id(pathways_all_rdms,
                                                           construction_order)

    plot_colormap_pathways(pathways_all_rdms_copy, 2400, s, rdm,
                           savefig_directory=savefig_directory,
                           nrealizations=1000, sort_by=sort_by,
                           ninfra=ninfra, sources=sources,
                           construction_order=construction_order,
                           utility_name=utility_name, year0=year0)
    plot_2d_pathways(pathways_all_rdms_copy, 2400, s, rdm, sources,
                     construction_order, ninfra=ninfra,
                     savefig_directory=savefig_directory,
                           utility_name=utility_name, year0=year0)

    # Bad visualizations
    # plot_3d_pathway_plot(durham_pathways, 2400, s, rdm,
    #                      savefig_directory)
    # plot_joyplot_pathways(durham_pathways, construction_order, 2400,
    #                       s, rdm, savefig_directory=savefig_directory)


