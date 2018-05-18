import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import colorbar
from copy import deepcopy


def __basic_plot_formatting(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylim(-0.02, 1.02)


def __calculate_data_min_max(joint_dataset, columns, axis_ranges):
    if len(axis_ranges) == 0:
        data_max = np.max(joint_dataset, axis=0)
        data_min = np.min(joint_dataset, axis=0)
    elif len(axis_ranges) != len(columns) \
            and len(axis_ranges) != len(joint_dataset[0]):
        raise ValueError('Number of axis_ranges != number of columns.')
    else:
        data_max = np.array(axis_ranges).T[1]
        data_min = np.array(axis_ranges).T[0]

    return data_max, data_min


def __invert_axis(datasets_mod, n_columns_datasets, invert_axis, axis_ranges,
                  brush_criteria):
    # Invert axis, if desired
    invert = np.ones(n_columns_datasets, dtype=float)
    for axes in invert_axis:
        invert[axes] = -1.
        axis_ranges[axes] *= -1
        axis_ranges[axes] = axis_ranges[axes][::-1]
        if axes in brush_criteria:
            brush_criteria[axes] = [-a for a in brush_criteria[axes][::-1]]

        for dataset in datasets_mod:
            dataset[:, axes] *= -1

    return invert


def __calculate_alphas(dataset, brush_criteria=dict(), base_alpha=0.5):
    n_points = len(dataset)
    alphas = np.ones(n_points, dtype=float) * base_alpha

    not_brushed = np.array([True] * n_points)
    for b in brush_criteria:
        not_brushed = \
            np.multiply(not_brushed,
                        np.multiply(dataset[:, b] < max(brush_criteria[b]),
                                    dataset[:, b] > min(brush_criteria[b])))
    alphas[[not c for c in not_brushed]] = 0.03

    return alphas


def __plot_datasets(datasets_mod, ax, data_min, data_max, color_maps, columns,
                    color_column, x_axis, brush_criteria=dict(),
                    base_alpha=0.5):

    # Plot data sets
    for dataset, cmap in zip(datasets_mod, color_maps):
        alphas = __calculate_alphas(dataset, brush_criteria=brush_criteria,
                                    base_alpha=base_alpha)
        dataset_normed = (dataset - data_min) \
                               / (data_max - data_min + 1e-8)
        dataset_color_values = (dataset - dataset.min(0)) \
                                / (dataset.ptp(0) + 1e-8)

        # Plot data
        for d, dc, a in zip(dataset_normed, dataset_color_values, alphas):
            ax.plot(x_axis, d[columns], c=cmap(dc[color_column]), alpha=a)


def __add_color_bar(datasets, ax, dataset_names, color_maps, color_column,
                    invert, labels, fontname_body):
    n_datasets = len(datasets)
    axis = []
    for cmap, name, i in zip(color_maps, dataset_names, range(n_datasets)):
        # Set color bar auxiliary variables
        orientation = 'horizontal'
        vmin = datasets[i][:, color_column].min()
        vmax = datasets[i][:, color_column].max()
        normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Reverse colormap is color_column is marked to be reversed.
        cmap_mod = cmap if invert[color_column] == 1 else cmap.reversed()

        # Create color bars
        cax, _ = colorbar.make_axes(ax, orientation=orientation, aspect=60)
        cbar = colorbar.ColorbarBase(cax, cmap=cmap_mod, norm=normalize,
                                     orientation=orientation)
        axis.append(cax)

        # Set color bar font
        cbar.set_label('{} {}'.format(name, labels[color_column]),
                       **{'family': fontname_body})

        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['{0:.3g}'.format(vmin), '{0:.3g}'.format(vmax)])

        for l in cax.xaxis.get_ticklabels():
            l.set_family(fontname_body)

    # Set color bars positions
    width_cbar = (0.8 - 0.05 * (n_datasets - 1)) / n_datasets
    ax.set_position([0.05, 0.17, 0.9, 0.65])
    for i in range(n_datasets):
        axis[i].set_position([0.1 + (width_cbar + 0.05) * i,
                              0., width_cbar, 0.1])


def __set_numbers_labels_axis(ax, data_min, data_max, columns, invert, labels,
                              x_axis, plot_font):
    for x in x_axis:
        ax.text(x, -0.07, '{0:.2g}'.format(invert[columns[x]] * data_min[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1.03, '{0:.2g}'.format(invert[columns[x]] * data_max[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1.09, np.array(labels)[columns][x],
                horizontalalignment='center', **plot_font)
        ax.plot((x, x), (-0.02, 1.02),
                c='black', alpha=0.3, lw=0.2)


def paxis_plot(datasets, columns, color_column, color_maps,
               axis_labels, title, dataset_names, axis_ranges=(),
               fontname_title='Gill Sans MT',
               fontname_body='CMU Bright', file_name='',
               size=(9, 6), axis_to_invert=(), brush_criteria=dict()):

    datasets_mod = deepcopy(datasets)
    axis_ranges = np.array(axis_ranges)
    brush_criteria = deepcopy(brush_criteria)
    n_columns_datasets = len(datasets_mod[0][0])
    n_axis = len(columns)
    x_axis = range(n_axis)

    # Invert axis, if needed
    invert = __invert_axis(datasets_mod, n_columns_datasets, axis_to_invert,
                           axis_ranges, brush_criteria=brush_criteria)

    # Create combined data set
    joint_dataset = np.vstack(datasets_mod)

    plot_font = {'fontname': fontname_body}
    plot_font_title = {'fontname': fontname_title, 'fontsize': 16}

    # Set title
    fig, ax = plt.subplots()
    fig.set_size_inches(size, forward=True)
    fig.suptitle(title, **plot_font_title)

    # Remove axis and ticks
    __basic_plot_formatting(ax)

    # Calculate axis_ranges or use axis_ranges argument
    data_max, data_min = __calculate_data_min_max(joint_dataset, columns,
                                                  axis_ranges)

    # Plot Datasets
    __plot_datasets(datasets_mod, ax, data_min, data_max, color_maps, columns,
                    color_column, x_axis, brush_criteria=brush_criteria)

    # Add color bars
    __add_color_bar(datasets, ax, dataset_names, color_maps, color_column,
                    invert, axis_labels, fontname_body)

    # Set numbers, axis labels, and axis spines
    __set_numbers_labels_axis(ax, data_min, data_max, columns, invert,
                              axis_labels, x_axis, plot_font)

    # Save file or display plot
    if file_name != '':
        plt.savefig(file_name)
    else:
        plt.show()
