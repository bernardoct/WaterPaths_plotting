import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import colorbar
from copy import deepcopy


def basic_plot_formatting(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylim(-0.02, 1.02)


def calculate_data_min_max(joint_dataset, columns, ranges):
    if len(ranges) == 0:
        data_max = np.max(joint_dataset, axis=0)
        data_min = np.min(joint_dataset, axis=0)
    elif len(ranges) != len(columns) and len(ranges) != len(joint_dataset[0]):
        raise ValueError('Number of ranges != number of columns.')
    else:
        data_max = np.array(ranges).T[1]
        data_min = np.array(ranges).T[0]

    return data_max, data_min


def invert_axis(datasets_mod, n_columns_datasets, invert_axes, ranges,
                brush_criteria):
    # Invert axis, if desired
    invert = np.ones(n_columns_datasets, dtype=float)
    for axis in invert_axes:
        invert[axis] = -1.
        ranges[axis] *= -1
        ranges[axis] = ranges[axis][::-1]
        if axis in brush_criteria:
            brush_criteria[axis] = [-a for a in brush_criteria[axis][::-1]]

        for dataset in datasets_mod:
            dataset[:, axis] *= -1

    return invert


def calculate_alphas(dataset, brush_criteria=dict(), base_alpha=0.5):
    n_points = len(dataset)
    alphas = np.ones(n_points, dtype=float) * base_alpha

    not_brushed = np.array([True] * n_points)
    for b in brush_criteria:
        not_brushed = \
            np.multiply(not_brushed,
                        np.multiply(dataset[:, b] < max(brush_criteria[b]),
                                    dataset[:, b] > min(brush_criteria[b])))
    alphas[[not c for c in not_brushed]] = 0.02

    return alphas


def plot_datasets(datasets_mod, ax, data_min, data_max, colors, columns,
                  color_column, x_axis, brush_criteria=dict(), base_alpha=0.5):

    # Plot data sets
    for dataset, cmap in zip(datasets_mod, colors):
        alphas = calculate_alphas(dataset, brush_criteria=brush_criteria,
                                  base_alpha=base_alpha)
        dataset_normed = (dataset - data_min + 1e-8) \
                               / (data_max - data_min)
        dataset_color_values = (dataset - dataset.min(0) + 1e-8) \
                                / dataset.ptp(0)

        # Plot data
        for d, dc, a in zip(dataset_normed, dataset_color_values, alphas):
            ax.plot(x_axis, d[columns], c=cmap(dc[color_column]), alpha=a)


def add_color_bar(datasets, ax, dataset_names, colors, color_column, invert,
                  labels, fontname_body):
    n_datasets = len(datasets)
    axes = []
    for cmap, name, i in zip(colors, dataset_names, range(n_datasets)):
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
        axes.append(cax)

        # Set color bar font
        cbar.set_label('{} {}'.format(name, labels[color_column]),
                       **{'family': fontname_body})

        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([vmin, vmax])

        for l in cax.xaxis.get_ticklabels():
            l.set_family(fontname_body)

    # Set color bars positions
    width_cbar = (0.8 - 0.05 * (n_datasets - 1)) / n_datasets
    ax.set_position([0.05, 0.17, 0.9, 0.65])
    for i in range(n_datasets):
        axes[i].set_position([0.1 + (width_cbar + 0.05) * i,
                              0., width_cbar, 0.1])


def set_numbers_labels_axes(ax, data_min, data_max, columns, invert, labels,
                            x_axis, plot_font):
    for x in x_axis:
        ax.text(x, -0.07, '{0:.2g}'.format(invert[x] * data_min[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1.03, '{0:.2g}'.format(invert[x] * data_max[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1.09, np.array(labels)[columns][x],
                horizontalalignment='center', **plot_font)
        ax.plot((x, x), (-0.02, 1.02),
                c='black', alpha=0.3, lw=0.2)


def paxis_matplotlib_hack(datasets, columns, color_column, colors, labels,
                          title, dataset_names, ranges=(),
                          fontname_title='Gill Sans MT',
                          fontname_body='CMU Bright', file_name='',
                          size=(9, 6), invert_axes=(), brush_criteria=dict()):

    datasets_mod = deepcopy(datasets)
    ranges = np.array(ranges)
    brush_criteria = deepcopy(brush_criteria)
    n_columns_datasets = len(datasets_mod[0][0])
    n_axis = len(columns)
    x_axis = range(n_axis)

    # Invert axis, if needed
    invert = invert_axis(datasets_mod, n_columns_datasets, invert_axes, ranges,
                         brush_criteria=brush_criteria)

    # Create combined data set
    joint_dataset = np.vstack(datasets_mod)

    plot_font = {'fontname': fontname_body}
    plot_font_title = {'fontname': fontname_title, 'fontsize': 16}

    # Set title
    fig, ax = plt.subplots()
    fig.set_size_inches(size, forward=True)
    fig.suptitle(title, **plot_font_title)

    # Remove axes and ticks
    basic_plot_formatting(ax)

    # Calculate ranges or use ranges argument
    data_max, data_min = calculate_data_min_max(joint_dataset, columns, ranges)

    # Plot Datasets
    plot_datasets(datasets_mod, ax, data_min, data_max, colors, columns,
                  color_column, x_axis, brush_criteria=brush_criteria)

    # Add color bars
    add_color_bar(datasets, ax, dataset_names, colors, color_column, invert,
                  labels, fontname_body)

    # Set numbers, axis labels, and axis spines
    set_numbers_labels_axes(ax, data_min, data_max, columns, invert, labels,
                            x_axis, plot_font)

    # Save file or display plot
    if file_name != '':
        plt.savefig(file_name)
    else:
        plt.show()
