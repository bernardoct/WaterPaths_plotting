import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import colorbar
from copy import deepcopy

from plotting.robustness_bar_chart import add_bubble_highlight

plt.rcParams['svg.fonttype'] = 'none'

def __basic_plot_formatting(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylim(-0.02, 1.02)

def __min_max_column_across_datasets(datasets, column):
    n_datasets = len(datasets)

    vmin = datasets[0][:, column].min()
    vmax = datasets[0][:, column].max()
    for i in range(1, n_datasets):
        vmin = np.min([vmin, datasets[i][:, column].min()])
        vmax = np.max([vmax, datasets[i][:, column].max()])

    return [vmin, vmax]

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


def __calculate_alphas(dataset, brush_criteria=dict(), base_alpha=1.):
    n_points = len(dataset)
    alphas = np.ones(n_points, dtype=float) * base_alpha

    not_brushed = np.array([True] * n_points)
    for b in brush_criteria:
        not_brushed = \
            np.multiply(not_brushed,
                        np.multiply(dataset[:, b] <= max(brush_criteria[b]),
                                    dataset[:, b] >= min(brush_criteria[b])))
    alphas[[not c for c in not_brushed]] = 0.05 # 0.07  * 150. / len(alphas)

    return alphas


def __plot_datasets(datasets_mod, ax, data_min, data_max, color_maps, columns,
                    color_column, x_axis, brush_criteria=dict(),
                    base_alpha=1.0, lw=1., same_scale=False,
                    highlight_sols=None):

    n_datasets = len(datasets_mod)

    if highlight_sols == None:
        highlight_sols = [{'ids': [], 'colors': [], 'labels': []}] * n_datasets

    if same_scale:
        color_min, color_max = __min_max_column_across_datasets(datasets_mod,
                                                              color_column)

    # Plot data sets
    i = 0
    dataset_normed_return = []
    for dataset, cmap, h in zip(datasets_mod, color_maps, highlight_sols):
        alphas = __calculate_alphas(dataset, brush_criteria=brush_criteria,
                                    base_alpha=base_alpha)

        not_filtered = np.where(alphas==base_alpha)
        print 'Dataset {} -- {}/{} solutions were not filtered: {}'\
            .format(i, len(not_filtered), len(alphas), not_filtered)
        i += 1

        if not same_scale:
            color_min, color_max = __min_max_column_across_datasets(
                [dataset], color_column)

        dataset_normed = (dataset - data_min) \
                               / (data_max - data_min + 1e-8)
        dataset_normed_return.append(dataset_normed)
        dataset_color_values = (dataset[:, color_column] - color_min) \
                                / (color_max - color_min + 1e-8)

        # Plot data
        for d, c, a in zip(dataset_normed, dataset_color_values, alphas):
            ax.plot(x_axis, d[columns], c=cmap(c), alpha=a, lw=lw)

        if len(h) > 0:
            for s, c, l in zip(h['ids'], h['colors'], h['labels']):
                ax.plot(x_axis, dataset_normed[s, columns], c=c, lw=lw + 1)

    return dataset_normed_return


def __add_color_bar(datasets, ax, dataset_names, color_maps, color_column,
                    invert, labels, fontname_body, fig, same_scale=False,
                    axis_number_formatting=[]):

    if len(axis_number_formatting) == 0:
        axis_number_formatting = ['{0:.2g}'] * len(labels)

    n_datasets = len(datasets)

    if same_scale:
        vmin, vmax = __min_max_column_across_datasets(datasets, color_column)
        normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap_axis = []
    for cmap, name, i in zip(color_maps, dataset_names, range(n_datasets)):
        # Set color bar auxiliary variables
        orientation = 'horizontal'

        if not same_scale:
            vmin = datasets[i][:, color_column].min()
            vmax = datasets[i][:, color_column].max()
            normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Reverse colormap is color_column is marked to be reversed.
        cmap_mod = cmap if invert[color_column] == 1 else cmap.reversed()

        # Create color bars
        cax, _ = colorbar.make_axes(ax, orientation=orientation, aspect=60)
        cbar = colorbar.ColorbarBase(cax, cmap=cmap_mod, norm=normalize,
                                     orientation=orientation)
        cmap_axis.append(cax)

        # Set color bar font
        cbar.set_label('{} - {}'.format(name, labels[color_column]),
                       **{'family': fontname_body})

        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([axis_number_formatting[color_column].format(vmin),
                             axis_number_formatting[color_column].format(vmax)])

        for l in cax.xaxis.get_ticklabels():
            l.set_family(fontname_body)

    # Set color bars positions
    size = fig.get_size_inches()
    axis_height = fig.get_size_inches()[1] * ax.get_position().height
    base_height = 0.65 * 5.

    cbar_height = 0.6 / size[1]
    title_height = 1.12 / size[1]
    spacing = 0.08
    base_width = 0.8
    width_cbar = (base_width - spacing * (n_datasets - 1)) / n_datasets
    ax.set_position([0.05, cbar_height + 0.07 * 5. / size[1], 0.9,
                     1. - cbar_height - title_height])
    for i in range(n_datasets):
        cmap_axis[i].set_position(
            [0.1 + (width_cbar + spacing * (n_datasets - 1)) * i,
             0., width_cbar, cbar_height])

    return cmap_axis


def __set_numbers_labels_axis(ax, fig, data_min, data_max, columns, invert,
                              labels, x_axis, plot_font,
                              axis_number_formating=[]):
    if len(axis_number_formating) == 0:
        axis_number_formating = ['{0:.2g}'] * len(x_axis)

    axis_height = fig.get_size_inches()[1] * ax.get_position().height
    base_height = 0.65 * 5.

    for x, f in zip(x_axis, axis_number_formating):
        ax.text(x, -0.07 * base_height / axis_height,
                f.format(invert[columns[x]] * data_min[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1. + 0.03 * base_height / axis_height,
                f.format(invert[columns[x]] * data_max[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1. + 0.09 * base_height / axis_height,
                np.array(labels)[columns][x], horizontalalignment='center',
                **plot_font)
        ax.plot((x, x), (-0.02, 1.02),
                c='black', alpha=0.3, lw=0.2)


def __add_highlight_labels(datasets, ax, highlight_sols=None, after_col=0):

    if highlight_sols != None:
        for d, h in zip(datasets, highlight_sols):
            if len(h) > 0:

                if after_col >= 0:
                    label_cols = range(after_col, d.shape[1] - 1)
                else:
                    label_cols = []

                while len(label_cols) < len(h['ids']):
                    label_cols += range(d.shape[1] - 1)

                for s, c, l, lc in zip(h['ids'], h['colors'], h['labels'],
                                       label_cols):
                    add_bubble_highlight(
                        (0.5 + lc, np.mean([d[s, lc], d[s, lc + 1]])),
                        c, l, ax)


def parallel_axis(datasets, columns, color_column, color_maps,
                  axis_labels, title, dataset_names, axis_ranges=(),
                  fontname_title='Gill Sans MT',
                  fontname_body='Open Sans Condensed', file_name='',
                  size=(9, 6), axis_to_invert=(), brush_criteria={}, lw=1.,
                  axis_number_formating=[], cbar_same_scale=False,
                  highlight_solutions=None, labels_after_col=0):

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
    fig, ax = plt.subplots(figsize=size)
    # fig.set_size_inches(size, forward=True)
    fig.suptitle(title, **plot_font_title)

    # Remove axis and ticks
    __basic_plot_formatting(ax)

    # Calculate axis_ranges or use axis_ranges argument
    data_max, data_min = __calculate_data_min_max(joint_dataset, columns,
                                                  axis_ranges)

    # Plot Datasets
    datasets_normed = __plot_datasets(datasets_mod, ax, data_min, data_max, color_maps, columns,
                    color_column, x_axis, brush_criteria=brush_criteria, lw=lw,
                    same_scale=cbar_same_scale,
                    highlight_sols=highlight_solutions)

    __add_highlight_labels(datasets_normed, ax,
                           highlight_sols=highlight_solutions,
                           after_col=labels_after_col)

    # Add color bars
    cbar_axis = __add_color_bar(datasets, ax, dataset_names, color_maps,
                                color_column, invert, axis_labels,
                                fontname_body, fig, same_scale=cbar_same_scale,
                                axis_number_formatting=axis_number_formating)

    # Set numbers, axis labels, and axis spines
    __set_numbers_labels_axis(ax, fig, data_min, data_max, columns, invert,
                              axis_labels, x_axis, plot_font,
                              axis_number_formating=axis_number_formating)

    # Save file or display plot
    if file_name != '':
        plt.savefig(file_name)
    else:
        plt.show()
