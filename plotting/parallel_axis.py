import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import colorbar


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


def paxis_matplotlib_hack(datasets, columns, color_column, colors, labels,
                          title, dataset_names, ranges=(),
                          fontname='CMU Bright', file_name='',
                          sequence_datasets=False):

    if sequence_datasets and file_name == '':
        raise ValueError('If sequence data sets is true, a file name must '
                         'be passed')

    # Create combined data set
    joint_dataset = np.vstack(datasets)

    n_datasets = len(datasets)
    n_axis = len(columns)
    x_axis = range(n_axis)

    plot_font = {'fontname': fontname}

    # Set title
    fig, ax = plt.subplots()
    fig.suptitle(title, **plot_font)

    # Remove axes and ticks
    basic_plot_formatting(ax)

    # Calculate ranges or use ranges argument
    data_max, data_min = calculate_data_min_max(joint_dataset, columns, ranges)

    # Plot data sets
    for dataset, cmap in zip(datasets, colors):
        dataset_normed = (dataset - data_min + 1e-8) \
                               / (data_max - data_min)
        dataset_color_values = (dataset - dataset.min(0) + 1e-8) \
                                / dataset.ptp(0)

        # Plot data
        for d, dc in zip(dataset_normed, dataset_color_values):
            ax.plot(x_axis, d[columns],
                    c=cmap(dc[color_column]),
                    alpha=0.5)

    # Add color bars
    axes = []
    for cmap, name, i in zip(colors, dataset_names, range(n_datasets)):
        # Create color bar
        orientation = 'horizontal'
        normalize = mcolors.Normalize(
            vmin=datasets[i][:, color_column].min(),
            vmax=datasets[i][:, color_column].max()
        )
        cax, _ = colorbar.make_axes(ax, orientation=orientation, aspect=60)
        cbar = colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize,
                                     orientation=orientation)
        axes.append(cax)

        # Set color bar font
        cbar.set_label('{} {}'.format(name, labels[color_column]),
                       **{'family': fontname})
        for l in cax.xaxis.get_ticklabels():
            l.set_family(fontname)

    # Set color bars positions
    width_cbar = (0.8 - 0.05 * (n_datasets - 1)) / n_datasets
    ax.set_position([0.05, 0.17, 0.9, 0.65])
    for i in range(n_datasets):
        axes[i].set_position([0.1 + (width_cbar + 0.05) * i,
                              0., width_cbar, 0.1])

    # Set numbers, axis labels, and axis spines
    for x in x_axis:
        ax.text(x, -0.07, '{0:.2g}'.format(data_min[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1.03, '{0:.2g}'.format(data_max[columns][x]),
                horizontalalignment='center', **plot_font)
        ax.text(x, 1.09, np.array(labels)[columns][x],
                horizontalalignment='center', **plot_font)
        ax.plot((x, x), (-0.02, 1.02),
                c='black', alpha=0.3, lw=0.2)

    plt.show()
