import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as po
import matplotlib.colors as mcolors


plotly.tools.set_credentials_file(username='bct52',
                                  api_key='pH2l59HCGPPUWSC8RdSA')


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


def cmocean_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


def combine_colormaps(cm1, cm2, range1, range2):
    print range1, range2
    colors = []
    delta = range2[1] - range1[0]
    y0 = range1[0]
    range1 -= y0
    range2 -= y0
    range1 /= delta
    range2 /= delta
    print y0, delta
    print range1, range2

    colors.append((range1[0], cm1(0.)))
    colors.append((range1[1], cm1(1.)))
    colors.append((range2[0], cm2(0.)))
    colors.append((range2[1], cm2(1.)))

    return mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)


def paxis_plot_hack(datasets, columns, color_column, colors, labels,
                    color_bar_label, title, ranges):

    # Add extra color column to dataset
    datasets_extra_column = []
    for i in range(len(datasets)):
        lines_data = len(datasets[i])
        color_column_array = datasets[i][:, color_column] + np.ones(lines_data) * i

        datasets_extra_column.append(np.hstack((color_column_array[:, None],
                                                datasets[i])))
    data = np.vstack(datasets_extra_column)

    # Create color map
    colormap_mpl = combine_colormaps(colors[0], colors[1],
                                     [np.min(datasets[0][:, color_column]),
                                      np.max(datasets[0][:, color_column])],
                                     [np.min(datasets[1][:, color_column]) + 1.,
                                      np.max(datasets[1][:, color_column]) + 1.])

    colormap_plotly = matplotlib_to_plotly(colormap_mpl, 255)

    # create parallel axis plots
    plot = go.Parcoords(
        line=dict(color=data[:, 0],
                  colorscale=colormap_plotly,
                  cmin=np.min(data[:, 0]),
                  cmax=np.max(data[:, 0]),
                  # showscale = True,
                  # colorbar = dict(ticks='inside', title=labels[color_column])
                  ),
        dimensions=list([dict(label=labels[c], values=data[:, c + 1],
                              range=ranges[c]) for c in columns])
    )

    # create layout
    layout = go.Layout(
        title=title,
        font=dict(family='Gill Sans MT', size=20, color='#7f7f7f')
    )

    # fig = go.Figure(data=plots, layout=layout)
    fig = dict(data=[plot], layout=layout)
    # py.iplot(fig, filename='parcoords.html')
    po.plot(fig, filename='parcoords.html')
