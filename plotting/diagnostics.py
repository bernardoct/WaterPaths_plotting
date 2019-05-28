import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from multiprocessing import Pool
from functools import partial
from matplotlib import cm
import re

# matplotlib.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 8})


def plot_axis(df, data_columns, axes, y_labels, axis_column, weeks, c, alpha,
              scale):
    for c, ax, yl, s in zip(data_columns, axes[axis_column], y_labels, scale):
        if s == 'log':
            ax.set_yscale("log", nonposy='clip')
        min_both = min(df[c][weeks])
        # df[c][weeks].plot(ax=ax, lw=1, c=cm.get_cmap('tab10').colors[0], alpha=alpha)#`, legend=True)
        # ax.fill_between(weeks, df[c][weeks].as_matrix() + (0.1 if s == 'log' else 0), 0, color=cm.get_cmap('tab10').colors[0], alpha=alpha)
        ax.plot(weeks, df[c][weeks].as_matrix() + (0.1 if s == 'log' else 0), color=cm.get_cmap('tab10').colors[0], alpha=alpha)

def plot_diagnostics(df, my_columns, y_labels, weeks, axes, c, alpha, scale):
    my_columns = map(list, zip(*my_columns))  # transpose list of lists of axes

    n_columns = len(my_columns)
    axes = map(list, zip(*axes))  # transpose list of lists of axes

    for i in range(n_columns):
        plot_axis(df, my_columns[i], axes, y_labels, i, weeks, c, alpha, scale)

    for ax, yl in zip(axes[0], y_labels):
        l = ax.set_ylabel(yl, {'fontname': 'Open Sans Condensed', 'size': 12})
        l.set_rotation(0)
        l.set_horizontalalignment('right')
        l.set_verticalalignment('center')

        # ax.set_yscale(s)
    for axes_col in axes:
        axes_col[-1].set_xlabel('Week',
                                {'fontname': 'Open Sans Condensed', 'size': 12})


def create_utility_column_headers(utilities_ids, my_vars, vars_ids,
                                  sources_ids, y_labels_all):
    my_columns = []
    y_labels = []

    for v in vars_ids:
        my_columns_row = []
        for s in sources_ids:
            my_columns_row.append(utilities_ids[s] + my_vars[v])
        # print HB_columns_row
        my_columns.append(my_columns_row)

    for v in vars_ids:
        y_labels.append(y_labels_all[v])

    return my_columns, y_labels


def create_water_source_column_headers(source_id, vars, vars_ids, y_labels_all):
    my_columns = []
    y_labels = []

    for v in vars_ids:
        my_columns.append('{}{}'.format(source_id, vars[v]))

    for v in vars_ids:
        y_labels.append(y_labels_all[v])

    return my_columns, y_labels


def create_data_frame(files):
    utility_file, policy_file = files
    my_df_utilities = pd.read_csv(utility_file)
    my_df_policy = pd.read_csv(policy_file)
    return pd.concat([my_df_utilities, my_df_policy], axis=1)


def print_triggers(axes, sol, utilities_ids, my_vars, vars_ids,
                   utilities_ids_int, rest_triggers, transfer_triggers,
                   ins_triggers, inf_triggers):
    st_rof_row = np.where(np.array(my_vars)[vars_ids] == 'st_rof')[0]
    lt_rof_row = np.where(np.array(my_vars)[vars_ids] == 'lt_rof')[0]

    if len(st_rof_row) > 0:
        for ax, u in zip(axes[st_rof_row[0]], utilities_ids_int):
            xlim = ax.get_xlim()

            ax.plot(xlim, [rest_triggers[u][sol]] * 2, label='rt', lw=0.5,
                    c='y')
            ax.plot(xlim, [transfer_triggers[u][sol]] * 2, label='tt', lw=0.5,
                    c='b')
            ax.plot(xlim, [ins_triggers[u][sol]] * 2, label='inst', lw=0.5,
                    c='m')
            ax.legend(ncol=3)

    if len(lt_rof_row) > 0:
        for ax, u in zip(axes[lt_rof_row[0]], utilities_ids_int):
            xlim = ax.get_xlim()

            ax.plot(xlim, [inf_triggers[u][sol]] * 2, label='inft', c='r')


def plot_utility_comparison_parallel_io(weeks, sols, rest_triggers,
                                        transfer_triggers, ins_trigger,
                                        inf_triggers, rdms=[''],
                                        output_directory='', nprocs=1):
    nreals = 100  # 500

    for s in sols:
        for r in rdms:
            dfs = import_data(output_directory, nreals, (s, r), nprocs=nprocs)

            ''' Columns to be plotted files '''
            utilities_ids = ['0', '1', '2', '3']
            my_vars = ['st_vol', 'st_rof', 'lt_rof', 'rest_demand',
                       'unrest_demand',
                       'transf', 'rest_m', 'cont_fund', 'ins_pout', 'ins_price']
            scale = ['linear', 'linear', 'linear', 'linear', 'linear', 'linear',
                     'linear', 'linear', 'log', 'log', 'linear']
            y_labels_all = ['Storage\n[MG]', 'ST-ROF\n[-]', 'LT-ROF\n[-]',
                            'Restricted Demand\n[MGD]',
                            'Unrest. Demand\n[MGD]', 'Transfered volume\n[MGD]',
                            'Restriction dem.\nmitigation',
                            'Fund size\n[MM$]', 'Payouts\n[MM$]', 'Price\n[MM$]']
            rof_triggers = ['Restriction', 'Transfers', 'Insurance']

            vars_ids = [0, 1, 5, 6, 7, 8, 9]
            utilities_ids_int = [1, 3]

            my_columns, y_labels = create_utility_column_headers(utilities_ids,
                                                                 my_vars,
                                                                 vars_ids,
                                                                 utilities_ids_int,
                                                                 y_labels_all)

            ''' Don't touch '''
            alpha = 3. / len(dfs)
            for df in dfs:
                for u in utilities_ids_int:
                    df['{}rest_m'.format(u)] = 1. - df['{}rest_m'.format(u)]

            fig, axes = plt.subplots(nrows=len(my_columns),
                                     ncols=len(my_columns[0]),
                                     figsize=(10 * 1.3, 9 * 1.3), sharex=True)
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.95,
                                wspace=0.13, hspace=0.43)
            for df in dfs:
                plot_diagnostics(df, my_columns, y_labels, weeks, axes, 'r',
                                 alpha, np.array(scale)[vars_ids])

            # ins_max_min = [0, 60]
            # ax_ylims = [[0, 3.5e4], [0, 1], [0, 1], [0, 700], [0, 700], [0, 125], [0, 0.5], [0, 1000], ins_max_min, ins_max_min]
            #
            # for rax, c in zip(axes, vars_ids):
            #   for ax in rax:
            #       ax.set_ylim(ax_ylims[c])

            for rax in axes:
                ax_lims = np.array([ax.get_ylim() for ax in rax])
                lims = [ax_lims[:, 0].min(), ax_lims[:, 1].max()]

                for ax in rax:
                    ax.set_ylim(lims)

            for ax, title in zip(axes[0], np.array(
                    ['OWASA', 'Durham', 'Cary', 'Raleigh'])[utilities_ids_int]):
                ax.set_title(title)

            print_triggers(axes, s, utilities_ids, my_vars, vars_ids,
                           utilities_ids_int, rest_triggers, transfer_triggers,
                           ins_trigger, inf_triggers)

            plt.suptitle('Solution {}'.format(s))
            plt.savefig('{}Diagnostics_s{}_RDM{}.png'.format(output_directory, s, r))
            plt.savefig('{}Diagnostics_s{}_RDM{}.svg'.format(output_directory, s, r))


def import_data(output_directory, nreals, sr, nprocs=1):
    s, r, _ = sr

    ''' Input files '''
    if r == '':
        utilities_files = glob(
            '{}Utilities_s{}_r*.csv'.format(output_directory, s))[:nreals]
        policy_files = glob(
            '{}Policies_s{}_r*.csv'.format(output_directory, s))[:nreals]
    else:
        utilities_files = glob(
            '{}Utilities_s{}_RDM{}_r*.csv'.format(output_directory, s, r))[:nreals]
        policy_files = glob(
            '{}Policies_s{}_RDM{}_r*.csv'.format(output_directory, s, r))[:nreals]
    files = zip(utilities_files, policy_files)

    ''' Don't touch '''
    if nprocs > 1:
        dfs = Pool(nprocs).map(create_data_frame, files)
    else:
        dfs = [create_data_frame(fs) for fs in files]

    return dfs


def plot_one_figure_utilities(weeks, rest_triggers, transfer_triggers, ins_trigger,
                              inf_triggers, output_directory, nreals, figsize, nprocs,
                              sr):
    s, r, name = sr
    print 'plotting {}'.format(name)

    ''' Columns to be plotted files '''
    utilities_ids = ['0', '1', '2', '3']
    my_vars = ['st_vol', 'st_rof', 'lt_rof', 'rest_demand', 'unrest_demand',
               'transf', 'rest_m', 'cont_fund', 'ins_pout', 'ins_price',
               'capacity']
    scale = ['linear', 'linear', 'linear', 'linear', 'linear', 'linear',
             'linear', 'linear', 'log', 'log', 'linear']
    y_labels_all = ['Storage\n[MG]', 'ST-ROF\n[-]', 'LT-ROF\n[-]',
                    'Restricted\nDemand [MGD]',
                    'Unrestricted\nDemand [MGD]', 'Transfered\nVolume\n[MGD]',
                    'Restriction\nCoefficient [-]',
                    'Fund size\n[MM$]', 'Insurance\nPayouts [MM$]', 'Insurance\nPrice [MM$]',
                    'Storage\n[MG]']
    rof_triggers = ['Restriction', 'Transfers', 'Insurance']

    vars_ids = [0, 1, 3, 4]
    # vars_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # vars_ids = [2, 10]
    utilities_ids_int = [1, 2, 3]

    my_columns, y_labels = create_utility_column_headers(utilities_ids,
                                                         my_vars, vars_ids,
                                                         utilities_ids_int,
                                                         y_labels_all)

    dfs = import_data(output_directory, nreals, sr, nprocs=nprocs)

    alpha = 3. / len(dfs)
    for df in dfs:
        for u in utilities_ids_int:
            df['{}rest_m'.format(u)] = 1. - df['{}rest_m'.format(u)]

    fig, axes = plt.subplots(nrows=len(my_columns), ncols=len(my_columns[0]),
                             figsize=figsize, sharex=True,
                             sharey='row')
    # plt.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.95,
    #                     wspace=0.13, hspace=0.43)
    for df in dfs:
        plot_diagnostics(df, my_columns, y_labels, weeks, axes, 'r', alpha,
                         np.array(scale)[vars_ids])

    ins_max_min = [1, 50]
    ax_ylims = [[0, 3.5e4], [0, 1], [0, 0.3], [0, 2100], [0, 2100], [0, 125],
                [0, 0.5], [-100, 500], ins_max_min, ins_max_min, [0, 3.5e4]]

    for rax, c in zip(axes, vars_ids):
        for ax in rax:
            ax.set_ylim(ax_ylims[c])
            ax.set_xlim(weeks[0], weeks[-1])
            ax.set_xticks([i * 52 for i in range(0, 46, 15)])
            ax.set_xticklabels(
                ['{:.0f}'.format(2015. + 1. * x / 52.148) for x in
                 ax.get_xticks()],
                {'fontname': 'Open Sans Condensed', 'size': 11})
            if scale[c] != 'log':
                ax.set_yticks(ax.get_ylim())
                ax.set_yticklabels(['{:.0f}'.format(x) for x in ax.get_yticks()],
                               {'fontname': 'Open Sans Condensed', 'size': 11})

    # for rax in axes:
    #   ax_lims = np.array([ax.get_ylim() for ax in rax])
    #   lims = [ax_lims[:, 0].min(), ax_lims[:, 1].max()]

    #   for ax in rax:
    #       ax.set_ylim(lims)

    for ax, title in zip(axes[0],
                         np.array(['OWASA', 'Durham', 'Cary', 'Raleigh'])[
                             utilities_ids_int]):
        ax.set_title(title, {'fontname': 'Gill Sans MT', 'size': 12})

    print_triggers(axes, s, utilities_ids, my_vars, vars_ids, utilities_ids_int,
                   rest_triggers, transfer_triggers, ins_trigger, inf_triggers)

    plt.suptitle(name, **{'fontname': 'Gill Sans MT', 'size': 14})
    print name
    # plt.savefig('{}Diagnostics_s{}_RDM{}.png'.format(output_directory, s, r))
    # plt.savefig('{}Diagnostics_s{}_RDM{}.svg'.format(output_directory, s, r))
    plt.show()


def plot_one_figure_water_sources(weeks, output_directory, nreals, figsize, sr):
    s, r, name = sr

    ''' Columns to be plotted files '''
    vars = ['volume', 's_area', 'demand', 'up_spill', 'ww_inflow', 'catch_inflow', 'evap', 'ds_spill', 'alloc_0',
            'alloc_1', 'alloc_2', 'alloc_3']
    scale = ['linear', 'linear', 'linear', 'linear', 'linear', 'linear',
             'linear', 'linear', 'linear', 'linear', 'linear', 'linear']
    y_labels_all = ['Storage\n[MG]', 'Area\n[af]', 'Demand\n[MGD]',
                    'Upstream Spill\n[MGD]',
                    'WW Efluent\nInflow [MGD]', 'Catchment\n Gain [MGD]',
                    'Evaporation\n[MGD]',
                    'Downstream\nSpillage [MGD]', 'Allocation 0\nStored [MG]', 'Allocation 1\nStored [MG]',
                    'Allocation 2\nStored [MG]', 'Allocation 3\nStored [MG]']

    vars_ids = [0, 8, 9, 10, 11]
    source_id = 6

    rows, y_labels = create_water_source_column_headers(source_id, vars, vars_ids, y_labels_all)

    dfs = [pd.read_csv(f) for f in glob(output_directory + 'WaterSources_s{}_RDM{}_r*'.format(s, r))[:nreals]]

    alpha = 3. / len(dfs)

    fig, axes = plt.subplots(nrows=len(rows), ncols=1,
                             figsize=figsize,
                             sharey='row')
    # plt.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.95,
    #                     wspace=0.13, hspace=0.43)
    for df in dfs:
        plot_diagnostics(df, [[ro] for ro in rows], y_labels, weeks, [[ax] for ax in axes], 'r', alpha,
                         np.array(scale)[vars_ids])

    # ins_max_min = [1, 400]
    # ax_ylims = [[0, 3.5e4], [0, 1], [0, 0.3], [0, 700], [0, 750], [0, 125],
    #             [0, 0.5], [0, 1000], ins_max_min, ins_max_min, [0, 3.5e4]]

    for ax, c in zip(axes, vars_ids):
        # ax.set_ylim(ax_ylims[c])
        ax.set_xlim(weeks[0], weeks[-1])
        ax.set_xticks([i * 52 for i in range(0, 46, 15)])
        ax.set_xticklabels(
                ['{:.0f}'.format(2015. + 1. * x / 52) for x in
                 ax.get_xticks()],
                {'fontname': 'Open Sans Condensed', 'size': 11})
        if scale[c] != 'log':
            ax.set_yticks(ax.get_ylim())
            ax.set_yticklabels(['{:.0f}'.format(x) for x in ax.get_yticks()],
                               {'fontname': 'Open Sans Condensed', 'size': 11})

    # for rax in axes:
    #   ax_lims = np.array([ax.get_ylim() for ax in rax])
    #   lims = [ax_lims[:, 0].min(), ax_lims[:, 1].max()]

    #   for ax in rax:
    #       ax.set_ylim(lims)

    # for ax, title in zip(axes[0],
    #                      np.array(['OWASA', 'Durham', 'Cary', 'Raleigh'])[
    #                          utilities_ids_int]):
    #     ax.set_title(title, {'fontname': 'Gill Sans MT', 'size': 12})

    plt.suptitle(name, **{'fontname': 'Gill Sans MT', 'size': 14})
    print '{}Diagnostics_ws_s{}_RDM{}.png'.format(output_directory, s, r)
    plt.savefig('{}Diagnostics_ws_s{}_RDM{}_{}.png'.format(output_directory, name, s, r))
    plt.savefig('{}Diagnostics_ws_s{}_RDM{}_{}.svg'.format(output_directory, name, s, r))


def plot_utility_comparison_parallel_figures(weeks, sols, rest_triggers,
                                             transfer_triggers, ins_trigger,
                                             inf_triggers, rdms_all_sols=[['']],
                                             output_directory='',
                                             figsize=(10 * 1.3, 9 * 1.3),
                                             nprocs=2, sol_rdm_names=[]):

    nreals = 100  # 500
    partial_plot_figure = partial(plot_one_figure_utilities, weeks, rest_triggers,
                                  transfer_triggers, ins_trigger, inf_triggers,
                                  output_directory, nreals, figsize, 1)
    srn = []
    if len(sol_rdm_names) == 0:
        for s, rdms in zip(sols, rdms_all_sols):
            sol_rdm_names.append([])
            for r in rdms:
                sol_rdm_names[-1].append('Solution {} RDM {}'.format(s, r))

    for s, rdms, names in zip(sols, rdms_all_sols, sol_rdm_names):
        for r, name in zip(rdms, names):
            srn.append((s, r, name))

    # Pool(4).map(partial_plot_figure, srn)
    for sr_ in srn:
        plot_one_figure_utilities(weeks, rest_triggers, transfer_triggers, ins_trigger, inf_triggers, output_directory,
                                  nreals, figsize, 1, sr_)


def plot_water_sources_parallel_figures(weeks, sols, rdms_all_sols=[['']],
                                             output_directory='',
                                             figsize=(10 * 1.3, 9 * 1.3),
                                             nprocs=2, sol_rdm_names=[]):

    nreals = 100  # 500
    partial_plot_figure = partial(plot_one_figure_water_sources, weeks, output_directory, nreals, figsize)

    srn = []
    if len(sol_rdm_names) == 0:
        for s, rdms in zip(sols, rdms_all_sols):
            sol_rdm_names.append([])
            for r in rdms:
                sol_rdm_names[-1].append('Solution {} RDM {}'.format(s, r))

    for s, rdms, names in zip(sols, rdms_all_sols, sol_rdm_names):
        for r, name in zip(rdms, names):
            srn.append((s, r, name))

    # Pool(min(nprocs, len(srn))).map(partial_plot_figure, srn)
    for sr_ in srn:
        partial_plot_figure(sr_)
