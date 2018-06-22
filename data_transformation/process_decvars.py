import numpy as np
from copy import deepcopy


def process_decvars(dec_vars, utilities_names):
    data = {}
    for u in range(len(utilities_names)):
        u_dec_vars = {}
        u_dec_vars['Restriction\nTriggers'] = dec_vars[:, u]
        u_dec_vars['Transfer\nTrigger'] = 0. if u == 'Cary' else dec_vars[:, u + 4]
        u_dec_vars['ACFC'] = dec_vars[:, u + 11]
        u_dec_vars['Insurance'] = dec_vars[:, u + 15]
        data[utilities_names[u]] = u_dec_vars

    return data

def process_decvars_inverse(dec_vars, utilities_names, dec_vars_names_columns):
    data = {}
    for dv, col in dec_vars_names_columns.iteritems():
        data_dv = {}
        for name, u in zip(utilities_names, range(len(utilities_names))):
            if not (dv == 'Transfer\nTrigger' and name == 'Cary'):
                data_dv[name] = dec_vars[:, col + u]

        data[dv] = data_dv

    return data


def check_repeated(dec_vars_orig, objs_orig):
    objs = deepcopy(objs_orig)

    uq, ix_rep, ix_inv, cts = np.unique([tuple(row) for row in dec_vars_orig],
                                        return_index=True, return_inverse=True,
                                        return_counts=True, axis=0)

    ix_rep.sort()
    for i in np.where(cts - 1 > 0):
        rep_ix = ix_inv == i
        objs_repeated_mean = objs[rep_ix].mean(axis=0)
        objs[rep_ix] = objs_repeated_mean

    return deepcopy(dec_vars_orig)[ix_rep], objs[ix_rep], ix_rep


