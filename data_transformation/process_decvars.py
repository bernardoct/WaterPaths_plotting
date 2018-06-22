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