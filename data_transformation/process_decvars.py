import numpy as np

def process_decvars(dec_vars, utilities_names):
    data = {}
    for u in range(len(utilities_names)):
        u_dec_vars = {}
        u_dec_vars['restriction_triggers'] = dec_vars[:, u]
        u_dec_vars['transfer_triggers'] = dec_vars[:, u + len(utilities_names)]
        data[utilities_names[u]] = u_dec_vars

    return data