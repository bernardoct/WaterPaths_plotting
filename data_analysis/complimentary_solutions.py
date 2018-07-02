import numpy as np


def find_complimentary_solution(sol_pass_fail, sols_pass_fails):
    ix, new_passes = -1, 0
    for i in range(len(sols_pass_fails)):
        s = sols_pass_fails[i]
        new_passes_s = sum(sol_pass_fail + s == 1)
        if new_passes_s > new_passes:
            ix = i
            new_passes = new_passes_s

    return ix, new_passes

def calc_pass_fail(objectives_by_solution, criteria):
    pass_fail = []

    # for o in objectives_by_solution:
