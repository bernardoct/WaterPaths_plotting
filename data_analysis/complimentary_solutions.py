import numpy as np

from data_transformation.process_rdm_objectives import group_objectives


def find_complimentary(sol_pass_fail, sols_pass_fails):
    ix, new_passes = -1, 0
    for i in range(len(sols_pass_fails)):
        s = sols_pass_fails[i]
        new_passes_s = sum(-sol_pass_fail + s == 1)
        if new_passes_s > new_passes:
            ix = i
            new_passes = new_passes_s

    return ix, new_passes


def calc_pass_fail(objectives_by_solution, non_crashed_by_solution, criteria, apply_on_objs, max_min):
    pass_fail = []
    apply_on_objs = list(apply_on_objs)
    n_rdms = max([len(nc) for nc in non_crashed_by_solution])

    try:
        pass_fail = np.load('pass_fail_all.npy')
    except:
        for objs_rdm, non_crashed in zip(objectives_by_solution, non_crashed_by_solution):
            print 'Pass-fail {}'.format(len(pass_fail))
            objs = group_objectives(objs_rdm, max_min)
            pass_fail_obj = np.zeros(n_rdms)

            pass_fail_sol_non_crashed = np.zeros(len(objs))
            for i in range(len(objs)):
                obj = objs[i]
                pass_fail_sol_non_crashed[i] = sum([o < c if m == 'min' else o > c for o, c, m in
                                                    zip(obj[apply_on_objs], criteria, max_min[apply_on_objs])])

            pass_fail_obj[non_crashed] = pass_fail_sol_non_crashed == len(criteria)

            pass_fail.append(pass_fail_obj)

        pass_fail = np.array(pass_fail)
        np.save('pass_fail_all.npy', pass_fail)

    return pass_fail


def find_complimentary_solution(sol, objectives_by_solution, non_crashed_by_solution, criteria,
                                apply_on_objs, max_min):

    pass_fail_all_sols = calc_pass_fail(objectives_by_solution, non_crashed_by_solution, criteria,
                                        apply_on_objs, max_min)

    return find_complimentary(pass_fail_all_sols[sol], pass_fail_all_sols)