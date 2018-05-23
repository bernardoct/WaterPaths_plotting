import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib import cm


def calculate_pseudo_robustness_uniform(pass_fail):
    return 1.* sum(pass_fail * 1) / len(pass_fail)


def plot_beta(x, p, pass_fail, ax, cmap=cm.get_cmap('cool'), from_middle=0.35):
    xu = np.linspace(0., 1., 200)

    ax.scatter(x[pass_fail], p[pass_fail], c=cmap(0.5 - from_middle), s=0.01)
    ax.scatter(x[pass_fail == False], p[pass_fail == False], c=cmap(0.5 + from_middle), s=0.01)


def calculate_pseudo_robustness_beta(pass_fail, rdm_factors, base_parameter,
                                     lows, highs, means, plot=False):

    if not np.array_equal(rdm_factors.shape, (len(pass_fail), len(means))):
        raise Exception(('rdm_factors shape is {} but pass_fail and means have '
               + 'lengths of {} and {}').format(rdm_factors.shape,
                                               len(pass_fail), len(means)))

    if plot:
        fig, ax = plt.subplots()

    pseudo_robustness = 0.
    for rdm_factor, mean, low, high in zip(rdm_factors.T, means, lows, highs):
        rdm_ptp = high - low
        m = (mean - low) / rdm_ptp

        if max(rdm_factor) <= m:
            a = base_parameter
            b = 1. / m * (a - 1 - m * a + 2 * m)
        else:
            b = base_parameter
            a = ((b - 2.) * m + 1) / (1. - m)

        rdm_factor_norm = (rdm_factor - low) / rdm_ptp
        p_rdm_beta = beta.pdf(rdm_factor_norm, a, b)

        pseudo_robustness += sum(p_rdm_beta[pass_fail]) / sum(p_rdm_beta)

        if plot:
            plot_beta(rdm_factor_norm, p_rdm_beta, pass_fail, ax)

    if plot:
        plt.show()

    return pseudo_robustness / len(means)