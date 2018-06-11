import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pyDOE import lhs

# plt.style.use('ggplot')


def trend(x, A, T, p):
    t = A * np.sin(2. * np.pi * x / T + p) - A * np.sin(p)
    return t


def plot_sinusoids(samples, ny, wy=52.148):
    print(np.min(samples[:, 0]), np.max(samples[:, 0]))
    print(np.min(samples[:, 1]), np.max(samples[:, 1]))
    print(np.min(samples[:, 2]), np.max(samples[:, 2]))

    x = np.arange(0, int(wy * ny) + 1)
    all_sinusoids = np.zeros((len(samples), len(x)))

    for i in range(len(samples)):
        y = trend(x, samples[i, 0], samples[i, 1], samples[i, 2])
        all_sinusoids[i, :] = y

    max_mult = all_sinusoids.max()
    min_mult = all_sinusoids.min()

    rmax = list(np.where(all_sinusoids.max(axis=1) == max_mult))[0][0]
    rmin = list(np.where(all_sinusoids.min(axis=1) == min_mult))[0][0]

    print(rmax, rmin)

    for i in [rmin, rmax]:
        y = trend(x, samples[i, 0], samples[i, 1], samples[i, 2])
        all_sinusoids[i, :] = y
        plt.plot(x, y, alpha=0.2, c='r')

    plt.xlabel('weeks [-]')
    plt.ylabel('mu multiplier [-]')
    plt.title('Flow mean multiplier')
    plt.show()


def rescale(v, l, h):
    v = l + v * (h - l)
    return v


def sample_uniform(lows, highs, means, nsamples, nparams, ny, wy=52.148):
    samples_raw = lhs(nparams, nsamples)
    samples_raw[:, 0] *= 0.01
    samples_raw[:, 1] = samples_raw[:, 1] * (0.1 * 2 * wy * ny) + 2 * wy * ny
    samples_raw[:, 2] = samples_raw[:, 2] * (0.1 * wy * 8) + wy * 8

    samples = np.zeros((nsamples, nparams))

    for i in range(3):
        samples[:, i] = samples_raw[:, i]

    for i in range(3, nparams):
        samples[:, i] = rescale(samples_raw[:, i], lows[i], highs[i])

    samples_including_built = np.zeros((nsamples, 62))
    valid_rdms = range(12) + range(26, 62)
    for i in range(len(lows)):
        samples_including_built[:, valid_rdms[i]] = samples[:, i]

    # plot_sinusoids(samples[:, :3], ny, wy)

    np.savetxt('TestFiles/rdm_inflows_reeval.csv', samples_including_built[:, :3])
    np.savetxt('TestFiles/rdm_utilities_reeval.csv',
               samples_including_built[:, 3:7])
    np.savetxt('TestFiles/rdm_dmp_reeval.csv', samples_including_built[:, 7:11])
    np.savetxt('TestFiles/rdm_water_sources_reeval.csv',
               samples_including_built[:, 11:])


def sample_beta(lows, highs, means, nsamples, nparams, base_factor,
                cmap=cm.get_cmap('cool')):
    # samples = np.zeros((nsamples, nparams))

    samples_including_built = np.zeros((nsamples, 62))
    valid_rdms = range(12) + range(26, 62)

    for i in range(len(lows)):
        norm = Normalize(vmin=lows[i], vmax=highs[i])
        mean = norm(means[i])
        if highs[i] == means[i]:
            a = base_factor
            b = 1. / mean * (a - 1 - mean * a + 2 * mean)
        else:
            b = base_factor
            a = ((b - 2.) * mean + 1) / (1. - mean)

        samples = beta.rvs(a, b, size=nsamples)
        samples_including_built[:, valid_rdms[i]] = samples

    fig, axes = plt.subplots(8, 6, sharex='all')#, sharey='all')
    axes_it = axes.reshape(-1)
    for i in range(len(lows)):
        norm = Normalize(vmin=lows[i], vmax=highs[i])
        mean = norm(means[i])
        print mean
        if highs[i] == means[i]:
            a = base_factor
            b = 1. / mean * (a - 1 - mean * a + 2 * mean)
        else:
            b = base_factor
            a = ((b - 2.) * mean + 1) / (1. - mean)

        x = np.linspace(0., 1., 200)
        axes_it[i].plot(x, beta.pdf(x, a, b), 'r-', lw=2, alpha=0.6,
                        label='beta pdf', c=cmap(float(i) / nparams))
        axes_it[i].hist(samples_including_built[:, valid_rdms[i]], normed=True,
                        alpha=0.2, color=cmap(float(i) / nparams))
    plt.show()

    # np.savetxt('TestFiles/rdm_inflows_reeval.csv',
    # samples_including_built[:, :3])
    suffix = 'reeval_beta_{}.csv'.format(base_factor)
    np.savetxt('TestFiles/rdm_utilities_{}'.format(suffix),
               samples_including_built[:, 3:7], delimiter=',')
    np.savetxt('TestFiles/rdm_dmp_{}'.format(suffix),
               samples_including_built[:, 7:11], delimiter=',')
    np.savetxt('TestFiles/rdm_water_sources_{}'.format(suffix),
               samples_including_built[:, 11:], delimiter=',')

    rdm_utilities_beta_opt = np.loadtxt(
        'TestFiles/rdm_utilities_{}'.format(suffix), delimiter=',')
    rdm_dmp_beta_opt = np.loadtxt('TestFiles/rdm_dmp_{}'.format(suffix),
                                  delimiter=',')
    rdm_water_sources_beta_opt = np.loadtxt(
        'TestFiles/rdm_water_sources_{}'.format(suffix), delimiter=',')

    fig, axes = plt.subplots(8, 7, sharex='all', sharey='all')
    axes_it = axes.reshape(-1)
    for i in range(48):
        axes_it[i].hist(rdm_water_sources_beta_opt[:, i], normed=True,
                        alpha=0.2, color=cmap(float(i) / nparams))
    plt.show()


def get_lows_highs_means_du_params():
    # 0 - Sinusoid amplitude
    # 1 - Sinusoid frequency
    # 2 - Sinusoid phase
    # 3 - Demand growth multiplier
    # 4 - Bond interest rate multiplier
    # 5 - Bond term multiplier
    # 6 - Discount rate multiplier
    # 7, 8, 9, 10 - Restriction stage effectiveness multiplier (one for
    #   each utility)
    # 11 - Evaporation rate multiplier
    # 12, 14, ..., 38 - Permitting time multiplier for new infrastructure
    # 13, 15, ..., 39 - Construction cost multiplier for new infrastructure

    lows = np.array(
        [0, 0, 0, 0.5, 1.0, 0.6, 0.6] + [0.9] * 4 + [0.8] + [0.75, 1.0] * 18)
    highs = np.array(
        [0, 0, 0, 2., 1.2, 1.0, 1.4] + [1.1] * 4 + [1.2] + [1.5, 1.2] * 18)
    means = np.array(
        [0, 0, 0, 1., 1.0, 1.0, 1.0] + [1.0] * 4 + [1.0] + [1.0, 1.0] * 18)

    return lows, highs, means


def generate_uniform_and_beta_samples(nsamples=2000, nparams=48,
                                      base_factor=1.5,
                                      cmap=cm.get_cmap('cool')):

    lows, highs, means = get_lows_highs_means_du_params()

    # sample_uniform(lows, highs, means, n_samples, n_params)
    sample_beta(lows, highs, means, nsamples, nparams, base_factor, cmap=cmap)
