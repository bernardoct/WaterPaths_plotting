import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm, patches

def plot_results(X, Y_, means, covariances, index, files_root_directory):
    # X = X[:, :2]
    # means = means[:, :2]
    print 'Plotting clusters, {}'.format(X.shape)
    fig, axes = plt.subplots(X.shape[1], X.shape[1], figsize=(50, 50))

    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, cm.get_cmap('Accent').colors)):

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        for dx in range(X.shape[1]):
            for dy in range(X.shape[1]):
                # as the DP will not use every component it has access to
                # unless it needs it, we shouldn't plot the redundant
                # components.
                if not np.any(Y_ == i):
                    continue
                axes[dx][dy].scatter(X[Y_ == i, dx], X[Y_ == i, dy], .8, color=color)

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[dy] / u[dx])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(mean, v[dx], v[dy], 180. + angle, color=color)
                ell.set_clip_box(axes[dx][dy].bbox)
                ell.set_alpha(0.5)
                axes[dx][dy].add_artist(ell)
                axes[dx][dy].set_xticks(())
                axes[dx][dy].set_yticks(())

    # plt.xlim(-9., 5.)
    # plt.ylim(-3., 6.)
    # plt.title(title)

    plt.savefig(files_root_directory + 'clusters.png')
    # plt.show()