from sklearn import mixture
from plotting.clustering_plotting import plot_results


def gmm_cluster(dec_vars, files_root_directory):

    print 'Running clustering.'
    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(covariance_type='full').fit(dec_vars)

    plot_results(dec_vars, dpgmm.predict(dec_vars), dpgmm.means_,
                 dpgmm.covariances_, 1, files_root_directory)

