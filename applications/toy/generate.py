import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# params
no_samples = 1000
means = np.array([[0,0], [5,5], [10,7]])
covariances = np.array([
    [[1, 0.5], [0.5, 1]],
    [[1, -0.3], [-0.3, 1]],
    [[0.5, 0], [0, 0.5]],
])
weights = [0.4, 0.35, 0.25]

# Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full"
)
gmm.weights_ = np.array(weights)
gmm.means_ = means
gmm.covariances_ = covariances
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))   # required to evaluate the PDF


# plot
if __name__ == "__main__":
    # generation of samples
    samples, labels = gmm.sample(no_samples)
    log_density = gmm.score_samples(samples)

    fig = plt.figure()
    plt.scatter(samples[:,0], samples[:,1])
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(samples[:,0], samples[:,1], log_density)
    plt.show()
