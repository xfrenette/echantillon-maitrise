"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2020-11-20
"""
from itertools import product
import math
from typing import Optional, Union, Sequence, Tuple
from scipy import stats
from scipy.spatial import distance
from scipy import special

import numpy as np

_ListOfNumpyArrays = Union[Sequence[np.ndarray], np.ndarray]


def _process_normal_params(mu1: _ListOfNumpyArrays, sigma1: _ListOfNumpyArrays, mu2: Optional[_ListOfNumpyArrays] = None,
                           sigma2: Optional[_ListOfNumpyArrays] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mu2 is None:
        mu2 = mu1

    if sigma2 is None:
        sigma2 = sigma1

    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    sigma1, sigma2 = np.asarray(sigma1), np.asarray(sigma2)

    if mu1.shape[1] != mu2.shape[1]:
        raise ValueError(f"All distributions must have the same dimension (mean dimensions: {mu1.shape[1]} ≠ {mu2.shape[2]})")
    if sigma1.shape[1:] != sigma2.shape[1:]:
        raise ValueError(f"All distributions must have the same dimension (covariance dimensions: {sigma1.shape[1:]} ≠ {sigma2.shape[1:]})")

    return mu1, sigma1, mu2, sigma2


def normals_kl_div(mu1: _ListOfNumpyArrays, sigma1: _ListOfNumpyArrays, mu2: Optional[_ListOfNumpyArrays] = None,
                   sigma2: Optional[_ListOfNumpyArrays] = None) -> np.ndarray:
    """
    Calculates the KL divergence between every pair of 2 lists of multivariate normals (parametrized by their mean
    vector and their covariance matrix). Will return a 2D array where element i,j is the KL divergence between
    distribution i of first list with the distribution j of the second list. The KL divergence is not symmetric,
    so element i,j ≠ element j,i.

    If the second distribution is not defined (mu2 and sigma2 are None), the divergences will be between every pair of
    the first list (element i,j will be the divergence between distribution i and distribution j of the first list, so
    elements i,i will always be 0). All distributions must have the same dimensions.

    :param mu1: Means of the first list of distributions. Array-like list of (n_dimensions,) arrays
    :param sigma1: Covariances of the first list of distributions. Array-like list of (n_dimensions, n_dimensions) arrays
    :param mu2: Means of the second list of distributions. Array-like list of (n_dimensions,) arrays
    :param sigma2: Covariances of the second list of distributions. Array-like list of (n_dimensions, n_dimensions) arrays
    :return: Array of KL divergences of shape (n_distributions, m_distributions).
    """
    # Analytical solution to KL divergence between multivariate normals:
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    # The code uses numpy's capabilities of operations on multiple matrices at the same time, so all the
    # n_classes x n_classes divergences are calculated in one go (no for loops).

    only_sigma1 = sigma2 is None
    mu1, sigma1, mu2, sigma2 = _process_normal_params(mu1, sigma1, mu2, sigma2)
    d = mu1.shape[1]
    sigma2_inv = np.linalg.inv(sigma2)

    # Because of the sizes of matrices, the log determinant is more robust to over/underflow
    _, log_det1 = np.linalg.slogdet(sigma1)
    if only_sigma1:
        log_det2 = log_det1
    else:
        _, log_det2 = np.linalg.slogdet(sigma2)
    # No more log(...) since we use the log determinants. And `* np.log2(np.e)` because log_det is natural log, but we calculate in log2
    log_term = (np.expand_dims(log_det2, 0) - np.expand_dims(log_det1, 1)) * np.log2(np.e)

    trace_term = np.trace(np.expand_dims(sigma2_inv, 0) @ np.expand_dims(sigma1, 1), axis1=-2, axis2=-1)
    mu_diff = np.expand_dims(mu2, 0) - np.expand_dims(mu1, 1)
    lin_term = np.einsum('...i,...ij->...j', mu_diff, sigma2_inv)
    quad_term = np.einsum('...i,...i->...', lin_term, mu_diff)

    return 0.5 * (log_term + trace_term + quad_term - d)


def normals_js_dist(mu1: _ListOfNumpyArrays, sigma1: _ListOfNumpyArrays, mu2: Optional[_ListOfNumpyArrays] = None,
                    sigma2: Optional[_ListOfNumpyArrays] = None, nb_samples: int = 5000,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Calculates the Jensen-Shannon distance between every pair of normal distributions defined by their mean and covariance.

    The Jensen-Shannon distance measures similarity between two distributions. It's the square-root of the
    Jensen-Shannon divergence. This methods calculates the distance (not the divergence) between all pairs of two lists
    of normal distributions: mu1 and sigma1 are lists of means and variances, same with mu2 and sigma2. The returned
    value will be a 2D array of size (number of elements in list 1) x (number of elements in list 2). The divergence is
    symmetric, so element i,j = element j,i (where number of distribution permits).

    If the second list (mu2 and sigma2) is not defined, the pairs will be between elements of the first list (like if
    mu2 = mu1 and sigma2 = sigma1). In this case, elements i,i will be 0.

    The method works by generating samples from both distributions in each pair. `nb_samples` specifies the total
    number of samples to generate.

    A random generator can be passed to reproduce exact results.

    The logarithm used for the distance is base 2 (distance is in √bits).

    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    :param mu1: Means of the first list of distributions. Array-like list of (n_dimensions,) arrays
    :param sigma1: Covariances of the first list of distributions. Array-like list of (n_dimensions, n_dimensions) arrays
    :param mu2: Means of the second list of distributions. Array-like list of (n_dimensions,) arrays
    :param sigma2: Covariances of the second list of distributions. Array-like list of (n_dimensions, n_dimensions) arrays
    :param nb_samples: Number of samples to generate total (half in first distribution, half in second)
    :param rng: Random number generator
    :return: 2D array of distances of shape (number of elements in list 1) x (number of elements in list 2)
    """
    only_one_list = mu2 is None
    mu1, sigma1, mu2, sigma2 = _process_normal_params(mu1, sigma1, mu2, sigma2)
    distances = np.empty((mu1.shape[0], mu2.shape[0]))

    for row, col in product(range(mu1.shape[0]), range(mu2.shape[0])):
        # If only one list, elements i,i is distance between distribution i and itself, so 0
        if only_one_list and row == col:
            distances[row, col] = 0.0
            continue

        # Since distance is symmetric, we don't recalculate values already calculated
        if mu2.shape[0] > row > col:
            distances[row, col] = distances[col, row]
            continue

        normal1 = stats.multivariate_normal(mu1[row], sigma1[row])
        normal2 = stats.multivariate_normal(mu2[col], sigma2[col])
        samples1 = normal1.rvs(math.floor(nb_samples / 2), random_state=rng)
        samples2 = normal2.rvs(math.ceil(nb_samples / 2), random_state=rng)
        samples = np.vstack((samples1, samples2))
        probs1 = special.softmax(normal1.logpdf(samples))
        probs2 = special.softmax(normal2.logpdf(samples))
        distances[row, col] = distance.jensenshannon(probs1, probs2, base=2)

    return distances
