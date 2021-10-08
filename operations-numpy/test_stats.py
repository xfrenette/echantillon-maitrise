import itertools
import unittest
from itertools import product
from typing import Optional, Tuple, Union, List, overload

import numpy as np
from scipy import stats
from scipy.spatial import distance

from couleurs.utils.stats import normals_kl_div, normals_js_dist


_ListOfNp = List[np.ndarray]


def _make_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    n_dim = matrix.shape[0]
    return np.dot(matrix, matrix.T) / (2 * n_dim ** 2)


@overload
def _generate_normal_params(nb_dim: int, nb_list1: int) -> Tuple[_ListOfNp, _ListOfNp]:
    pass


@overload
def _generate_normal_params(nb_dim: int, nb_list1: int, nb_list2: int) -> Tuple[_ListOfNp, _ListOfNp, _ListOfNp, _ListOfNp]:
    pass


def _generate_normal_params(
        nb_dim: int, nb_list1: int, nb_list2: Optional[int] = None
) -> Union[Tuple[_ListOfNp, _ListOfNp], Tuple[_ListOfNp, _ListOfNp, _ListOfNp, _ListOfNp]]:
    rng = np.random.default_rng(1984)

    # We test with lists of numpy arrays
    mu_list1 = [rng.uniform(-2 + i*2, -1 + i*2, nb_dim) for i in range(nb_list1)]
    sigma_list1 = [_make_positive_semidefinite(rng.uniform(-2 + i*3, 2 + i*3, (nb_dim, nb_dim))) for i in range(nb_list1)]

    if nb_list2 is not None:
        mu_list2 = [rng.uniform(5 + i*2, 6 + i*2, nb_dim) for i in range(nb_list2)]
        sigma_list2 = [_make_positive_semidefinite(rng.uniform(-3 + i*3, 3 + i*3, (nb_dim, nb_dim))) for i in range(nb_list2)]
        return mu_list1, sigma_list1, mu_list2, sigma_list2
    else:
        return mu_list1, sigma_list1


def _calculate_kl_divergence(m1: np.ndarray, m2: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Not efficient, but easy to follow procedure to calculate the KL divergence between 2 multivariate normals
    See: https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """
    d = m1.shape[0]
    det1 = np.linalg.det(s1)
    det2 = np.linalg.det(s2)
    s2_inv = np.linalg.inv(s2)
    mu_diff = m2 - m1

    log_term = np.log2(det2 / det1)
    trace_term = np.trace(s2_inv @ s1)
    quad_term = mu_diff @ s2_inv @ mu_diff

    return 0.5 * (log_term - d + trace_term + quad_term)


def _calculate_js_dist(m1: np.ndarray, m2: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    normal1 = stats.multivariate_normal(m1, s1)
    normal2 = stats.multivariate_normal(m2, s2)
    samples1 = normal1.rvs(50000)
    samples2 = normal2.rvs(50000)
    samples = np.vstack((samples1, samples2))
    probs1 = normal1.pdf(samples)
    probs2 = normal2.pdf(samples)
    return distance.jensenshannon(probs1, probs2, base=2)


class NormalsKLDivTestCase(unittest.TestCase):
    def test_one_set(self):
        rng = np.random.default_rng(2012)

        # Test with two identical distributions
        nb_dim = 3
        mu = rng.uniform(-1, 5, nb_dim)
        sigma = _make_positive_semidefinite(rng.uniform(-2, 2, (nb_dim, nb_dim)))
        mu_list = np.array([mu, mu])
        sigma_list = np.array([sigma, sigma])
        res = normals_kl_div(mu_list, sigma_list)
        self.assertTrue(np.allclose(res, 0))

        # Test with 2 different distributions
        mu1 = mu
        mu2 = rng.uniform(-1, 5, nb_dim)
        sigma1 = sigma
        sigma2 = _make_positive_semidefinite(rng.uniform(-2, 2, (nb_dim, nb_dim)))
        mu_list = np.array([mu1, mu2])
        sigma_list = np.array([sigma1, sigma2])
        res = normals_kl_div(mu_list, sigma_list)
        self.assertAlmostEqual(0, res[0, 0])
        self.assertAlmostEqual(0, res[1, 1])
        expected1 = _calculate_kl_divergence(mu1, mu2, sigma1, sigma2)
        expected2 = _calculate_kl_divergence(mu2, mu1, sigma2, sigma1)
        self.assertAlmostEqual(expected1, res[0, 1])
        self.assertAlmostEqual(expected2, res[1, 0])

    def test_two_sets(self):
        nb_dim = 4
        nb_list1 = 2
        nb_list2 = 3

        mu_list1, sigma_list1, mu_list2, sigma_list2 = _generate_normal_params(nb_dim, nb_list1, nb_list2)
        res = normals_kl_div(mu_list1, sigma_list1, mu_list2, sigma_list2)

        # Check correct size
        self.assertEqual((nb_list1, nb_list2), res.shape)

        # Check all values equal divergences calculated explicitly
        for i, j in product(range(nb_list1), range(nb_list2)):
            with self.subTest(i=i, j=j):
                mu1 = mu_list1[i]
                mu2 = mu_list2[j]
                sigma1 = sigma_list1[i]
                sigma2 = sigma_list2[j]

                expected = _calculate_kl_divergence(mu1, mu2, sigma1, sigma2)
                self.assertAlmostEqual(expected, res[i, j])


class NormalJSDistTestCase(unittest.TestCase):
    def test_one_set(self):
        nb_dim = 4
        nb_list1 = 2

        mu_list1, sigma_list1 = _generate_normal_params(nb_dim, nb_list1)
        res = normals_js_dist(mu_list1, sigma_list1)
        self.assertEqual((nb_list1, nb_list1), res.shape)
        self.assertEqual(0, res[0, 0])
        self.assertEqual(0, res[1, 1])

    def test_two_sets_vert(self):
        nb_dim = 4
        nb_list1 = 3
        nb_list2 = 2

        mu_list1, sigma_list1, mu_list2, sigma_list2 = _generate_normal_params(nb_dim, nb_list1, nb_list2)
        res = normals_js_dist(mu_list1, sigma_list1, mu_list2, sigma_list2, nb_samples=100000)
        self.assertEqual((nb_list1, nb_list2), res.shape)
        self.assertAlmostEqual(res[0, 1], res[1, 0], 4)

        # Check each element if they are equal
        for i, j in itertools.product(range(nb_list1), range(nb_list2)):
            mu1 = mu_list1[i]
            mu2 = mu_list2[j]
            sigma1 = sigma_list1[i]
            sigma2 = sigma_list2[j]
            with self.subTest(i=i, j=j):
                expected = _calculate_js_dist(mu1, mu2, sigma1, sigma2)
                actual = res[i, j]
                self.assertAlmostEqual(actual, expected, 4)


if __name__ == '__main__':
    unittest.main()
