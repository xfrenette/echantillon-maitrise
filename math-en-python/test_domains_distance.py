import itertools
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List, Sequence, Dict
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
import torch
from random_words import LoremIpsum
from sklearn.covariance import oas

from couleurs.classifiers import GaussianClassifier
from couleurs.domains_distance import L2Distance, CosineDistance, MMD, FLD, CORALLoss, KLDiv, Tfidf
from couleurs.utils.stats import normals_kl_div
from couleurs.cli.utils.env import load_env


class HFTBasedDistanceTestCase(unittest.TestCase):
    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_uses_embeddings_cache(self, MockEmbedder):
        nb_embeddings = 25
        embeddings = torch.rand((nb_embeddings, 15))
        expected_representation = torch.mean(embeddings, dim=0)
        mock_embedder = MockEmbedder.return_value
        mock_embedder.embed.return_value = embeddings

        # Tests that the HFTEmbedder is not called again when processing the same file (that cache was used)
        with TemporaryDirectory() as cache_dir_name, \
                create_mock_domain_files(("DOM1",), ("DOM2",), nb_sentences=nb_embeddings) as (left_dir, _):
            cache_dir = Path(cache_dir_name)
            l2_distance = L2Distance(model="distilbert-base-cased", device="cpu", cache_dir=cache_dir, batch_size=nb_embeddings)

            actual = l2_distance.make_representation_from_file(left_dir / "DOM1.en")
            self.assertTrue(torch.equal(actual, expected_representation))
            mock_embedder.embed.assert_called_once()

            # We do it again, but this time, the HFTEmbedder.embed method should not have been called
            actual = l2_distance.make_representation_from_file(left_dir / "DOM1.en")
            self.assertTrue(torch.equal(actual, expected_representation))
            mock_embedder.embed.assert_called_once()


class DomainsDistanceTestCase(unittest.TestCase):
    def test_calculate_between_dirs(self):
        embed_dim = 10
        l2_distance = L2Distance()
        representations = {
            "DOM1": torch.rand(embed_dim),
            "DOM2": torch.rand(embed_dim),
            "DOM3": torch.rand(embed_dim),
            "DOM4": torch.rand(embed_dim),
            "DOM5": torch.rand(embed_dim),
        }
        domains = list(representations.keys())
        left_domains = domains[:2]
        right_domains = domains[2:]

        def mock_make_representation(sentences: Sequence[str], _):
            parts = sentences[0].split()
            domain = parts[0]
            return representations[domain]

        l2_distance.make_representation = Mock(side_effect=mock_make_representation)

        # Create the expected distance pandas DataFrame
        expected = pd.DataFrame(index=left_domains, columns=right_domains, dtype=float)
        for index, col in itertools.product(expected.index, expected.columns):
            expected.loc[index, col] = l2_distance.calculate(representations[index], representations[col])

        with create_mock_domain_files(left_domains, right_domains) as (left_dir, right_dir):
            # Also create files that should not be considered
            (left_dir / "DOMi.fr").touch()
            (right_dir / "DOMj.fr").touch()
            actual = l2_distance.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")
            pd.testing.assert_frame_equal(expected, actual, check_like=True)


def dress_mock_embedder(Embedder, nb_sentences, embed_dim=10) -> Tuple[torch.Tensor, List[str]]:
    lipsum = LoremIpsum()
    raw_sentences = lipsum.get_sentences_list(nb_sentences)
    sentences = [f"{i} {s}" for i, s in enumerate(raw_sentences)]
    embeddings = torch.rand((nb_sentences, embed_dim)) * 10 - 3

    def mock_embed(sentences: Sequence[str]):
        results = torch.empty((len(sentences), embed_dim))
        for i, sentence in enumerate(sentences):
            parts = sentence.split()
            index = int(parts[0])
            results[i, :] = embeddings[index]
        return results

    mock_embedder = Embedder.return_value
    mock_embedder.embed.side_effect = mock_embed

    return embeddings, sentences


@contextmanager
def create_mock_domain_files(left_domains: Sequence[str], right_domains: Sequence[str], suffix=".en", nb_sentences=20):
    lipsum = LoremIpsum()
    with TemporaryDirectory() as tmp_dir_name:
        left_dir = Path(tmp_dir_name) / "left"
        right_dir = Path(tmp_dir_name) / "right"

        left_dir.mkdir()
        right_dir.mkdir()

        for left_domain in left_domains:
            file_path = left_dir / f"{left_domain}{suffix}"
            with open(file_path, "w") as f:
                sentences = [f"{left_domain} {s}" for s in lipsum.get_sentences_list(nb_sentences)]
                f.write("\n".join(sentences) + "\n")

        for right_domain in right_domains:
            file_path = right_dir / f"{right_domain}.en"
            with open(file_path, "w") as f:
                sentences = [f"{right_domain} {s}" for s in lipsum.get_sentences_list(nb_sentences)]
                f.write("\n".join(sentences) + "\n")

        yield left_dir, right_dir


class L2DistanceTestCase(unittest.TestCase):
    def test_calculate(self):
        """
        calculate() should be the l2 norm between 2 vectors
        """
        v1 = torch.tensor([1, 2, 3])
        v2 = torch.tensor([4, 5, 6])
        diff = v2 - v1
        expected = np.sqrt(np.dot(diff, diff))

        distance = L2Distance()
        actual = distance.calculate(v1, v2)
        self.assertEqual(expected, actual)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_make_representation(self, MockEmbedder):
        embeddings, sentences = dress_mock_embedder(MockEmbedder, 20)
        expected = torch.mean(embeddings, dim=0)
        distance_calculator = L2Distance(batch_size=9)
        actual = distance_calculator.make_representation(sentences)
        self.assertTrue(torch.equal(expected, actual))

    def test_functional(self):
        # Makes sure the whole process works without problems
        left_domains = ("DOM1", "DOM2", "DOM3")
        right_domains = ("DOM4", "DOM5")

        with TemporaryDirectory() as cache_dir_name, \
                create_mock_domain_files(left_domains, right_domains, nb_sentences=100) as (left_dir, right_dir):
            cache_dir = Path(cache_dir_name)
            l2_distance = L2Distance(model="distilbert-base-cased", device="cpu", cache_dir=cache_dir, max_sentences=20)
            result = l2_distance.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")

            cache_files = list(cache_dir.glob("*.pkl"))
            self.assertEqual(1, len(cache_files))
            self.assertIsInstance(result, pd.DataFrame)
            self.assertSetEqual(set(left_domains), set(result.index))
            self.assertSetEqual(set(right_domains), set(result.columns))


class CosineDistanceTestCase(unittest.TestCase):
    def test_calculate(self):
        """
        calculate() should be 1 - cosinus distance norm between 2 vectors
        """
        v1 = torch.tensor([1., 2., -3.])
        v2 = torch.tensor([-4., 5., 6.])
        v1_norm = torch.linalg.norm(v1, 2)
        v2_norm = torch.linalg.norm(v2, 2)
        expected = 1 - torch.dot(v1, v2) / (v1_norm * v2_norm)

        distance = CosineDistance()
        actual = distance.calculate(v1, v2)
        self.assertAlmostEqual(expected.item(), actual, 6)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_make_representation(self, MockEmbedder):
        """
        Representation should be the mean of the sentences embedding
        """
        embeddings, sentences = dress_mock_embedder(MockEmbedder, 20)
        expected = torch.mean(embeddings, dim=0)
        distance_calculator = CosineDistance(batch_size=9)
        actual = distance_calculator.make_representation(sentences)
        self.assertTrue(torch.equal(expected, actual))

    def test_functional(self):
        """
        Makes sure the whole process works without problems
        """
        left_domains = ("DOM1", "DOM2", "DOM3")
        right_domains = ("DOM4", "DOM5")

        with TemporaryDirectory() as cache_dir_name, \
                create_mock_domain_files(left_domains, right_domains, nb_sentences=100) as (left_dir, right_dir):
            cache_dir = Path(cache_dir_name)
            cosine_distance = CosineDistance(model="distilbert-base-cased", device="cpu", cache_dir=cache_dir, max_sentences=20)
            result = cosine_distance.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")

            cache_files = list(cache_dir.glob("*.pkl"))
            self.assertEqual(1, len(cache_files))
            self.assertIsInstance(result, pd.DataFrame)
            self.assertSetEqual(set(left_domains), set(result.index))
            self.assertSetEqual(set(right_domains), set(result.columns))


class MMDTestCase(unittest.TestCase):
    def test_suggest_rbf_gamma(self):
        left = torch.rand((10, 5))
        right = torch.rand((7, 5))
        all_distances = pairwise.euclidean_distances(np.vstack((left, right)))
        distances = all_distances[np.tril_indices(all_distances.shape[0], k=-1)]
        median_distance = np.percentile(distances, 50)
        expected = 1 / (2 * median_distance**2)

        dist = MMD()
        actual = dist.suggest_rbf_gamma(left, right)
        self.assertAlmostEqual(expected, actual, 6)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_make_representation(self, MockEmbedder):
        """
        Representation should simply be the embedding without any changes
        """
        embeddings, sentences = dress_mock_embedder(MockEmbedder, 20)
        expected = embeddings
        distance_calculator = MMD(batch_size=9)
        actual = distance_calculator.make_representation(sentences)
        self.assertTrue(torch.equal(expected, actual))

    def test_functional(self):
        """
        Makes sure the whole process works without problems
        """
        left_domains = ("DOM1", "DOM2", "DOM3")
        right_domains = ("DOM4", "DOM5")

        with TemporaryDirectory() as cache_dir_name, \
                create_mock_domain_files(left_domains, right_domains, nb_sentences=100) as (left_dir, right_dir):
            cache_dir = Path(cache_dir_name)
            mmd = MMD(model="distilbert-base-cased", device="cpu", cache_dir=cache_dir, max_sentences=20)
            result = mmd.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertSetEqual(set(left_domains), set(result.index))
            self.assertSetEqual(set(right_domains), set(result.columns))

    def test_rbf_gamma_large_sets(self):
        left = torch.rand((7000, 768))
        right = torch.rand((7000, 768))
        mmd = MMD()
        _ = mmd.suggest_rbf_gamma(left, right)
        # Should not raise
        self.assertTrue(True)


class FLDTestCase(unittest.TestCase):
    def test_calculate(self):
        """
        It's hard to test that the calculation are correct without simply reusing the code we try to test! So we will
        generate different pairs, knowing which pairs are farther/nearer together, and we will test that resulting
        distance have the correct order
        """
        dist = FLD()
        rng = np.random.default_rng()

        cov = np.eye(4)
        left = torch.from_numpy(rng.multivariate_normal([1, 1, 1, 1], cov, 100))
        # nearest
        right1 = torch.from_numpy(rng.multivariate_normal([0, 1, 1, 2], cov, 100))
        # middle
        right2 = torch.from_numpy(rng.multivariate_normal([-2, 1, 1, 4], cov * 2, 100))
        # farthest
        right3 = torch.from_numpy(rng.multivariate_normal([8, 5, 2, 7], cov * 3, 100))

        actual1 = dist.calculate(left, right1)
        actual2 = dist.calculate(left, right2)
        actual3 = dist.calculate(left, right3)

        self.assertTrue(actual1 < actual2 < actual3)

    def test_calculate_large_sets(self):
        """
        Test we don't have any error when working with large sets
        """
        dist = FLD()
        left = torch.rand((7000, 768))
        right = torch.rand((7000, 768)) + 4
        dist.calculate(left, right)
        # Should not fail
        self.assertTrue(True)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_make_representation(self, MockEmbedder):
        """
        Representation should simply be the embedding without any changes
        """
        embeddings, sentences = dress_mock_embedder(MockEmbedder, 20)
        expected = embeddings
        distance_calculator = FLD(batch_size=9)
        actual = distance_calculator.make_representation(sentences)
        self.assertTrue(torch.equal(expected, actual))

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_functional(self, MockEmbedder):
        """
        Makes sure the whole process works without problems
        Though we still have to mock the HFTEmbedder since we would need at least 800 sentences per file and testing
        would take too long
        """
        left_domains = ("DOM1", "DOM2", "DOM3")
        right_domains = ("DOM4", "DOM5")

        def mock_embed(sentences: Sequence[str]) -> torch.Tensor:
            return torch.rand([len(sentences), 768])

        mock_embedder = MockEmbedder.return_value
        mock_embedder.embed.side_effect = mock_embed

        with TemporaryDirectory() as cache_dir_name, \
                create_mock_domain_files(left_domains, right_domains, nb_sentences=1400) as (left_dir, right_dir):
            cache_dir = Path(cache_dir_name)
            dist = FLD(model="distilbert-base-cased", device="cpu", cache_dir=cache_dir)
            result = dist.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertSetEqual(set(left_domains), set(result.index))
            self.assertSetEqual(set(right_domains), set(result.columns))


class CORALTestCase(unittest.TestCase):
    @staticmethod
    def _calculate_coral_loss(left: torch.Tensor, right: torch.Tensor) -> float:
        fact = 1 / (4 * left.shape[1]**2)
        cov_left = np.cov(left, rowvar=False, ddof=1)
        cov_right = np.cov(right, rowvar=False, ddof=1)
        norm = np.linalg.norm(cov_left - cov_right, ord="fro")
        return fact * norm**2

    def test_calculate(self):
        dist = CORALLoss()
        n_dim = 103
        rng = np.random.default_rng()
        cov_left = np.diag(rng.random(n_dim)) * 5
        cov_right = np.diag(rng.random(n_dim)) * 5
        left = torch.from_numpy(rng.multivariate_normal(rng.random(n_dim) * 4 - 5, cov_left, 1000))
        right = torch.from_numpy(rng.multivariate_normal(rng.random(n_dim) * 8 - 2, cov_right, 985))

        expected = self._calculate_coral_loss(left, right)
        actual = dist.calculate(left, right)
        self.assertAlmostEqual(expected, actual)

    def test_calculate_large_sets(self):
        """
        Test we don't have any error when working with large sets
        """
        dist = CORALLoss()
        left = torch.rand((7000, 768))
        right = torch.rand((7000, 768)) + 4
        dist.calculate(left, right)
        # Should not fail
        self.assertTrue(True)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_make_representation(self, MockEmbedder):
        """
        Representation should simply be the embedding without any changes
        """
        embeddings, sentences = dress_mock_embedder(MockEmbedder, 20)
        expected = embeddings
        distance_calculator = CORALLoss(batch_size=9)
        actual = distance_calculator.make_representation(sentences)
        self.assertTrue(torch.equal(expected, actual))

    def test_functional(self):
        """
        Makes sure the whole process works without problems
        """
        left_domains = ("DOM1", "DOM2", "DOM3")
        right_domains = ("DOM4", "DOM5")

        with TemporaryDirectory() as cache_dir_name, \
                create_mock_domain_files(left_domains, right_domains, nb_sentences=100) as (left_dir, right_dir):
            cache_dir = Path(cache_dir_name)
            dist = CORALLoss(model="distilbert-base-cased", device="cpu", cache_dir=cache_dir, max_sentences=20)
            result = dist.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertSetEqual(set(left_domains), set(result.index))
            self.assertSetEqual(set(right_domains), set(result.columns))


class KLDivTestCase(unittest.TestCase):
    @staticmethod
    def _fit_classifier(embeddings: Dict[str, torch.Tensor]):
        all_embeddings = None
        classes = []
        for cls, embd in embeddings.items():
            if all_embeddings is None:
                all_embeddings = embd.numpy()
            else:
                all_embeddings = np.vstack((all_embeddings, embd))
            classes.extend([cls] * len(embd))

        classifier = GaussianClassifier(cov_method="oas")
        classifier.fit(all_embeddings, classes)
        return classifier

    def _calculate_kl_div(self, classifier: GaussianClassifier, left_domains: Sequence[str], right_domains: Sequence[str]):
        mu1 = [classifier.mu_[i] for i, d in enumerate(classifier.classes_) if d in left_domains]
        sigma1 = [classifier.sigma_[i] for i, d in enumerate(classifier.classes_) if d in left_domains]
        mu2 = [classifier.mu_[i] for i, d in enumerate(classifier.classes_) if d in right_domains]
        sigma2 = [classifier.sigma_[i] for i, d in enumerate(classifier.classes_) if d in right_domains]
        return normals_kl_div(mu1, sigma1, mu2, sigma2)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_make_representation(self, MockEmbedder):
        embeddings, sentences = dress_mock_embedder(MockEmbedder, 15, 301)
        embeddings = embeddings.numpy()
        expected_mean = np.mean(embeddings, 0)
        expected_cov, _ = oas(embeddings)
        dist = KLDiv()
        actual_mean, actual_cov = dist.make_representation(sentences)
        np.testing.assert_almost_equal(expected_mean, actual_mean)
        np.testing.assert_almost_equal(expected_cov, actual_cov)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_calculate_between_domains(self, MockEmbedder):
        embeddings = {
            'DOM1': torch.rand((200, 58)),
            'DOM2': torch.rand((200, 58)) + 10,
            'DOM3': torch.rand((200, 58)) + 20,
            'DOM4': torch.rand((200, 58)) + 30,
            'DOM5': torch.rand((200, 58)) + 40,
        }
        left_domains = ("DOM1", "DOM2")
        right_domains = ("DOM3", "DOM4", "DOM5")
        classifier = self._fit_classifier(embeddings)
        kl_div = self._calculate_kl_div(classifier, left_domains, right_domains)
        expected = pd.DataFrame(kl_div, index=left_domains, columns=right_domains, dtype=np.float)

        def mock_embed(sentences: Sequence[str]) -> torch.Tensor:
            domain = sentences[0]
            return embeddings[domain]

        mock_embedder = MockEmbedder.return_value
        mock_embedder.embed.side_effect = mock_embed

        dist = KLDiv()
        representations = {d: dist.make_representation([d]) for d in embeddings.keys()}
        actual = dist.calculate_between_domains(representations, left_domains, right_domains)

        pd.testing.assert_frame_equal(expected, actual, check_like=True)

    @patch("couleurs.domains_distance.HFTEmbedder", autospec=True)
    def test_functional(self, MockEmbedder):
        """
        Makes sure the whole process works without problems
        Though we still have to mock the HFTEmbedder since we would need at least 800 sentences per file and testing
        would take too long
        """
        left_domains = ("DOM1", "DOM2", "DOM3")
        right_domains = ("DOM4", "DOM5")

        def mock_embed(sentences: Sequence[str]) -> torch.Tensor:
            return torch.rand([len(sentences), 768])

        mock_embedder = MockEmbedder.return_value
        mock_embedder.embed.side_effect = mock_embed

        with TemporaryDirectory() as cache_dir_name, \
                create_mock_domain_files(left_domains, right_domains, nb_sentences=100) as (left_dir, right_dir):
            cache_dir = Path(cache_dir_name)
            dist = KLDiv(model="distilbert-base-cased", device="cpu", cache_dir=cache_dir, max_sentences=20)
            result = dist.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertSetEqual(set(left_domains), set(result.index))
            self.assertSetEqual(set(right_domains), set(result.columns))


class TfidfTestCase(unittest.TestCase):
    def setUp(self):
        load_env()

    def test_functional(self):
        """
        Makes sure the whole process works without problems
        """
        left_domains = ("DOM1", "DOM2", "DOM3")
        right_domains = ("DOM4", "DOM5")

        with create_mock_domain_files(left_domains, right_domains, nb_sentences=100) as (left_dir, right_dir):
            dist = Tfidf()
            result = dist.calculate_between_dirs(left_dir, right_dir, glob_left="*.en", glob_right="*.en")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertSetEqual(set(left_domains), set(result.index))
            self.assertSetEqual(set(right_domains), set(result.columns))


if __name__ == '__main__':
    unittest.main()
