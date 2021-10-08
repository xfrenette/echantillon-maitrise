"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-01-04
Distance measures are in part inspired by the review found in :
[1] Guo et al. (2020), Multi-Source Domain Adaptation for Text Classification via DistanceNet-Bandits
"""
import itertools
import logging
import math
import pickle
import string
from pathlib import Path
from typing import Sequence, Dict, Optional, Union

import nltk
import numpy as np
import pandas as pd
import scipy.spatial
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from couleurs.classifiers import GaussianClassifier
from couleurs.sentence_embedder.embedders import HFTEmbedder
from couleurs.utils import NULL_LOGGER, chunks
from couleurs.utils.stats import normals_kl_div

_DomainRepresentation = Union[Sequence[torch.Tensor], torch.Tensor]
_RepresentationsDict = Dict[str, _DomainRepresentation]
STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "becau"]


class DomainsDistance:
    """
    Abstract class for all domains distance classes. Note that distance classes must implement a symmetric distance
    (i.e. dist(A, B) == dist(B, A))
    """

    def __init__(self, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        :param cache_dir: If set, directory where to save cache domain representations
        :param logger: If set, logger where to log messages
        """
        self.cache_dir = cache_dir
        self._cache: Optional[_RepresentationsDict] = None
        self.logger = logger or NULL_LOGGER
        self.name = "__"
        self.symmetric = True

    def get_cache_file_path(self) -> Path:
        """
        File path where already calculated domain representations are cached
        :return: Cache file path
        """
        raise NotImplemented()

    def load_cache(self) -> _RepresentationsDict:
        """
        Loads cached domain representations from the cache
        :return: A dictionary where the key is the domain, the value is its representation
        """
        if self.cache_dir is None:
            return {}

        if self._cache is None:
            path = self.get_cache_file_path()
            if path.exists():
                with open(path, "rb") as file:
                    self._cache = pickle.load(file)
            else:
                self._cache = {}
        return self._cache

    def save_cache(self, data: _RepresentationsDict):
        """
        Saves domain representations to cache
        :param data: A dictionary where the key is the domain, the value is its representation
        """
        if self.cache_dir is None:
            return

        path = self.get_cache_file_path()
        with open(path, "wb") as file:
            pickle.dump(data, file)

    def make_representation_from_file(self, file_path: Path) -> _DomainRepresentation:
        """
        Returns a domain representation from sentences in the `file_path`
        :param file_path: File path to a file containing one sentence per line
        :return: Domain representation for this file
        """
        with open(file_path, "r") as file:
            return self.make_representation(file.readlines())

    def make_representation(self, sentences: Sequence[str]) -> _DomainRepresentation:
        """
        Calculates and returns a domain representation for a domain represented by a set of sentences
        :param sentences: All sentences of the domain
        :return: Domain representation for this set of sentences
        """
        raise NotImplemented()

    def calculate_between_dirs(self, dir_left: Path, dir_right: Optional[Path] = None, glob_left="*", glob_right="*") -> pd.DataFrame:
        """
        Calculates and returns distances between domains represented by text files. Will make representations for each
        file in `dir_left` (matching `glob_left` pattern) and will calculate the distance with each file in `dir_right`
        (matching `glob_right` pattern). If `dir_right` is None, it will be set to `dir_left` (so every file will be
        compared with all other files).
        :param dir_left: Directory containing one file per domain
        :param dir_right: Directory containing one file per domain
        :param glob_left: Pattern for files in `dir_left`
        :param glob_right: Pattern for files in `dir_right`
        :return: A pandas DataFrame where rows are domains (files) in `dir_left` and cols are from `dir_right`
        """
        if dir_right is None:
            dir_right = dir_left
        left_file_paths = dir_left.glob(glob_left)
        right_file_paths = dir_right.glob(glob_right)
        left_domains = []
        right_domains = []

        representations = self.load_cache()

        def process_file(file_path, domains_list):
            nonlocal representations
            domain_name = file_path.stem
            domains_list.append(domain_name)
            if domain_name not in representations:
                self.logger.info(f"Calculating {self.name} representation for {domain_name}")
                representation = self.make_representation_from_file(file_path)
                representations[domain_name] = representation
                self.save_cache(representations)
            else:
                self.logger.info(f"Retrieving {self.name} representation for {domain_name} from cache (already calculated)")

        for file_path in left_file_paths:
            process_file(file_path, left_domains)

        for file_path in right_file_paths:
            process_file(file_path, right_domains)

        return self.calculate_between_domains(representations, left_domains, right_domains)

    def calculate_between_domains(self, representations: _RepresentationsDict, left_domains: Sequence[str], right_domains: Sequence[str]) -> pd.DataFrame:

        distances = pd.DataFrame(columns=right_domains, index=left_domains, dtype=np.float)

        for left_domain, right_domain in itertools.product(left_domains, right_domains):
            # If symmetric and distance was already calculated for this pair (distances are symmetric)
            if self.symmetric and right_domain in left_domains and left_domain in right_domains and not pd.isna(
                    distances.loc[right_domain, left_domain]):
                distances.loc[left_domain, right_domain] = distances.loc[right_domain, left_domain]
                continue

            self.logger.info(f"Calculating {self.name} distance between {left_domain} and {right_domain}")
            distances.loc[left_domain, right_domain] = self.calculate(representations[left_domain], representations[right_domain])

        return distances

    def calculate(self, left: _DomainRepresentation, right: _DomainRepresentation) -> float:
        """
        Calculates and returns the distance between two domain representations
        :param left: Left domain representation
        :param right: Right domain representation
        :return: Distance. Positive value. The greater, the farthest. 0 means the same domain
        """
        raise NotImplemented()


class HFTBasedDistance(DomainsDistance):
    """
    Mixin for all classes that depend on HuggingFace Transformer to embed sentences
    """

    def __init__(self, batch_size=64, model='roberta-large', max_sentences: Optional[int] = None, device='cuda',
                 seed: Optional[int] = None, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        super().__init__(cache_dir, logger)
        self._embedder = HFTEmbedder(model, device=device, logger=self.logger)
        self.max_sentences = max_sentences
        self._embeddings_cache_dir = None
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

        if cache_dir:
            self._embeddings_cache_dir = cache_dir / f"embeddings_{model.replace('/', '__')}"
            self._embeddings_cache_dir.mkdir(exist_ok=True)

        self.loaded = False

    def load(self):
        """
        Loads the embedder model
        """
        if not self.loaded:
            self._embedder.load()
            self.loaded = True

    def _select_sentence_set(self, sentences: Sequence[str]) -> Sequence[str]:
        """
        Reduces the number of sentences to `self.max_sentences` by randomly selecting samples in the `sentences` list.
        If the list is already smaller than `self.max_sentences`, all sentences are returned
        :param sentences: All sentences to select from
        :return: Selected sub-set
        """
        if self.max_sentences is None or len(sentences) <= self.max_sentences:
            return sentences

        indexes = self._rng.choice(len(sentences), self.max_sentences, replace=False)
        return [sentences[i] for i in indexes]

    def _load_embeddings(self, domain: str) -> Union[None, torch.Tensor]:
        """
        Retrieves cached embeddings for this domain. If no embeddings were cached, returns None
        :param domain: Domain name
        :return: None if no cache, else torch.Tensor of size (nb_embeddings, embed_dim)
        """
        if self._embeddings_cache_dir is None:
            return None

        cache_path = self._embeddings_cache_dir / f"{domain}.pkl"
        if not cache_path.exists():
            return None

        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def _cache_embeddings(self, domain: str, embeddings: torch.Tensor):
        """
        Caches embeddings calculated for a domain
        :param domain: Name of the domain the embeddings were calculated for
        :param embeddings: The calculated embeddings, torch.Tensor of size (nb_embeddings, embed_dim)
        """
        if self._embeddings_cache_dir is None:
            return

        with open(self._embeddings_cache_dir / f"{domain}.pkl", "wb") as f:
            return pickle.dump(embeddings, f)

    def make_representation(self, sentences: Sequence[str], embeddings_cache_name: Optional[str] = None) -> _DomainRepresentation:
        """
        Uses the HFTEmbedder to make embeddings of each sentences and uses their mean as representation
        :param sentences: Sentences to make representation of
        :param embeddings_cache_name: If set, embeddings will be cached under this name
        :return: torch.Tensor of size (embed_dim,)
        """
        self.load()
        sentences = self._select_sentence_set(sentences)
        embeddings: Optional[_DomainRepresentation] = None
        nb_chunks = math.ceil(len(sentences) / self.batch_size)
        for sentence_chunk in tqdm(chunks(self.batch_size, sentences), total=nb_chunks):
            chunk_embeddings = self._embedder.embed(sentence_chunk)
            if embeddings is None:
                embeddings = chunk_embeddings
            else:
                embeddings = torch.cat((embeddings, chunk_embeddings))

        if embeddings_cache_name is not None:
            self._cache_embeddings(embeddings_cache_name, embeddings)

        return self._construct_representation(embeddings)

    def make_representation_from_file(self, file_path: Path) -> _DomainRepresentation:
        """
        Returns a domain representation from sentences in the `file_path`. Uses a cache for the embeddings.
        :param file_path: File path to a file containing one sentence per line
        :return: Domain representation for this file
        """
        stem = file_path.stem
        cached_embeddings = self._load_embeddings(stem)

        if cached_embeddings is not None:
            return self._construct_representation(cached_embeddings)

        with open(file_path, "r") as file:
            return self.make_representation(file.readlines(), stem)

    def _construct_representation(self, embeddings: torch.Tensor) -> _DomainRepresentation:
        raise NotImplemented()


class L2Distance(HFTBasedDistance):
    """
    Defines distance between domains as the euclidean distance between the mean of their sentences embedding
    """

    def __init__(self, batch_size=64, model='roberta-large', max_sentences: Optional[int] = None, device='cuda',
                 seed: Optional[int] = None, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        super().__init__(batch_size, model, max_sentences, device, seed, cache_dir, logger)
        self.name = "L2"

    def _construct_representation(self, embeddings: torch.Tensor) -> _DomainRepresentation:
        """
        The representation of a domain is the mean of its sentences embedding
        :param embeddings: Embeddings of sentence. Tensor of size (nb_embeddings, embed_dim)
        :return: Tensor of size (embed_dim,)
        """
        return torch.mean(embeddings, dim=0)

    def get_cache_file_path(self) -> Path:
        return self.cache_dir / "l2.pkl"

    def calculate(self, left: _DomainRepresentation, right: _DomainRepresentation) -> float:
        """
        Calculates distance: euclidean distance between mean
        :param left: Left domain mean. Tensor of size (embed_dim,)
        :param right: Right domain mean. Tensor of size (embed_dim,)
        :return: The distance
        """
        return np.linalg.norm(right - left)


class CosineDistance(HFTBasedDistance):
    """
    Defines distance between two domains as the cosine distance between their mean.
    """

    def __init__(self, batch_size=64, model='roberta-large', max_sentences: Optional[int] = None, device='cuda',
                 seed: Optional[int] = None, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        super().__init__(batch_size, model, max_sentences, device, seed, cache_dir, logger)
        self.name = "cosinus"

    def _construct_representation(self, embeddings: torch.Tensor) -> _DomainRepresentation:
        """
        The representation of a domain is the mean of its sentences embedding
        :param embeddings: Embeddings of sentence. Tensor of size (nb_embeddings, embed_dim)
        :return: Tensor of size (embed_dim,)
        """
        return torch.mean(embeddings, dim=0)

    def get_cache_file_path(self) -> Path:
        return self.cache_dir / "cosinus.pkl"

    def calculate(self, left: _DomainRepresentation, right: _DomainRepresentation) -> float:
        """
        Calculates cosine distance between the means.
        distance = 1 - dot(left, right)/(norm(left) * norm(right))

        :param left: Left domain mean. Tensor of size (embed_dim,)
        :param right: Right domain mean. Tensor of size (embed_dim,)
        :return: The distance
        """
        return scipy.spatial.distance.cosine(left, right)


class MMD(HFTBasedDistance):
    """
    Similarity based on Maximum Mean Discrepancy. See [1].
    """

    def __init__(self, batch_size=64, model='roberta-large', max_sentences: Optional[int] = None, device='cuda',
                 seed: Optional[int] = None, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        super().__init__(batch_size, model, max_sentences, device, seed, cache_dir, logger)
        self.name = "MMD"

    def save_cache(self, data: _RepresentationsDict):
        """
        For MMD, no need to cache representation, since its the same as domain embeddings which is already cached
        """
        return

    def _construct_representation(self, embeddings: torch.Tensor) -> _DomainRepresentation:
        """
        The representation of a domain is all the embeddings of its sentences. So this function simply returns the
        embeddings
        :param embeddings: Embeddings of sentence. Tensor of size (nb_embeddings, embed_dim)
        :return: Tensor of size (nb_embeddings, embed_dim)
        """
        return embeddings

    def get_cache_file_path(self) -> Path:
        return self.cache_dir / "mmd.pkl"

    @staticmethod
    def mmd2_rbf(left: torch.Tensor, right: torch.Tensor, gamma=None) -> float:
        """
        Unbiased MMD^2 estimation between two sample sets using rbf (gaussian) kernel (i.e.,
        k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Based on (but they don't use the unbiased equation):
        https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

        :param left: Embeddings of left sentences. Tensor of size (nb_embeddings_left, embed_dim)
        :param right: Embeddings of right sentence. Tensor of size (nb_embeddings_right, embed_dim)
        :param gamma: Parameter for kernel
        :return: Unbiased MMD^2 estimation between those two sets
        """
        ll = metrics.pairwise.rbf_kernel(left, left, gamma)
        rr = metrics.pairwise.rbf_kernel(right, right, gamma)
        lr = metrics.pairwise.rbf_kernel(left, right, gamma)

        ll_avg_mask = 1 - np.identity(ll.shape[0])
        rr_avg_mask = 1 - np.identity(rr.shape[0])

        # To be unbiased, we must not consider elements where i=j for ll and rr
        return np.average(ll, weights=ll_avg_mask) + np.average(rr, weights=rr_avg_mask) - 2 * lr.mean()

    def suggest_rbf_gamma(self, left: torch.Tensor, right: torch.Tensor) -> float:
        """
        We use the median heuristic to calculate the gamma to use in RBF kernel from sets of samples.
        The heuristic defines sigma as the median distance between all (aggregated) sample points.
        Then, gamma = 1 / (2 * sigma^2)
        :return:
        """
        # We limit left and right tensors to 3000 samples
        def select_samples(tensor: torch.Tensor) -> torch.Tensor:
            max_size = 3000
            size = tensor.shape[0]

            if size <= max_size:
                return tensor

            indexes = self._rng.choice(size, max_size, replace=False)
            return tensor[indexes, :]

        left = select_samples(left)
        right = select_samples(right)
        tensors = torch.cat((left, right))
        distances = F.pdist(tensors)
        sigma = np.median(distances.numpy(), overwrite_input=True)
        gamma = 1 / (2 * sigma**2)
        return gamma.item()

    def calculate(self, left: _DomainRepresentation, right: _DomainRepresentation) -> float:
        """
        Calculates MMD (using gaussian kernel) between left and right representations.

        :param left: Left domain mean. Tensor of size (nb_embeddings, embed_dim)
        :param right: Right domain mean. Tensor of size (nb_embeddings, embed_dim)
        :return: The MMD
        """
        self.logger.info("  (calculating gamma...)")
        gamma = self.suggest_rbf_gamma(left, right)
        self.logger.info("  (calculating MMD^2...)")
        return self.mmd2_rbf(left, right, gamma)


class FLD(HFTBasedDistance):
    """
    Similarity based on Fisher Linear Discriminant. See [1].
    """

    def __init__(self, batch_size=64, model='roberta-large', max_sentences: Optional[int] = None, device='cuda',
                 seed: Optional[int] = None, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        super().__init__(batch_size, model, max_sentences, device, seed, cache_dir, logger)
        self.name = "FLD"

    def save_cache(self, data: _RepresentationsDict):
        """
        For FLD, no need to cache representation, since its the same as domain embeddings which is already cached
        """
        return

    def _construct_representation(self, embeddings: torch.Tensor) -> _DomainRepresentation:
        """
        The representation of a domain is all the embeddings of its sentences. So this function simply returns the
        embeddings
        :param embeddings: Embeddings of sentence. Tensor of size (nb_embeddings, embed_dim)
        :return: Tensor of size (nb_embeddings, embed_dim)
        """
        return embeddings

    def get_cache_file_path(self) -> Path:
        return self.cache_dir / "fld.pkl"

    # noinspection PyPep8Naming
    def calculate(self, left: _DomainRepresentation, right: _DomainRepresentation) -> float:
        """
        Calculates an "FLD" distance (the FLD technique is used to calculate a distance) between left and right
        representations.

        :param left: Left domain mean. Tensor of size (nb_embeddings, embed_dim)
        :param right: Right domain mean. Tensor of size (nb_embeddings, embed_dim)
        :return: The "FLD" distance
        """

        # We use scikit-learn LinearDiscriminantAnalysis to easily calculate some values
        left = left.numpy()
        right = right.numpy()
        lda = LinearDiscriminantAnalysis(solver="eigen", priors=(0.5, 0.5))
        X = np.vstack((left, right))
        y = [0] * left.shape[0] + [1] * right.shape[0]
        lda.fit(X, y)
        mean_diff = lda.means_[0] - lda.means_[1]
        S_B = mean_diff[:, np.newaxis] @ mean_diff[np.newaxis, :]
        w = lda.coef_
        S_W = lda.covariance_
        distance = (w @ S_B @ w.T) / (w @ S_W @ w.T)
        return distance[0, 0]


class CORALLoss(HFTBasedDistance):
    """
    Similarity based on CORAL loss. See [1].
    """

    def __init__(self, batch_size=64, model='roberta-large', max_sentences: Optional[int] = None, device='cuda',
                 seed: Optional[int] = None, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        super().__init__(batch_size, model, max_sentences, device, seed, cache_dir, logger)
        self.name = "CORAL"

    def save_cache(self, data: _RepresentationsDict):
        """
        For CORAL, no need to cache representation, since its the same as domain embeddings which is already cached
        """
        return

    def _construct_representation(self, embeddings: torch.Tensor) -> _DomainRepresentation:
        """
        The representation of a domain is all the embeddings of its sentences. So this function simply returns the
        embeddings
        :param embeddings: Embeddings of sentence. Tensor of size (nb_embeddings, embed_dim)
        :return: Tensor of size (nb_embeddings, embed_dim)
        """
        return embeddings

    def get_cache_file_path(self) -> Path:
        return self.cache_dir / "coral.pkl"

    def calculate(self, left: _DomainRepresentation, right: _DomainRepresentation) -> float:
        """
        Calculates the CORAL loss between left and right representations.

        :param left: Left domain mean. Tensor of size (nb_embeddings, embed_dim)
        :param right: Right domain mean. Tensor of size (nb_embeddings, embed_dim)
        :return: The CORAL loss
        """
        d2 = left.shape[1]**2
        cov_left = np.cov(left, rowvar=False, ddof=1)
        cov_right = np.cov(right, rowvar=False, ddof=1)
        norm = np.sum((cov_left - cov_right)**2)
        return norm / (4 * d2)


class KLDiv(HFTBasedDistance):
    """
    Similarity based on KL divergence between multivariate gaussians fit over samples.
    """

    def __init__(self, batch_size=64, model='sentence-transformers/roberta-large-nli-stsb-mean-tokens',
                 max_sentences: Optional[int] = None, device='cuda', seed: Optional[int] = None,
                 cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        super().__init__(batch_size, model, max_sentences, device, seed, cache_dir, logger)
        self.name = "KLDiv"
        self._classifier = GaussianClassifier(cov_method="oas")
        self.symmetric = False

    def _construct_representation(self, embeddings: torch.Tensor) -> _DomainRepresentation:
        """
        The representation of a domain is the empirical mean and covariance matrix of embeddings
        :param embeddings: Embeddings of sentence. Tensor of size (nb_embeddings, embed_dim)
        :return: Tuple of mean (Tensor of size (embed_dim,)) and covariance (Tensor of size (embed_dim, embed_dim))
        """
        mean, cov = self._classifier.fit_gaussian(embeddings.numpy())
        return torch.from_numpy(mean), torch.from_numpy(cov)

    def get_cache_file_path(self) -> Path:
        return self.cache_dir / "kl_div.pkl"

    def calculate_between_domains(self, representations: _RepresentationsDict, left_domains: Sequence[str], right_domains: Sequence[str]) -> pd.DataFrame:
        """
        Calculates the KL divergence between multi-variate gaussians fitted on the samples and returns it as the
        similarity metric. Note that the KL divergence is not symmetric.

        Since it more efficient to calculate all divergences together than calculating each individually, this method
        does the actual calculations, not `calculate()`.

        :param representations: Domain representations. Key is domain name, value is result of `make_representation()`
        :param left_domains: List of domains we are interested in
        :param right_domains: List of domains to compare with each `left_domains`
        :return: Pandas dataframe (index is `left_domains`, columns is `right_domains`) of KL divergences
        """
        mu1 = [representations[d][0].numpy() for d in left_domains]
        sigma1 = [representations[d][1].numpy() for d in left_domains]
        mu2 = [representations[d][0].numpy() for d in right_domains]
        sigma2 = [representations[d][1].numpy() for d in right_domains]
        divergences = normals_kl_div(mu1, sigma1, mu2, sigma2)

        self.logger.info(f"Calculating {self.name} distances between <{', '.join(left_domains)}> and <{', '.join(right_domains)}>")
        return pd.DataFrame(divergences, columns=right_domains, index=left_domains, dtype=np.float)


class Tfidf(DomainsDistance):
    """
    Cosine distance between TF-IDF vectorisation of documents
    """

    def __init__(self, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None, **kwargs):
        super().__init__(cache_dir, logger)
        self.name = "TFIDF"
        self.kwargs = kwargs

    def get_cache_file_path(self) -> Path:
        return self.cache_dir / "tfidf.pkl"

    def calculate_between_dirs(self, dir_left: Path, dir_right: Optional[Path] = None, glob_left="*", glob_right="*") -> pd.DataFrame:
        filenames = []
        all_domains = []
        left_domains = []
        right_domains = []

        stemmer = nltk.stem.porter.PorterStemmer()
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

        def stem_tokens(tokens):
            return [stemmer.stem(item) for item in tokens]

        def normalize(text):
            return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

        stopwords = stem_tokens(STOP_WORDS)

        for f in dir_left.glob(glob_left):
            filenames.append(f.absolute())
            left_domains.append(f.stem)
            all_domains.append(f.stem)

        if dir_right:
            for f in dir_right.glob(glob_right):
                filenames.append(f.absolute())
                right_domains.append(f.stem)
                all_domains.append(f.stem)
        else:
            right_domains = left_domains

        cache = self.load_cache()

        if "tfidf" not in cache:
            vectorizer = TfidfVectorizer(input="filename", tokenizer=normalize, stop_words=stopwords, **self.kwargs)
            self.logger.info("Fit-transforming the TF-IDF vectorizer")
            tfidf = vectorizer.fit_transform(filenames)
            self.save_cache({"tfidf": tfidf})
        else:
            tfidf = cache["tfidf"]

        # We calculate cosine similarity (no need to normalize since TfidfVectorizer normalizes l2 rows)
        pairwise_similarity = tfidf * tfidf.T
        # The "distance" is 1 - cosine-similarity
        all_distances = pd.DataFrame(1 - pairwise_similarity.A, index=all_domains, columns=all_domains)
        return all_distances.loc[left_domains, right_domains]
