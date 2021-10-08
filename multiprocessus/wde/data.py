"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-03-15
"""
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict


class DomainsDataSource:
    """
    Class to access sentences in domains data files.

    Can be used to retrieve source and target sentences from domain files in `data_dir`. Inside `data_dir`, source files
    are expected to have the following nomenclature: `{domain}{source_suffix}` and target files:
    `{domain}{target_suffix}`.
    """

    def __init__(self, data_dir: Path, source_suffix=".en", target_suffix=".fr"):
        """
        :param data_dir: Directory of domain files.
        :param source_suffix: Suffix (including dot, if applicable) of source files
        :param target_suffix:  Suffix (including dot, if applicable) of target files
        """
        self._data_dir = data_dir
        self._source_suffix = source_suffix
        self._target_suffix = target_suffix
        self._use_cache = False
        self._cache: Dict[Path, List[str]] = {}
        self._max_cache_size = 0

    @contextmanager
    def use_cache(self, use_cache=True, max_cache_size=0):
        """
        Context manager that enables caching for all calls to `get_*_sentences`.

        To prevent a too big cache, `max_cache_size` can be set to the maximum number of data files to keep in cache.
        If 0, no maximum is set.

        When the context manager exits, the cache is cleared.

        :param use_cache: Flag to enable or not the cache. Setting to False is like if this context manager was never
                          called.
        :param max_cache_size: Maximum number of files to keep in cache. Set to 0 to disable a maximum.
        """
        try:
            if use_cache:
                self._use_cache = True
                self._max_cache_size = max_cache_size
            yield
        finally:
            if use_cache:
                self._cache = {}
            self._use_cache = False

    def _save_to_cache(self, path: Path, sentences: List[str]):
        """
        Save the `sentences` in the cache associated with `path`.

        Makes sure the cache is never bigger than `self._max_cache_size` (if not 0).

        :param path: Path of the file to associate in the cache to the `sentences`
        :param sentences: Sentences to keep in cache for this `path`
        """
        cache_keys = list(self._cache.keys())

        if 0 < self._max_cache_size <= len(cache_keys) and path not in self._cache:
            del self._cache[cache_keys[0]]

        self._cache[path] = sentences

    def _get_sentences(self, domain: str, suffix: str) -> List[str]:
        """
        Returns sentences in the data file for this `domain` and `suffix`. If no file correspond, a `FileNotFoundError`
        is raised.

        If we are using the cache (see `use_cache()`), loaded sentences for each requested file is saved in internal
        cache and returned if requested again.

        :param domain: Domain to retrieve
        :param suffix: Suffix of the data file
        :return: List of sentences, with ending new lines stripped
        """
        path = (self._data_dir / domain).with_suffix(suffix)

        if self._use_cache and path in self._cache:
            return self._cache[path]

        if not path.exists():
            raise FileNotFoundError(f"File not found for domain {domain} with suffix {suffix}")

        with open(path, "r") as file:
            lines = file.readlines()
            sentences = [line.rstrip("\n") for line in lines]

        if self._use_cache:
            self._save_to_cache(path, sentences)

        return sentences

    def get_source_sentences(self, domain: str) -> List[str]:
        """
        Returns list of sentences in the source file for this `domain`. Ending new lines are stripped.

        :param domain: Domain to retrieve
        :return: List of sentences
        """
        return self._get_sentences(domain, self._source_suffix)

    def get_target_sentences(self, domain: str) -> List[str]:
        """
        Returns list of sentences in the target file for this `domain`. Ending new lines are stripped.

        :param domain: Domain to retrieve
        :return: List of sentences
        """
        return self._get_sentences(domain, self._target_suffix)

    @property
    def domains(self) -> List[str]:
        """
        Returns all valid domain names in the data directory.

        A domain is valid if it has a source and a target file named with the domain name.

        :return: List of domains
        """
        domains = []
        for source_path in self._data_dir.glob(f"*{self._source_suffix}"):
            target_path = source_path.with_suffix(self._target_suffix)
            if not target_path.exists():
                continue
            domains.append(source_path.stem)
        return domains
