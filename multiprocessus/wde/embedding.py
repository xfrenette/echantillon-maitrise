"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-03-15
"""
import logging
import pickle
from pathlib import Path
from typing import Optional, Iterable, Dict

import torch
import tqdm

from couleurs.domain_embedding import Embedder
from . import BatchBuilder, ScoreTask


class DomainEmbedder:
    """
    Class that calculates and returns domain embedding for ScoreTasks.
    """

    def __init__(self, embedder: Embedder, batch_builder: BatchBuilder, cache_file: Optional[Path] = None,):
        self._embedder = embedder
        self._batch_builder = batch_builder
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._cache_file = cache_file
        self._logger = logging.getLogger(__name__ + ".DomainEmbedder")

    def precalculate_for_tasks(self, tasks: Iterable[ScoreTask], progress: Optional[tqdm.tqdm] = None):
        """
        Calculates embedding for all ScoreTask instances in `tasks` and save them in the internal cache and in the file
        (if set)

        `tqdm` is a tqdm instance that that represents the progress. At every step, the description will be updated
            and the `update()` method will be called after each completion. Make sure the instance has its
            `total` correctly set before passing it to this method.

        :param tasks: iterable of `ScoreTask` for which to precalculate the embeddings
        :param progress: tqdm instance that tracks progress.
        """
        save_every = 5
        i = 0
        for i, task in enumerate(tasks, start=1):
            key = task.embedding_key

            if key in self._embeddings:
                if progress is not None:
                    progress.update()
                continue
            else:
                self._logger.info(f"Calculating embedding for {key}")
                if progress is not None:
                    progress.set_postfix(embedding=key)
                self.calculate_for_task(task)
                if progress is not None:
                    progress.update()
                if i % save_every == 0:
                    self.save()

        if i % save_every != 0:
            self.save()

    def calculate_for_task(self, task: ScoreTask) -> torch.Tensor:
        """
        Calculates and returns the torch embedding for the specified task

        Embeddings are always calculated for an "embedding" `set_type`. The `set_type` will be changed before
        calculating the embedding. If the embedding is cached, it will be returned and not re-calculated.

        :param task: `ScoreTask` for which to calculate the embedding
        :return: A 1D torch tensor of the embedding
        """
        key = task.embedding_key

        if key not in self._embeddings:
            task = task.as_embedding_set()
            batch = self._batch_builder.build_for_task(task)
            embedding = self._embedder.calculate(batch.source)
            self._embeddings[key] = embedding

        return self._embeddings[key]

    def load(self):
        """
        Loads the embeddings from the cache file, if set and if it exists
        """
        if self._cache_file and self._cache_file.exists():
            with open(self._cache_file, "rb") as out_file:
                self._embeddings = pickle.load(out_file)

    def save(self):
        """
        Save the embeddings from the cache file, if set
        """
        if self._cache_file:
            self._logger.info(f"Saving calculated embeddings")
            with open(self._cache_file, "wb") as out_file:
                pickle.dump(self._embeddings, out_file)
