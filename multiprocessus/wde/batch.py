"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-03-15
"""
import hashlib
import os
from typing import List, Optional, NamedTuple, Sequence, TypeVar, Tuple, overload, Union

import numpy as np
import torch

from .data import DomainsDataSource
from .score_task import ScoreTask


_T = TypeVar('_T')


class Batch(NamedTuple):
    """
    A batch of source sentences and their respective target sentence
    """
    source: List[str]
    target: List[str]


class BatchBuilder:
    """
    Class that creates reproducible batches for score tasks. Batches are randomly selected, with replacement, from
    domain data files. It is guaranteed that the same seed will produce the same batch.
    """

    def __init__(self, domains_data_source: DomainsDataSource, test_set_size: int, seed: Optional[int] = None):
        self.data_source = domains_data_source
        self.test_set_size = test_set_size
        self.seed = int.from_bytes(os.urandom(16), 'big') if seed is None else seed

    def _batchify(self, task: ScoreTask,
                  *datasets: Union[Sequence[_T], torch.Tensor]) -> Tuple[Union[List[_T], torch.Tensor], ...]:
        task_hash = hashlib.md5(task.key.encode("utf-8")).hexdigest()
        seed = self.seed + int(task_hash, 16)
        rng = np.random.default_rng(seed)
        batch_size = task.embedding_set_size if task.set_type == ScoreTask.EMBEDDING_SET else self.test_set_size

        indices = rng.choice(len(datasets[0]), batch_size, replace=True)
        batches = []

        for ds in datasets:
            if isinstance(ds, torch.Tensor):
                batches.append(ds[indices])
            else:
                batches.append([ds[i] for i in indices])

        return tuple(batches)

    def build_for_task(self, task: ScoreTask) -> Batch:
        """
        Generate a Batch instance for the specified task. If the `task.set_type` is "embedding", the batch size will be
        of `task.embedding_set_size`. Else it will be of `test_set_size` passed to `__init__`.

        :param task: The ScoreTask for which to generate a batch
        :return: The resulting Batch instance
        """
        source_sentences = self.data_source.get_source_sentences(task.domain)
        target_sentences = self.data_source.get_target_sentences(task.domain)

        batch_source, batch_target = self._batchify(task, source_sentences, target_sentences)

        return Batch(batch_source, batch_target)

    @overload
    def batchify(self, data: torch.Tensor, task: ScoreTask) -> torch.Tensor:
        pass

    @overload
    def batchify(self, data: Sequence[_T], task: ScoreTask) -> List[_T]:
        pass

    def batchify(self, data, task: ScoreTask):
        """
        Generate a random batch from `data` for this `task` and return it.

        The size of the returned batch depends on the `set_type` of the task. If it's an embedding set, the size is
        defined in the task (as `embedding_set_size` property). If it's a test set, the size is defined by the
        `test_set_size` property defined when instantiating the BatchBuilder.

        Calling this method with the same data and same task will always return the same random batch. This method
        also guarantees to return the same set of source and target sentences as in the batch returned by
        `build_for_task`, if the `data` is this instance's datasource's source or target sentences.

        :param data: The data from which to select the data (can be a torch Tensor)
        :param task: The task for which to create a batch
        :return: A batch of elements taken from `data`
        """
        batches = self._batchify(task, data)
        return batches[0]
