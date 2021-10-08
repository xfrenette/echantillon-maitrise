"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-03-15
"""
import itertools
import logging
from multiprocessing import Process
from pathlib import Path
from typing import Optional, Sequence, Iterator, NamedTuple

import numpy as np
import pandas as pd

from couleurs.utils.multiprocessing import StoppableQueue, StoppedException
from .data import DomainsDataSource


class ScoreTask:
    """
    Represents a scoring task
    """

    TEST_SET = "test"
    EMBEDDING_SET = "embedding"

    def __init__(self, domain: str, embedding_set_size: int, embedding_set_id: int, set_id: int, set_type: str):
        self.domain = domain
        self.embedding_set_size = embedding_set_size
        self.embedding_set_id = embedding_set_id
        self.set_id = set_id
        self.set_type = set_type

    def as_embedding_set(self) -> "ScoreTask":
        """
        Returns a new ScoreTask instance that is a copy of this one, except that the `set_type` is set to "embedding".

        Returns a new instance even if the current ScoreTask already has an "embedding" `set_type`.

        :return: A new ScoreTask instance
        """
        return ScoreTask(self.domain, self.embedding_set_size, self.embedding_set_id, self.set_id, self.EMBEDDING_SET)

    @property
    def embedding_key(self) -> str:
        return f"{self.domain}_{self.embedding_set_size}_{self.embedding_set_id}"

    @property
    def key(self) -> str:
        return f"{self.embedding_key}_{self.set_id}_{self.set_type}"

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__


class ScoreTaskListStats(NamedTuple):
    """Statistic object returned by `ScoreTaskList.stats`"""
    total: int
    incomplete: int
    completed: int


class ScoreTaskList:
    """A list of scoring tasks"""

    SCHEMA = {
        "domain": str,
        "embedding_set_size": int,
        "embedding_set_id": int,
        "set_type": str,
        "set_id": int,
        "metric": str,
        "score": float,
    }

    def __init__(self, cache_file: Optional[Path] = None):
        """
        :param cache_file: File where to save the score task list, or from which to reload the list
        """
        self.data = pd.DataFrame(columns=self.SCHEMA.keys()).astype(self.SCHEMA)
        self.cache_file = cache_file

    @classmethod
    def from_file(cls, cache_file: Path) -> "ScoreTaskList":
        """
        Create a instance from the score task list in the `cache_file`

        :param cache_file: File to read data from
        :return: An instance of ScoreTaskList
        """
        score_task_list = cls(cache_file)
        score_task_list.reload()
        return score_task_list

    @staticmethod
    def _sort_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Sort a DataFrame of data
        :param data: DataFrame
        :return: The sorted DataFrame
        """
        return data.sort_values(by=list(ScoreTaskList.SCHEMA.keys()), ignore_index=True)

    @classmethod
    def create(cls, domains_data: DomainsDataSource, embedding_set_sizes: Sequence[int], n_embedding_sets: int,
               n_test_sets: int, metrics: Sequence[str]) -> "ScoreTaskList":
        """
        Creates an instance of `ScoreTaskList` with a predefined list of tasks created.

        To create a tasks list without any embedding sets (only test tasks), set embedding_set_sizes
        to an empty sequence. n_embedding_sets must still be an int, but it will be ignored, so
        better set it to 0 to remove any confusion. Test tasks will have an "embedding_set_size" of
        0 and a "embedding_set_id" of 0.

        :param domains_data: DomainsDataSource that will return all domains
        :param embedding_set_sizes: Sizes of embedding sets
        :param n_embedding_sets: Number of embedding sets (bootstrap)
        :param n_test_sets: Number of test sets (bootstrap)
        :param metrics: List of metrics to use
        :return: Instance of ScoreTaskList
        """
        data = []
        skip_embeddings = False

        if len(embedding_set_sizes) == 0:
            embedding_set_sizes = (0,)
            n_embedding_sets = 1
            skip_embeddings = True

        product = itertools.product(domains_data.domains, embedding_set_sizes, range(n_embedding_sets))

        for domain, embedding_set_size, embedding_set_id in product:
            if not skip_embeddings:
                for metric in metrics:
                    data.append([domain, embedding_set_size, embedding_set_id, ScoreTask.EMBEDDING_SET, 0, metric, np.NaN])
            for set_id, metric in itertools.product(range(n_test_sets), metrics):
                data.append([domain, embedding_set_size, embedding_set_id, ScoreTask.TEST_SET, set_id, metric, np.NaN])

        instance_data = pd.DataFrame(data, columns=cls.SCHEMA.keys()).astype(cls.SCHEMA)
        instance = cls()
        instance.data = cls._sort_data(instance_data)

        return instance

    @staticmethod
    def _generate_index(data: pd.DataFrame):
        """
        Return the `data` DataFrame with an index generated for each row. Used when merging two sets to determine
        rows with the same task.

        :param data: DataFrame for which to generate the index
        :return: A copy of `data` with the new index
        """
        def calculate_key(r):
            key = ""
            for column in data.columns:
                if column == "score":
                    continue
                key += f"{r[column]}_"
            return key
        index = data.apply(calculate_key, axis=1)
        return data.set_index(index)

    def merge(self, other: "ScoreTaskList"):
        """
        Merge score tasks from the `other` list into the current list.

        :param other: Other `ScoreTaskList` from which to import task list
        """
        other_reindexed = self._generate_index(other.data)
        self_reindexed = self._generate_index(self.data)

        self_index = self_reindexed.index
        other_index = other_reindexed.index

        indices_to_update = other_index.difference(self_index).append(self_index.intersection(other_index))

        for index in indices_to_update:
            self_reindexed.loc[index, :] = other_reindexed.loc[index, :]

        self_reindexed = self_reindexed.astype(self.SCHEMA, copy=False)

        self.data = self._sort_data(self_reindexed.reset_index(drop=True))

    @property
    def incomplete_tasks(self) -> Iterator[ScoreTask]:
        """
        Returns an iterator that iterates on ScoreTask that still don't have a score assigned.

        Note that two identical tasks with a different metric are considered as just one task. If one or both don't
        have a score, a ScoreTask will returned in this property.

        A copy of the list is created when this function starts. Any modification to the list won't be reflected in this
        iterator.

        :return: Iterator of incomplete ScoreTask
        """
        no_score_data = self.data.loc[self.data["score"].isna()]
        return self._tasks_iterator(no_score_data)

    @property
    def tasks(self) -> Iterator[ScoreTask]:
        """
        Returns an iterator of all ScoreTasks, completed or not.

        Note that two identical tasks with a different metric are considered as just one task.

        A copy of the list is created when this function starts. Any modification to the list won't be reflected in this
        iterator.

        :return: Iterator of incomplete ScoreTask
        """
        return self._tasks_iterator(self.data)

    def _tasks_iterator(self, data: pd.DataFrame) -> Iterator[ScoreTask]:
        cols_subset = [c for c in self.SCHEMA.keys() if c not in ["score", "metric"]]
        tasks_data = data.drop_duplicates(cols_subset, ignore_index=True)

        for row in tasks_data.itertuples():
            yield ScoreTask(row.domain, row.embedding_set_size, row.embedding_set_id, row.set_id, row.set_type)

    @property
    def stats(self) -> ScoreTaskListStats:
        """
        Returns statistics about the tasks
        :return: a ScoreTaskListStats
        """
        total = self.data.shape[0]
        incomplete = self.data["score"].isna().sum()
        complete = total - incomplete
        return ScoreTaskListStats(total, incomplete, complete)

    def dump(self):
        """
        Save current list in pickle file, if set.
        """
        if self.cache_file is None:
            return
        pd.to_pickle(self.data, self.cache_file)

    def reload(self):
        """
        Reload the score tasks from the pickle file, if set.
        :return:
        """
        if self.cache_file is None:
            return
        self.data = pd.read_pickle(self.cache_file)

    def __eq__(self, other: "ScoreTaskList"):
        return isinstance(other, self.__class__) and self.data.equals(other.data)

    def listen(self, scores: StoppableQueue) -> Process:
        """
        Creates and returns a process that will listen to a `ScoreResult` queue and will update the pickled list.

        When the process starts, it makes its own copy of the internal list, so the process and its parent ScoreTaskList
        may not be synchronized. To resynchronize, call `reload()` on the parent list.

        :param scores: Queue of `ScoreResult` to listen to
        :return: The process that was started
        """
        p = Process(target=self._listen, args=(scores,))
        return p

    def _listen(self, scores: StoppableQueue):
        """
        Listens to the `ScoreResult` queue and update the internal list and pickle file
        :param scores: Queue of `ScoreResult` to listen to
        """
        logger = logging.getLogger("ScoreTaskList(listener)")
        try:
            iter_without_save = 0
            while True:
                try:
                    next_score = scores.watching_get()

                    if next_score is None:
                        # No more scores will be generated, so we exit the process
                        if iter_without_save > 0:
                            self.dump()
                        return

                    self.update(next_score)

                    if scores.qsize() == 0 or iter_without_save > 5:
                        self.dump()
                        iter_without_save = 0
                    else:
                        iter_without_save += 1
                except StoppedException:
                    # We break the loop, to exit the process
                    break
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt raised")

    # score is of type ScoreResult, but we cannot import it because of an import loop
    def update(self, score):
        """
        Add or update a ScoreTask with a ScoreResult.

        :param score: ScoreResult
        """
        new_row = {
            "domain": score.score_task.domain,
            "embedding_set_size": score.score_task.embedding_set_size,
            "embedding_set_id": score.score_task.embedding_set_id,
            "set_id": score.score_task.set_id,
            "set_type": score.score_task.set_type,
            "metric": score.metric,
            "score": score.score,
        }

        select: pd.Series = pd.Series(True, index=self.data.index)
        for col, value in new_row.items():
            if col == "score":
                continue
            # noinspection PyTypeChecker
            select = select & (self.data[col] == value)

        if select.any():
            self.data.loc[select, "score"] = score.score
        else:
            # For a new row, we append it and then resort the data
            self.data = self._sort_data(self.data.append(new_row, ignore_index=True))
