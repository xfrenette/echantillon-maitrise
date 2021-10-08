import itertools
import shutil
import time
import unittest
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import Mock

import numpy as np
import pandas as pd

from couleurs.evaluate.with_domain_embedding import DomainsDataSource, ScoreResult, ScoreTaskList, ScoreTask
from couleurs.utils.multiprocessing import StoppableQueue

_rng = np.random.default_rng()


def _task_sort_key(task: ScoreTask):
    return task.key


def _create_score_result(score: Optional[float] = None, metric: Optional[str] = None) -> ScoreResult:
    domain = _rng.choice(["dom1", "dom2", "dom3", "dom4"])
    embedding_set_size = _rng.choice([10, 100, 1000, 500, 200, 400])
    embedding_set_id = _rng.choice(50)
    set_type = _rng.choice([ScoreTask.TEST_SET, ScoreTask.EMBEDDING_SET])
    set_id = _rng.choice(50)
    result_metric = _rng.choice(["bleu", "bert"]) if metric is None else metric
    result_score = _rng.random() if score is None else score
    task = ScoreTask(domain, embedding_set_size, embedding_set_id, set_id, set_type)
    return ScoreResult(task, result_metric, result_score)


def _trigger_shutdown_event(event: EventType, timeout: float):
    def trigger(event: EventType, timeout: float):
        time.sleep(timeout)
        event.set()

    p = Process(target=trigger, args=(event, timeout))
    p.start()


class ScoreTaskListTestCase(unittest.TestCase):
    def assertScoreInList(self, score_list: ScoreTaskList, result: ScoreResult):
        data = score_list.data
        select = pd.Series(True, index=data.index)
        select = select & (data["domain"] == result.score_task.domain)
        select = select & (data["embedding_set_size"] == result.score_task.embedding_set_size)
        select = select & (data["embedding_set_id"] == result.score_task.embedding_set_id)
        select = select & (data["set_id"] == result.score_task.set_id)
        select = select & (data["set_type"] == result.score_task.set_type)
        select = select & (data["metric"] == result.metric)

        self.assertEqual(1, select.sum())
        self.assertTrue((data.loc[select, "score"] == result.score).all())

    def test_update_add(self):
        """Test that `update()` can add new scores"""
        scores = ScoreTaskList()
        new_score = _create_score_result()
        scores.update(new_score)

        self.assertEqual(1, scores.data.shape[0])
        self.assertScoreInList(scores, new_score)

    def test_update_modify(self):
        """Test that `update()` can modify existing scores"""
        scores = ScoreTaskList()
        first_score = _create_score_result()
        scores.update(first_score)

        # We change the score of the last score result
        new_score_value = first_score.score * 2
        new_score = ScoreResult(first_score.score_task, first_score.metric, new_score_value)
        scores.update(new_score)

        self.assertEqual(1, scores.data.shape[0])
        self.assertScoreInList(scores, new_score)

    def test_dump(self):
        """Test the `dump()` method"""
        with TemporaryDirectory() as tmp_dir_name:
            pickle_path = Path(tmp_dir_name) / "tmp.pkl"
            scores = ScoreTaskList(pickle_path)
            scores.update(_create_score_result())
            expected_data = scores.data.copy()

            scores.dump()

            actual_data = pd.read_pickle(pickle_path)
            pd.testing.assert_frame_equal(expected_data, actual_data)

    def test_reload(self):
        """Test the `reload()` method"""
        with TemporaryDirectory() as tmp_dir_name:
            orig_pkl = Path(tmp_dir_name) / "orig.pkl"
            other_pkl = Path(tmp_dir_name) / "other.pkl"

            # We first create a first list and dump it in a file
            original_scores = ScoreTaskList(orig_pkl)
            original_scores.update(_create_score_result())
            original_scores.dump()

            # We then create a new list that will load from the dump file of the other
            shutil.copy(orig_pkl, other_pkl)
            new_scores = ScoreTaskList(other_pkl)
            new_scores.reload()

            pd.testing.assert_frame_equal(original_scores.data, new_scores.data)

    def test_from_file(self):
        """Test the class method `from_file()`"""
        with TemporaryDirectory() as tmp_dir_name:
            pickle_file = Path(tmp_dir_name) / "orig.pkl"

            # We first create a first list and dump it in a file
            original_scores = ScoreTaskList(pickle_file)
            original_scores.update(_create_score_result())
            original_scores.dump()

            # We then create a new list from that file
            new_scores = ScoreTaskList.from_file(pickle_file)

            self.assertIsInstance(new_scores, ScoreTaskList)
            pd.testing.assert_frame_equal(original_scores.data, new_scores.data)

    def test_create(self):
        domains = ("dom1", "dom2")  # Alphabetical order
        embedding_set_sizes = (10, 20)  # Numeric order
        embedding_set_ids = (0, 1)
        metrics = ("bert", "bleu")  # Alphabetical order
        test_set_id = (0, 1)
        data = []

        for domain, embedding_set_size, embedding_set_id in itertools.product(domains, embedding_set_sizes, embedding_set_ids):
            for metric in metrics:
                data.append([domain, embedding_set_size, embedding_set_id, ScoreTask.EMBEDDING_SET, 0, metric, np.NaN])
            for set_id, metric in itertools.product(test_set_id, metrics):
                data.append([domain, embedding_set_size, embedding_set_id, ScoreTask.TEST_SET, set_id, metric, np.NaN])

        expected = pd.DataFrame(data, columns=ScoreTaskList.SCHEMA.keys()).astype(ScoreTaskList.SCHEMA)

        mock_domain_data_source = Mock(DomainsDataSource)
        mock_domain_data_source.domains = domains[::-1]  # We reverse the list

        scores = ScoreTaskList.create(mock_domain_data_source, embedding_set_sizes[::-1], len(embedding_set_ids), len(test_set_id), metrics[::-1])
        pd.testing.assert_frame_equal(expected, scores.data)

    def test_create_no_embedding(self):
        """Creates expected tasks when no embedding set sizes are provided"""
        domains = ("dom1", "dom2")  # Alphabetical order
        metrics = ("bert", "bleu")  # Alphabetical order
        test_set_id = (0, 1)
        data = []

        for domain in domains:
            for set_id, metric in itertools.product(test_set_id, metrics):
                data.append([domain, 0, 0, ScoreTask.TEST_SET, set_id, metric, np.NaN])

        expected = pd.DataFrame(data, columns=ScoreTaskList.SCHEMA.keys()).astype(ScoreTaskList.SCHEMA)

        mock_domain_data_source = Mock(DomainsDataSource)
        mock_domain_data_source.domains = domains[::-1]  # We reverse the list

        scores = ScoreTaskList.create(mock_domain_data_source, [], 0, len(test_set_id), metrics[::-1])
        pd.testing.assert_frame_equal(expected, scores.data)

    def test_merge(self):
        """Test that `merge()` updates existing scores and adds the new ones"""
        main = ScoreTaskList()
        score1 = _create_score_result()
        score2 = _create_score_result()
        main.update(score1)
        main.update(score2)

        other = ScoreTaskList()
        # The following will overwrite the score1
        score1_other = ScoreResult(score1.score_task, score1.metric, score1.score * 2)
        # The following doesn't exist in `main`, so it will be added
        score3_other = _create_score_result()
        other.update(score1_other)
        other.update(score3_other)

        expected_dtypes = main.data.dtypes

        # We merge the `other` in `main`. It should now consist of 3 scores: score1_other, score2 and score3_other
        main.merge(other)
        self.assertEqual(3, main.data.shape[0])
        self.assertScoreInList(main, score1_other)
        self.assertScoreInList(main, score2)
        self.assertScoreInList(main, score3_other)

        # Also, the `data` attribute should have columns of the correct type
        self.assertTrue((expected_dtypes == main.data.dtypes).all())


    def test_incomplete_tasks(self):
        """Test that `incomplete_tasks` returns an iterator of tasks with NaN scores"""
        scores = ScoreTaskList()
        score1 = _create_score_result()
        # The next 2 score results are identical, except for the metric.
        # Only one ScoreTask should be returned for those two by `incomplete_tasks`
        score2 = _create_score_result(score=np.NaN, metric="bleu")
        score2_2 = ScoreResult(score2.score_task, "bert", np.NaN)
        score3 = _create_score_result()
        score4 = _create_score_result(score=np.NaN)

        scores.update(score1)
        scores.update(score2)
        scores.update(score2_2)
        scores.update(score3)
        scores.update(score4)

        incomplete_tasks = scores.incomplete_tasks

        expected = [score2.score_task, score4.score_task]
        actual = [next(incomplete_tasks), next(incomplete_tasks)]

        with self.assertRaises(StopIteration):
            next(incomplete_tasks)

        expected.sort(key=_task_sort_key)
        actual.sort(key=_task_sort_key)

        self.assertListEqual(expected, actual)

    def test_tasks(self):
        scores = ScoreTaskList()
        score1 = _create_score_result()
        # The next 2 score results are identical, except for the metric.
        # Only one ScoreTask should be returned for those two by `tasks`
        score2 = _create_score_result(score=np.NaN, metric="bleu")
        score2_2 = ScoreResult(score2.score_task, "bert", np.NaN)
        score3 = _create_score_result()
        score4 = _create_score_result(score=np.NaN)

        scores.update(score1)
        scores.update(score2)
        scores.update(score2_2)
        scores.update(score3)
        scores.update(score4)

        tasks = scores.tasks

        expected = [score1.score_task, score2.score_task, score3.score_task, score4.score_task]
        actual = [next(tasks) for _ in expected]

        with self.assertRaises(StopIteration):
            next(tasks)

        expected.sort(key=_task_sort_key)
        actual.sort(key=_task_sort_key)

        self.assertListEqual(expected, actual)

    def test_equals(self):
        scores1 = ScoreTaskList(Path("some/file/path.pt"))
        scores2 = ScoreTaskList()

        score1 = _create_score_result()
        score2 = _create_score_result()
        score3 = _create_score_result()

        # After the following, both ScoreTaskList should not be equal
        scores1.update(score1)
        scores1.update(score2)
        scores1.update(score3)

        scores2.update(score1)
        scores2.update(score3)

        self.assertNotEqual(scores1, scores2)

        # But after adding the missing score, even if out of order, they should now be equal
        scores2.update(score2)
        self.assertEqual(scores1, scores2)

    def test_stats(self):
        scores = ScoreTaskList()

        scores.update(_create_score_result())
        scores.update(_create_score_result(np.NaN))
        scores.update(_create_score_result())

        stats = scores.stats

        self.assertEqual(3, stats.total)
        self.assertEqual(1, stats.incomplete)
        self.assertEqual(2, stats.completed)

    def test_listen(self):
        """Test that `listen()` updates the pickle file when data is added and stops """
        with TemporaryDirectory() as tmp_dir_name:
            pickle_path = Path(tmp_dir_name) / "data.pkl"
            shutdown_event = Event()
            model_list = ScoreTaskList()
            listening_list = ScoreTaskList(pickle_path)
            scores_queue = StoppableQueue(stop_event=shutdown_event)

            score1 = _create_score_result()
            score2 = _create_score_result()

            model_list.update(score1)
            model_list.update(score2)

            process = listening_list.listen(scores_queue)
            process.start()
            scores_queue.put(score2)
            scores_queue.put(score1)
            scores_queue.put(None)

            process.join(1)

            self.assertFalse(process.is_alive())

            actual_data = pd.read_pickle(pickle_path)
            pd.testing.assert_frame_equal(model_list.data, actual_data)

    def test_shutdown_event(self):
        """Test the listening process stops when the scores queue's shutdown event is triggered"""
        shutdown_event = Event()
        scores_list = ScoreTaskList()
        scores_queue = StoppableQueue(stop_event=shutdown_event)
        process = scores_list.listen(scores_queue)
        process.start()
        scores_queue.put(_create_score_result())

        # We trigger the stop event after a delay
        _trigger_shutdown_event(shutdown_event, 0.5)

        # The process should join before 2 seconds
        process.join(2)

        self.assertFalse(process.is_alive())


class ScoreTaskTestCase(unittest.TestCase):
    def test_as_embedding_set(self):
        task_test = ScoreTask("dom1", 100, 2, 3, ScoreTask.TEST_SET)
        task_embedding = ScoreTask("dom1", 100, 2, 3, ScoreTask.EMBEDDING_SET)
        expected = ScoreTask("dom1", 100, 2, 3, ScoreTask.EMBEDDING_SET)

        self.assertEqual(expected, task_test.as_embedding_set())
        self.assertEqual(expected, task_embedding.as_embedding_set())


if __name__ == '__main__':
    unittest.main()
