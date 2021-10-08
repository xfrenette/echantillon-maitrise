import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call

import numpy as np
import torch

from couleurs.domain_embedding import Embedder
from couleurs.evaluate.with_domain_embedding import DomainEmbedder, BatchBuilder, ScoreTask, Batch


def _create_score_task(rng) -> ScoreTask:
    domain = rng.choice(["dom1", "dom2", "dom3", "dom4"])
    embedding_set_size = rng.choice([10, 100, 1000, 500, 200, 400])
    embedding_set_id = rng.choice(50)
    set_type = rng.choice([ScoreTask.TEST_SET, ScoreTask.EMBEDDING_SET])
    set_id = rng.choice(50)
    return ScoreTask(domain, embedding_set_size, embedding_set_id, set_id, set_type)


def _create_mock_embedder(reproducible=False) -> Embedder:
    calculated_embeddings = {}

    def mock_calculate(sentences):
        if reproducible:
            key = sentences[0]
            if key not in calculated_embeddings:
                calculated_embeddings[key] = torch.rand(356)
            return calculated_embeddings[key]
        return torch.rand(356)

    embedder = Mock(Embedder)
    embedder.calculate.side_effect = mock_calculate
    return embedder


def _create_mock_batch_builder() -> BatchBuilder:
    def mock_create_batch(task: ScoreTask):
        source = [f"source: {task.key}"]
        target = [f"target: {task.key}"]
        return Batch(source, target)

    mock_batch_builder = Mock(BatchBuilder)
    mock_batch_builder.build_for_task.side_effect = mock_create_batch
    return mock_batch_builder


class DomainEmbedderTestCase(unittest.TestCase):
    def test_calculate_for_task(self):
        mock_embedder = _create_mock_embedder()
        mock_builder = _create_mock_batch_builder()
        embedder = DomainEmbedder(mock_embedder, mock_builder)

        test_task1 = ScoreTask("test", 100, 0, 0, ScoreTask.TEST_SET)
        test_task2 = ScoreTask("test", 100, 1, 0, ScoreTask.TEST_SET)
        embedding_task1 = test_task1.as_embedding_set()
        embedding_task2 = test_task2.as_embedding_set()

        # We do 4 calls, but 2 of them should generate the same embedding, so in total, the `mock_embedder` should be
        # called two times
        res1 = embedder.calculate_for_task(test_task1)
        res2 = embedder.calculate_for_task(embedding_task1)
        res3 = embedder.calculate_for_task(test_task2)
        res4 = embedder.calculate_for_task(test_task2)

        self.assertTrue(torch.equal(res1, res2))
        self.assertTrue(torch.equal(res3, res4))

        expected_batch1 = mock_builder.build_for_task(embedding_task1)
        expected_batch2 = mock_builder.build_for_task(embedding_task2)

        # Since 2 of the calls should have been cached from the previous 2 others, the `mock_embedder` should have been
        # called only 2 times
        mock_embedder.calculate.assert_has_calls([
            call(expected_batch1.source),
            call(expected_batch2.source),
        ])

    def test_precalculate_for_tasks(self):
        rng = np.random.default_rng(1984)
        with TemporaryDirectory() as tmp_dir_name:
            mock_embedder = _create_mock_embedder(True)
            mock_builder = _create_mock_batch_builder()
            cache_path = Path(tmp_dir_name) / "cache.pkl"

            # We create tasks for which we want to calculate embeddings
            tasks = [_create_score_task(rng) for _ in range(4)]
            # We keep a copy of the associated embeddings of those tasks
            expected_embeddings = [mock_embedder.calculate(mock_builder.build_for_task(t.as_embedding_set()).source) for t in tasks]

            # We precalculate all task embeddings
            embedder = DomainEmbedder(mock_embedder, mock_builder, cache_path)
            embedder.precalculate_for_tasks(iter(tasks))

            # To check that the embeddings were cached and saved in the cache file, we create a new instance of
            # DomainEmbedder from the cache file, we change the mock_embedder to always output None, and call
            # `calculate_for_task` for each task. Even if the mock_embedder returns None, it won't be called since the
            # embedding is already cached
            mock_embedder.calculate.side_effect = lambda s: None
            new_embedder = DomainEmbedder(mock_embedder, mock_builder, cache_path)
            new_embedder.load()
            for task, expected_embedding in zip(tasks, expected_embeddings):
                actual_embedding = new_embedder.calculate_for_task(task)
                self.assertTrue(torch.equal(actual_embedding, expected_embedding))


if __name__ == '__main__':
    unittest.main()
