import unittest
from typing import Sequence
from unittest.mock import Mock

import torch
from random_words import LoremIpsum

from evaluate.with_domain_embedding import DomainsDataSource, BatchBuilder, ScoreTask, Batch


def _create_mock_domains_data_source(domains: Sequence[str], nb_sentences=30) -> DomainsDataSource:
    lipsum = LoremIpsum()
    data = {d: lipsum.get_sentences_list(nb_sentences) for d in domains}

    def mock_source_sentences(domain):
        return [f"source: {s}" for s in data[domain]]

    def mock_target_sentences(domain):
        return [f"target: {s}" for s in data[domain]]

    mock = Mock(DomainsDataSource)
    mock.get_source_sentences.side_effect = mock_source_sentences
    mock.get_target_sentences.side_effect = mock_target_sentences

    return mock


class BatchBuilderTestCase(unittest.TestCase):
    def test_build_for_task_batch_size(self):
        """Test that `build_for_task` returns correct batch sizes"""
        test_set_size = 21
        mock_data_source = _create_mock_domains_data_source(["dom1"])
        builder = BatchBuilder(mock_data_source, test_set_size)
        task1 = ScoreTask("dom1", 8, 0, 0, ScoreTask.EMBEDDING_SET)
        task2 = ScoreTask("dom1", 8, 0, 0, ScoreTask.TEST_SET)

        # The batch for the embedding set should be the one in the task
        batch = builder.build_for_task(task1)
        self.assertEqual(task1.embedding_set_size, len(batch.source))
        self.assertEqual(task1.embedding_set_size, len(batch.target))

        # The batch for a test set should be the default size
        batch = builder.build_for_task(task2)
        self.assertEqual(test_set_size, len(batch.source))
        self.assertEqual(test_set_size, len(batch.target))

    def test_build_for_task_valid_data(self):
        """Test that `build_for_task` returns the correct data for source and target"""
        mock_data_source = _create_mock_domains_data_source(["dom1"])
        builder = BatchBuilder(mock_data_source, 10)
        task = ScoreTask("dom1", 5, 0, 0, ScoreTask.TEST_SET)
        batch = builder.build_for_task(task)

        for source, target in zip(batch.source, batch.target):
            expected_target = "target: " + source[len("source: "):]
            self.assertEqual(target, expected_target)

    def test_build_for_task_randomness_with_seed(self):
        """Test `build_for_task` returns different batches for different seed, but the same batch for the same seed"""
        mock_data_source = _create_mock_domains_data_source(["dom1"], 5)  # The data source has less sentences than the batch size
        seed1 = 123
        seed2 = 456
        task = ScoreTask("dom1", 5, 0, 0, ScoreTask.TEST_SET)

        builder = BatchBuilder(mock_data_source, 10, seed1)

        batch1 = builder.build_for_task(task)
        # A different seed (batch should be different)
        builder.seed = seed2
        batch2 = builder.build_for_task(task)
        # Back to the first seed, batch3 should be the same as batch1
        builder.seed = seed1
        batch3 = builder.build_for_task(task)

        self.assertEqual(batch1, batch3)
        self.assertNotEqual(batch1, batch2)

    def test_build_for_task_randomness_between_tasks(self):
        """Test that different tasks return different batches"""
        mock_data_source = _create_mock_domains_data_source(["dom1", "dom2"], 500)
        builder = BatchBuilder(mock_data_source, 10, 1234)

        tasks = [
            ScoreTask("dom1", 5, 1, 3, ScoreTask.TEST_SET),
            # Same as first, but with different domain
            ScoreTask("dom2", 5, 1, 3, ScoreTask.TEST_SET),
            # Same as first, but with different embedding set size
            ScoreTask("dom1", 6, 1, 3, ScoreTask.TEST_SET),
            # Same as first, but with different embedding set id
            ScoreTask("dom1", 5, 2, 3, ScoreTask.TEST_SET),
            # Same as first, but with different set id
            ScoreTask("dom1", 5, 2, 2, ScoreTask.TEST_SET),
            # Same as first, but with different set type
            ScoreTask("dom1", 5, 1, 3, ScoreTask.EMBEDDING_SET),
        ]

        batches = [builder.build_for_task(task) for task in tasks]

        def sort_key(batch: Batch):
            return "\n".join(batch.source)

        batches.sort(key=sort_key)

        prev_batch = None
        for i, batch in enumerate(batches):
            self.assertNotEqual(prev_batch, batch, f"Batch #{i} equals previous batch")
            prev_batch = batch

    def test_batchify(self):
        mock_data_source = _create_mock_domains_data_source(["dom1"], 500)
        test_set_size = 11
        builder = BatchBuilder(mock_data_source, test_set_size, 1234)
        data = list(range(19))

        task_embedding = ScoreTask("dom1", 5, 4, 3, ScoreTask.EMBEDDING_SET)
        task_test = ScoreTask("dom1", 10, 1, 0, ScoreTask.TEST_SET)
        task_test_other = ScoreTask("dom1", 10, 1, 2, ScoreTask.TEST_SET)

        # The size for a "embedding set" batch should be the same as in the task
        batch = builder.batchify(data, task_embedding)
        self.assertEqual(task_embedding.embedding_set_size, len(batch))

        # The size for a "test set" batch should be the same as defined in the batch builder
        batch = builder.batchify(data, task_test)
        self.assertEqual(builder.test_set_size, len(batch))

        # Calling with another task (same size) should generate a different batch
        batch_other = builder.batchify(data, task_test_other)
        self.assertNotEqual(batch, batch_other)

        # Calling again with the same task should generate the same batch
        batch_same = builder.batchify(data, task_test)
        self.assertEqual(batch, batch_same)

        # But a builder with another seed should generate a different batch
        builder_other = BatchBuilder(mock_data_source, test_set_size, 4321)
        batch_other = builder_other.batchify(data, task_test)
        self.assertNotEqual(batch, batch_other)

        # Applying `batchify` on the batch builder's data source should generate the same sets as `build_for_task`
        expected_batch = builder.build_for_task(task_test)
        actual_source = builder.batchify(builder.data_source.get_source_sentences(task_test.domain), task_test)
        actual_target = builder.batchify(builder.data_source.get_target_sentences(task_test.domain), task_test)
        self.assertEqual(expected_batch.source, actual_source)
        self.assertEqual(expected_batch.target, actual_target)

    def test_batchify_works_with_tensors(self):
        mock_data_source = _create_mock_domains_data_source(["dom1"], 500)
        builder = BatchBuilder(mock_data_source, 11, 1234)
        task_embedding = ScoreTask("dom1", 5, 4, 3, ScoreTask.EMBEDDING_SET)

        data = list(range(20))
        tensor = torch.tensor(data)

        expected = torch.tensor(builder.batchify(data, task_embedding))
        actual = builder.batchify(tensor, task_embedding)

        self.assertTrue(torch.equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
