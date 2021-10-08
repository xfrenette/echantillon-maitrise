import time
import unittest
from contextlib import contextmanager
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import List, Sequence, ContextManager, Optional
from unittest import mock
from unittest.mock import Mock

import numpy as np
import torch
from fairseq.data import Dictionary
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel, TransformerEncoder
from random_words import LoremIpsum
from random_words import RandomWords

from couleurs.domain_embedding import Embedder
from couleurs.evaluate.with_domain_embedding import DomainsDataSource, BatchBuilder, ScoreTaskList, DomainEmbedder, \
    ExperimentsRunner
from couleurs.utils.multiprocessing import ExtendedSyncManager
from couleurs.utils.multiprocessing_patch import apply


@contextmanager
def _generate_domain_files(domains: Sequence[str], seed: int) -> ContextManager[Path]:
    lipsum = LoremIpsum()
    rng = np.random.default_rng(seed)

    with TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)

        for domain in domains:
            nb_sentences = rng.integers(50, 100)
            for ext in (".en", ".fr"):
                content = lipsum.get_sentences_list(nb_sentences)
                with open(tmp_dir / (domain + ext), "w") as f:
                    f.write("\n".join(content) + "\n")

        yield tmp_dir


@contextmanager
def _mock_translation_model(special_token: Optional[str] = None, nb_regular_tokens=2, translation_delay=0.2):
    def mock_translate(sentences: List[str]) -> List[str]:
        time.sleep(translation_delay)
        return sentences

    with torch.no_grad(), mock.patch("couleurs.evaluate.with_domain_embedding.translation.TransformerModel") as mock_class:
        dictionary = Dictionary()
        words = RandomWords()
        for q in words.random_words(count=nb_regular_tokens):
            dictionary.add_symbol(q)

        if special_token is not None:
            dictionary.add_symbol(special_token)

        default_embeddings = torch.nn.Embedding(len(dictionary), 768)

        mock_instance = Mock(TransformerModel)
        mock_instance.encoder = Mock(TransformerEncoder)
        mock_instance.encoder.dictionary = dictionary
        mock_instance.encoder.embed_tokens = default_embeddings

        mock_generator = Mock(GeneratorHubInterface)
        mock_generator.models = [mock_instance]
        mock_generator.translate.side_effect = mock_translate

        mock_class.from_pretrained.return_value = mock_generator

        yield


def _trigger_shutdown_event(shutdown_event: EventType, delay: float):
    time.sleep(delay)
    shutdown_event.set()


class _FalseEmbedder(Embedder):
    def calculate(self, sentences: List[str]):
        return torch.rand(768)


class ExperimentsRunnerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        apply()

    def test_functional(self):
        """The whole process works without errors (but we mock translation models)"""
        seed = 1984
        special_token = "<dom_-X->"

        with _generate_domain_files(["dom1", "dom2"], seed) as tmp_dir, _mock_translation_model(special_token, 100, 0.1), \
                ExtendedSyncManager() as manager:
            shutdown_event = manager.Event()
            scores_pickle_path = tmp_dir / "scores.pkl"
            domains_data = DomainsDataSource(tmp_dir)
            batch_builder = BatchBuilder(domains_data, 35, seed)
            domain_embedder = DomainEmbedder(_FalseEmbedder(), batch_builder, tmp_dir / "embbed_cache.pkl")
            tasks_list = ScoreTaskList.create(domains_data, [10, 100], 2, 2, ["BLEU"])
            tasks_list.cache_file = scores_pickle_path
            translations_dir = tmp_dir / "translations"
            # noinspection PyTypeChecker
            runner = ExperimentsRunner(manager, shutdown_event, batch_builder, domain_embedder, tmp_dir / "path.pt",
                                       {"config": True}, special_token, None, translations_dir, 4)
            runner.run(tasks_list)

            # We create another task list from the saved scores of `task_list`. Both should be equal
            new_tasks_list = ScoreTaskList.from_file(scores_pickle_path)
            tasks_list.reload()
            self.assertEqual(tasks_list, new_tasks_list)

    def test_functional_no_embedder(self):
        """The whole process works without errors even without an embedder"""
        seed = 2001

        with _generate_domain_files(["dom1", "dom2"], seed) as tmp_dir, _mock_translation_model(None, 100, 0.1), \
                ExtendedSyncManager() as manager:
            shutdown_event = manager.Event()
            scores_pickle_path = tmp_dir / "scores.pkl"
            domains_data = DomainsDataSource(tmp_dir)
            batch_builder = BatchBuilder(domains_data, 35, seed)
            tasks_list = ScoreTaskList.create(domains_data, [10, 100], 2, 2, ["BLEU"])
            tasks_list.cache_file = scores_pickle_path
            translations_dir = tmp_dir / "translations"
            # noinspection PyTypeChecker
            runner = ExperimentsRunner(manager, shutdown_event, batch_builder, None, tmp_dir / "path.pt",
                                       {"config": True}, None, None, translations_dir, 4)
            runner.run(tasks_list)

            # We create another task list from the saved scores of `task_list`. Both should be equal
            new_tasks_list = ScoreTaskList.from_file(scores_pickle_path)
            tasks_list.reload()
            self.assertEqual(tasks_list, new_tasks_list)

    def test_shutdown_stops_all(self):
        """Setting the shutdown event stops the runner"""
        seed = 1984
        special_token = "<dom_-X->"
        # We make translators that take a long time
        translation_delay = 1

        with _generate_domain_files(["dom1", "dom2"], seed) as tmp_dir, _mock_translation_model(special_token, 100, translation_delay), \
                ExtendedSyncManager() as manager:
            shutdown_event = manager.Event()
            domains_data = DomainsDataSource(tmp_dir)
            batch_builder = BatchBuilder(domains_data, 35, seed)
            domain_embedder = DomainEmbedder(_FalseEmbedder(), batch_builder)
            tasks_list = ScoreTaskList.create(domains_data, [10, 100], 2, 2, ["BLEU"])
            # noinspection PyTypeChecker
            runner = ExperimentsRunner(manager, shutdown_event, batch_builder, domain_embedder, tmp_dir / "path.pt",
                                       {"config": True}, special_token, None, None, 4)

            runner_thread = Thread(target=runner.run, args=(tasks_list,), daemon=True)
            runner_thread.start()

            # We trigger the stop event after a delay
            time.sleep(0.4)
            shutdown_event.set()

            # The runner should end quickly after that
            runner_thread.join(3)
            self.assertFalse(runner_thread.is_alive())


if __name__ == '__main__':
    unittest.main()
