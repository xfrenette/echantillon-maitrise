import queue
import random
import unittest
from contextlib import contextmanager
from multiprocessing import Event
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ContextManager, Tuple, List, Optional
from unittest import mock
from unittest.mock import Mock

import torch
from fairseq.data import Dictionary
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel, TransformerEncoder
from random_words import LoremIpsum

from couleurs.evaluate.with_domain_embedding import BatchBuilder, Batch, DomainEmbedder, ScoreTask, TranslationResult, \
    DomainsDataSource
from couleurs.evaluate.with_domain_embedding.translation import ScoreTaskTranslator, ScoreTaskTranslatorClosedException
from couleurs.utils.multiprocessing import ExtendedSyncManager, StoppableQueue
from couleurs.fairseq_plugins.transformer import CouleursTransformerModel, CouleursTransformerEncoder, EmbeddingsCombiner


def _mock_batch_builder() -> BatchBuilder:
    lipsum = LoremIpsum()
    source_sentences = lipsum.get_sentences_list(400)
    target_sentences = lipsum.get_sentences_list(400)

    def mock_get_source_sentences(domain: str):
        return [f"{domain} {s}" for s in source_sentences]

    def mock_get_target_sentences(domain: str):
        return [f"{domain} {s}" for s in target_sentences]

    mock_data_source = Mock(DomainsDataSource)
    mock_data_source.get_source_sentences.side_effect = mock_get_source_sentences
    mock_data_source.get_target_sentences.side_effect = mock_get_target_sentences

    return BatchBuilder(mock_data_source, 50)


def _mock_domain_embedder(repeatable=True, dim=768) -> DomainEmbedder:
    embeddings = {}

    def mock_calculate_for_task(task: ScoreTask) -> Batch:
        key = task.key
        if not repeatable or key not in embeddings:
            embeddings[key] = torch.rand(dim)
        return embeddings[key]

    mock_embedder = Mock(DomainEmbedder)
    mock_embedder.calculate_for_task.side_effect = mock_calculate_for_task
    return mock_embedder


def _mock_manager() -> ExtendedSyncManager:
    def return_queue(maxsize=0, *, stop_event):
        return StoppableQueue(maxsize, stop_event=stop_event)

    mock = Mock(ExtendedSyncManager)
    mock.StoppableQueue.side_effect = return_queue
    return mock


@contextmanager
def _create_mocks(embedder_dim=768) -> ContextManager[Tuple[ExtendedSyncManager, Event, BatchBuilder, DomainEmbedder]]:
    shutdown_event = Event()
    yield _mock_manager(), shutdown_event, _mock_batch_builder(), _mock_domain_embedder(dim=embedder_dim)


@contextmanager
def _mock_transformer(special_token: Optional[str] = None, nb_regular_tokens=50, couleurs_instances=False,
                      random_translations=False
                      ) -> ContextManager[Tuple[GeneratorHubInterface, TransformerModel, TransformerModel]]:
    def mock_translate(sentences: List[str]) -> List[str]:
        rand_token = f" rand-{random.randint(1, 99999)}" if random_translations else ""
        translations = [f"trans: {s}{rand_token}" for s in sentences]
        return translations

    with torch.no_grad(), \
            mock.patch("couleurs.evaluate.with_domain_embedding.translation.TransformerModel") as mock_class:
        dictionary = Dictionary()
        for i in range(nb_regular_tokens):
            dictionary.add_symbol(f"sample{i+1}")

        if special_token is not None:
            dictionary.add_symbol(special_token)

        default_embeddings = torch.nn.Embedding(len(dictionary), 768)

        mock_instance = Mock(CouleursTransformerModel) if couleurs_instances else Mock(TransformerModel)
        mock_instance.encoder = Mock(CouleursTransformerEncoder) if couleurs_instances else Mock(TransformerEncoder)
        mock_instance.encoder.dictionary = dictionary
        mock_instance.encoder.embed_tokens = default_embeddings

        if couleurs_instances:
            mock_instance.encoder.embeddings_combiner = EmbeddingsCombiner(768, foreign_dim=1024)

        mock_generator = Mock(GeneratorHubInterface)
        mock_generator.models = [mock_instance]
        mock_generator.translate.side_effect = mock_translate

        mock_class.from_pretrained.return_value = mock_generator

        yield mock_generator, mock_class, mock_instance


def _empty_queue(queue_: StoppableQueue):
    while True:
        try:
            queue_.get(timeout=0.1)
        except queue.Empty:
            break


def _extract_translation_results(translations: StoppableQueue) -> List[TranslationResult]:
    results = []
    while True:
        result = translations.get()
        if result is None:
            break
        results.append(result)
    return results


class ScoreTaskTranslatorTestCase(unittest.TestCase):
    def test_translates(self):
        """The translation model is called with expected source"""
        special_token = "<special>"
        tasks = [
            # Tasks of embedding 1
            ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET),
            ScoreTask("dom1", 100, 0, 0, ScoreTask.TEST_SET),
            # Task of embedding 2
            ScoreTask("dom1", 100, 1, 0, ScoreTask.TEST_SET),
            # Task of embedding 3
            ScoreTask("dom2", 100, 0, 0, ScoreTask.TEST_SET),
        ]

        with _create_mocks() as (manager, shutdown_event, batch_builder, domain_embedder), \
                _mock_transformer(special_token) as (mock_generator, _, _):
            translator = ScoreTaskTranslator(batch_builder, domain_embedder, shutdown_event, manager, Path("model.pt"),
                                             {}, special_token)
            for task in tasks:
                translator.translate(task)

            # We finalize all translations
            translator.close()

            i = 0
            while True:
                result: TranslationResult = translator.translations.get()
                if result is None:
                    self.assertEqual(len(tasks), i)
                    break
                expected = mock_generator.translate([f"{special_token} {s}" for s in result.batch.source])
                self.assertEqual(expected, result.translations)
                i += 1

    def test_no_special_token_in_source_sentence(self):
        """The source sentence to translate shouldn't have a special token if none is supplied"""
        tasks = [
            ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET),
            ScoreTask("dom1", 100, 0, 0, ScoreTask.TEST_SET),
        ]

        with _create_mocks() as (manager, shutdown_event, batch_builder, _), \
                _mock_transformer() as (mock_generator, _, _):
            translator = ScoreTaskTranslator(batch_builder, None, shutdown_event, manager, Path("model.pt"),
                                             {}, None)
            for task in tasks:
                translator.translate(task)

            # We finalize all translations
            translator.close()

            i = 0
            while True:
                result: TranslationResult = translator.translations.get()
                if result is None:
                    self.assertEqual(len(tasks), i)
                    break
                expected = mock_generator.translate(result.batch.source)
                self.assertEqual(expected, result.translations)
                i += 1

    def test_uses_cache(self):
        """
        The translations are saved in a cache

        We create 2 translation tasks for the same embedding set. If the cache is correctly used, the translation
        model will run only one time. To ensure this, we set up the mock translation model to generate a random token
        each time it's used. So, if everything works correctly, both time the translations will have the same token.
        """
        with TemporaryDirectory() as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            tasks = [
                ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET),
                ScoreTask("dom1", 100, 0, 0, ScoreTask.TEST_SET),
            ]
            special_token = "<special>"

            with _create_mocks() as (manager, shutdown_event, batch_builder, domain_embedder), \
                    _mock_transformer(special_token, random_translations=True):
                # The mock translation model is configured to add a random token to the translations each time it's
                # called. If everything works correctly, if we do 2 times the translation of the same tasks, their
                # translations should not change

                translations = []

                for _ in range(2):
                    translator = ScoreTaskTranslator(batch_builder, domain_embedder, shutdown_event, manager,
                                                     Path("model.pt"), {}, special_token, output_dir=output_dir)
                    for task in tasks:
                        translator.translate(task)

                    translator.close()

                    # `translations` is a list of list. Each element contains a list of the task translations
                    # Later on, we will test that both lists contain the same translations
                    translations.append([result.translations for result in _extract_translation_results(translator.translations)])

                # The same task in each run should have the same translations
                self.assertListEqual(translations[0], translations[1])

    def test_uses_model_config(self):
        """The translation model is passed the correct configurations"""
        special_token = "<special>"
        model_config = {"test": True, "other": "value"}
        model_path = Path("/path/to/model.pt")
        with _create_mocks() as (manager, shutdown_event, batch_builder, domain_embedder), \
                _mock_transformer(special_token) as (_, mock_class, _):
            translator = ScoreTaskTranslator(batch_builder, domain_embedder, shutdown_event, manager, model_path,
                                             model_config, special_token)
            translator.translate(ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET))
            translator.close()

            # Empty the queue
            _extract_translation_results(translator.translations)

            mock_class.from_pretrained.assert_called_with("/path/to", "model.pt", ".", **model_config)

    def test_updates_model_special_token(self):
        """The model's special token is updated with the correct embedding (for models with special tokens)"""
        special_token = "<special>"
        tasks = [
            ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET),
            ScoreTask("dom2", 50, 2, 3, ScoreTask.TEST_SET),
        ]
        with _create_mocks() as (manager, shutdown_event, batch_builder, domain_embedder), \
                _mock_transformer(special_token) as (_, _, mock_model):
            special_token_index = mock_model.encoder.dictionary.index(special_token)
            translator = ScoreTaskTranslator(batch_builder, domain_embedder, shutdown_event, manager, Path("model.pt"),
                                             {}, special_token)
            for task in tasks:
                translator.translate(task)
                expected = domain_embedder.calculate_for_task(task)
                actual = mock_model.encoder.embed_tokens.weight[special_token_index]
                self.assertTrue(torch.equal(expected, actual))
            _empty_queue(translator.translations)

    def test_updates_model_foreign_embedding(self):
        """The model's foreign embedding is updated with the correct embedding (for models with foreign embeddings)"""
        special_token = "<special>"
        tasks = [
            ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET),
            ScoreTask("dom2", 50, 2, 3, ScoreTask.TEST_SET),
        ]
        with _create_mocks(embedder_dim=1024) as (manager, shutdown_event, batch_builder, domain_embedder), \
                _mock_transformer(special_token, couleurs_instances=True) as (_, _, mock_model):
            special_token_index = mock_model.encoder.dictionary.index(special_token)
            translator = ScoreTaskTranslator(batch_builder, domain_embedder, shutdown_event, manager, Path("model.pt"),
                                             {}, special_token, "foreign")
            for task in tasks:
                translator.translate(task)
                expected = domain_embedder.calculate_for_task(task)
                actual = mock_model.encoder.embeddings_combiner.foreign_embeddings[special_token_index]
                self.assertTrue(torch.equal(expected, actual))
            _empty_queue(translator.translations)

    def test_works_without_domain_embedder(self):
        """Creating a translator without a domain embedder and special token still works (no update is made)"""
        tasks = [
            ScoreTask("dom1", 100, 0, 0, ScoreTask.TEST_SET),
            ScoreTask("dom2", 50, 2, 3, ScoreTask.TEST_SET),
        ]
        with _create_mocks() as (manager, shutdown_event, batch_builder, domain_embedder), \
                _mock_transformer() as (_, _, mock_model):
            translator = ScoreTaskTranslator(batch_builder, None, shutdown_event, manager, Path("model.pt"),
                                             {}, None)
            for task in tasks:
                translator.translate(task)
                # Should not raise any exception
            _empty_queue(translator.translations)

    def test_close(self):
        """Calling close() stops all translations"""
        special_token = "<special>"
        with _create_mocks() as (manager, shutdown_event, batch_builder, domain_embedder), \
                _mock_transformer(special_token):
            translator = ScoreTaskTranslator(batch_builder, domain_embedder, shutdown_event, manager, Path("model.pt"),
                                             {}, special_token)
            translator.translate(ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET))
            translator.close()
            with self.assertRaises(ScoreTaskTranslatorClosedException):
                translator.translate(ScoreTask("dom1", 100, 0, 0, ScoreTask.EMBEDDING_SET))

            _empty_queue(translator.translations)


if __name__ == '__main__':
    unittest.main()
