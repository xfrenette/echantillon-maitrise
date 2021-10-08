import queue
import time
import unittest
from contextlib import contextmanager
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ContextManager, List, cast, Callable
from unittest.mock import patch, Mock

import numpy as np
from sacrebleu.metrics import BLEU, BLEUScore, CHRF, CHRFScore
import torch
from bert_score import BERTScorer
from random_words import LoremIpsum

from couleurs.evaluate.with_domain_embedding import BLEUScorerProcess, ScoreResult, TranslationResult, ScoreTask, \
    Batch, ScorersPool, ScorerProcess, ScoreTaskList, BatchBuilder, TaskListBERTScorer, CHRFScorerProcess, \
    DomainsDataSource, ScoreTaskTranslator
from couleurs.tests.evaluate.with_domain_embedding.test_score_task import ScoreTaskListTestCase
from couleurs.utils.multiprocessing import ExtendedSyncManager, StoppableQueue

_rng = np.random.default_rng()


@contextmanager
def _mock_BLEU(delay=0.0) -> ContextManager[BLEU]:
    def mock_corpus_score(sys, refs):
        if delay > 0:
            time.sleep(delay)
        mock_result = Mock(BLEUScore)
        mock_result.score = 0.324
        return mock_result

    mock_bleu_instance = Mock(BLEU)
    mock_bleu_instance.corpus_score.side_effect = mock_corpus_score

    with patch("sacrebleu.BLEU") as mock_bleu:
        mock_bleu.return_value = mock_bleu_instance
        yield mock_bleu


@contextmanager
def _mock_CHRF(delay=0.0) -> ContextManager[CHRF]:
    def mock_corpus_score(sys, refs):
        if delay > 0:
            time.sleep(delay)
        mock_result = Mock(CHRFScore)
        mock_result.score = 0.781
        return mock_result

    mock_chrf_instance = Mock(CHRF)
    mock_chrf_instance.corpus_score.side_effect = mock_corpus_score

    with patch("sacrebleu.CHRF") as mock_chrf:
        mock_chrf.return_value = mock_chrf_instance
        yield mock_chrf


def _calculate_bleu(translation_result: TranslationResult) -> ScoreResult:
    metric = BLEU()
    score = metric.corpus_score(translation_result.translations, [translation_result.batch.target])
    return ScoreResult(translation_result.score_task, BLEUScorerProcess.get_metric_name(), score.score)


def _calculate_chrF(translation_result: TranslationResult) -> ScoreResult:
    metric = CHRF(word_order=2)
    score = metric.corpus_score(translation_result.translations, [translation_result.batch.target])
    return ScoreResult(translation_result.score_task, CHRFScorerProcess.get_metric_name(), score.score)


def _create_mock_translation_result(nb_sentences=10) -> TranslationResult:
    lipsum = LoremIpsum()
    task = ScoreTask("test", 100,  0, 0, ScoreTask.TEST_SET)
    batch = Batch(lipsum.get_sentences_list(nb_sentences), lipsum.get_sentences_list(nb_sentences))
    translations = lipsum.get_sentences_list(nb_sentences)
    return TranslationResult(task, batch, translations)


def _create_mock_score_result(score=np.NAN, metric="BERTScore") -> ScoreResult:
    domain = _rng.choice(["dom1", "dom2", "dom3", "dom4"])
    embedding_set_size = _rng.choice([10, 100, 1000, 500, 200, 400])
    embedding_set_id = _rng.choice(50)
    set_type = _rng.choice([ScoreTask.TEST_SET, ScoreTask.EMBEDDING_SET])
    set_id = _rng.choice(50)
    task = ScoreTask(domain, embedding_set_size, embedding_set_id, set_id, set_type)
    return ScoreResult(task, metric, score)


def _trigger_shutdown_event(event: EventType, timeout: float):
    def trigger(event: EventType, timeout: float):
        time.sleep(timeout)
        event.set()

    p = Process(target=trigger, args=(event, timeout))
    p.start()


class BLEUScorerProcessTestCase(unittest.TestCase):
    def test_stops_with_none(self):
        """Test that the process stops once it receives None"""
        with _mock_BLEU(1):
            shutdown_event = Event()
            translations = StoppableQueue(stop_event=shutdown_event)
            scores = StoppableQueue(stop_event=shutdown_event)
            scorer = BLEUScorerProcess()
            scorer.integrate(translations, scores, shutdown_event)
            scorer.start()

            # We add a first translation result
            translations.watching_put(_create_mock_translation_result())

            # We add a None, which should end the process
            translations.watching_put(None)

            # The scorer should end before the 3 seconds
            scorer.join(3)

            self.assertFalse(scorer.is_alive())

    def test_calculates_score(self):
        """Test that the BLEU score is calculated and outputted on the score queue"""
        shutdown_event = Event()
        translations = StoppableQueue(stop_event=shutdown_event)
        scores = StoppableQueue(stop_event=shutdown_event)
        scorer = BLEUScorerProcess()
        scorer.integrate(translations, scores, shutdown_event)
        scorer.start()

        # The translation we will put on the scoring queue and for which we will also validate the BLEU scores
        translation_result = _create_mock_translation_result()
        expected_score = _calculate_bleu(translation_result)

        # We add the translation result, and then a None to stop the process
        translations.watching_put(translation_result)
        translations.watching_put(None)

        # The scorer should end before the 3 seconds
        scorer.join(3)

        # The scores queue should have a ScoreResult equal to the expected one
        try:
            result = scores.get(timeout=1)
            self.assertEqual(expected_score, result)
        except queue.Empty:
            self.fail("The scores queue is empty")

    def test_stops_with_shutdown_event(self):
        """Test that calculation of BLEU score stops when the shutdown event is triggered"""
        with _mock_BLEU(0.5):
            shutdown_event = Event()
            translations = StoppableQueue(stop_event=shutdown_event)
            scores = StoppableQueue(stop_event=shutdown_event)
            scorer = BLEUScorerProcess()
            scorer.integrate(translations, scores, shutdown_event)
            scorer.start()

            # We add some translation results that should take at least 5 seconds to complete
            for _ in range(10):
                translations.watching_put(_create_mock_translation_result())

            # We trigger the shutdown event after a timeout
            _trigger_shutdown_event(shutdown_event, 0.8)

            # The scorer should end before the 2 seconds
            scorer.join(2)

            self.assertFalse(scorer.is_alive())


class CHRFScorerProcessTestCase(unittest.TestCase):
    def test_stops_with_none(self):
        """Test that the process stops once it receives None"""
        with _mock_CHRF(1):
            shutdown_event = Event()
            translations = StoppableQueue(stop_event=shutdown_event)
            scores = StoppableQueue(stop_event=shutdown_event)
            scorer = CHRFScorerProcess()
            scorer.integrate(translations, scores, shutdown_event)
            scorer.start()

            # We add a first translation result
            translations.watching_put(_create_mock_translation_result())

            # We add a None, which should end the process
            translations.watching_put(None)

            # The scorer should end before the 3 seconds
            scorer.join(3)

            self.assertFalse(scorer.is_alive())

    def test_calculates_score(self):
        """Test that the chrF++ score is calculated and outputted on the score queue"""
        shutdown_event = Event()
        translations = StoppableQueue(stop_event=shutdown_event)
        scores = StoppableQueue(stop_event=shutdown_event)
        scorer = CHRFScorerProcess()
        scorer.integrate(translations, scores, shutdown_event)
        scorer.start()

        # The translation we will put on the scoring queue and for which we will also validate the BLEU scores
        translation_result = _create_mock_translation_result()
        expected_score = _calculate_chrF(translation_result)

        # We add the translation result, and then a None to stop the process
        translations.watching_put(translation_result)
        translations.watching_put(None)

        # The scorer should end before the 3 seconds
        scorer.join(3)

        # The scores queue should have a ScoreResult equal to the expected one
        try:
            result = scores.get(timeout=1)
            self.assertEqual(expected_score, result)
        except queue.Empty:
            self.fail("The scores queue is empty")

    def test_stops_with_shutdown_event(self):
        """Test that calculation of BLEU score stops when the shutdown event is triggered"""
        with _mock_CHRF(0.5):
            shutdown_event = Event()
            translations = StoppableQueue(stop_event=shutdown_event)
            scores = StoppableQueue(stop_event=shutdown_event)
            scorer = CHRFScorerProcess()
            scorer.integrate(translations, scores, shutdown_event)
            scorer.start()

            # We add some translation results that should take at least 5 seconds to complete
            for _ in range(10):
                translations.watching_put(_create_mock_translation_result())

            # We trigger the shutdown event after a timeout
            _trigger_shutdown_event(shutdown_event, 0.8)

            # The scorer should end before the 2 seconds
            scorer.join(2)

            self.assertFalse(scorer.is_alive())


class MockScorerProcess(ScorerProcess):
    def __init__(self, mock_score: float, metric_name: str, delay=0.0):
        super(MockScorerProcess, self).__init__()
        self._mock_score = mock_score
        self._delay = delay
        self._metric_name = metric_name

    @staticmethod
    def get_metric_name() -> str:
        return "_Mock_"

    def _process_translation(self, translation: TranslationResult):
        if self._delay > 0:
            time.sleep(self._delay)
        result = ScoreResult(translation.score_task, self._metric_name, self._mock_score)
        self._scores.watching_put(result, False)


class ScorersPoolTestCase(unittest.TestCase):
    def test_calls_groups(self):
        """
        Test that scorers sub processes are correctly called by the pool.

        Scorer sub processes will be mocked and will generate a specific score that will correctly indicate which
        process was called.

        In each process group, all process have the same base score, except for a difference of one between them. So,
        for a specific group, the base score can be 12.1 and then each process of the group will generate a score of
        12.1, 13.1, 14.1, ... That way, we can easily distinguish processes in a group and between groups.

        So we will create 2 groups, one with 2 processes, one with one process. We will send them translation results
        and we compare the scores we received with what we expect based on the algorithm above.
        """
        manager = ExtendedSyncManager()
        manager.start()

        try:
            shutdown_event = manager.Event()
            translations_queue = manager.StoppableQueue(2, stop_event=shutdown_event)

            group1_mock_score = 12.1
            group2_mock_score = 53.8

            scorers_count = {}

            def create_mock_scorer(base_score: float, metric_name: str, *args, **kwargs):
                if base_score in scorers_count:
                    scorers_count[base_score] += 1
                else:
                    scorers_count[base_score] = 0

                return MockScorerProcess(base_score + scorers_count[base_score], metric_name, 1)

            # First group will have 2 scorers, second just one
            pool = ScorersPool(translations_queue, manager, shutdown_event)
            pool.add_scorers_group("group1", create_mock_scorer, 2, (group1_mock_score, "metric1"))
            pool.add_scorers_group("group2", create_mock_scorer, 1, (group2_mock_score, "metric2"))

            pool.start()

            # We add 2 translation results to the queue
            translation_result1 = _create_mock_translation_result()
            translation_result2 = _create_mock_translation_result()
            translations_queue.put(translation_result1)
            translations_queue.put(translation_result2)

            # We end the translations
            translations_queue.put(None)

            # The pool should end before 3 seconds
            pool.join(3)
            self.assertFalse(pool.is_alive())

            # We compare the expected scores of the queue with what we actually receive
            expected_scores = [
                ScoreResult(translation_result1.score_task, "metric1", group1_mock_score),
                ScoreResult(translation_result1.score_task, "metric1", group1_mock_score + 1),
                ScoreResult(translation_result1.score_task, "metric2", group2_mock_score),
                ScoreResult(translation_result1.score_task, "metric2", group2_mock_score),
            ]

            actual_scores = []
            while not pool.scores.empty():
                score = pool.scores.get()
                if score is not None:
                    actual_scores.append(score)

            # Important to sort both expected and actual scores, since the processes might not process them in order
            def score_result_key(s: ScoreResult):
                return f"{s.score_task.embedding_key}_{s.metric}_{s.score}"

            actual_scores.sort(key=score_result_key)
            expected_scores.sort(key=score_result_key)

            self.assertListEqual(expected_scores, actual_scores)

        finally:
            manager.shutdown()

    def test_shutdown_terminates(self):
        """Test that the shutdown event terminates even long running scorers"""
        manager = ExtendedSyncManager()
        manager.start()

        try:
            with _mock_BLEU(20):
                shutdown_event = manager.Event()
                translations_queue = manager.StoppableQueue(stop_event=shutdown_event)

                pool = ScorersPool(translations_queue, manager, shutdown_event)
                pool.add_scorers_group("group", BLEUScorerProcess, 1)
                pool.start()

                # We add a translation results to the queue
                translations_queue.put(_create_mock_translation_result())

                # We trigger the shutdown event in a small delay
                _trigger_shutdown_event(shutdown_event, 1)

                # Pool should end quickly, well before 2sec
                pool.join(2)

                self.assertFalse(pool.is_alive())

        finally:
            manager.shutdown()

    def test_functional(self):
        """A functional test making sure everything works without error. Also, None should be outputted on the scores queue."""
        manager = ExtendedSyncManager()
        manager.start()

        try:
            shutdown_event = manager.Event()
            translations_queue = manager.StoppableQueue(stop_event=shutdown_event)

            pool = ScorersPool(translations_queue, manager, shutdown_event)
            pool.add_scorers_group("group1", BLEUScorerProcess, 1)
            pool.start()

            # We add 2 translation results to the queue
            translations_queue.put(_create_mock_translation_result())
            translations_queue.put(_create_mock_translation_result())
            translations_queue.put(None)

            pool.join()

            # We should have 2 results on the scores queue and a final None
            score = pool.scores.get(timeout=0.5)
            self.assertIsInstance(score, ScoreResult)
            score = pool.scores.get(timeout=0.5)
            self.assertIsInstance(score, ScoreResult)
            score = pool.scores.get(timeout=0.5)
            self.assertIsNone(score)
        finally:
            manager.shutdown()


class TaskListBERTScorerTestCase(unittest.TestCase):
    def test_calculates_score(self):
        lipsum = LoremIpsum()
        nb_sentences = 100
        base_source = lipsum.get_sentences_list(nb_sentences)
        base_target = lipsum.get_sentences_list(nb_sentences)
        scores_cache = {}

        def mock_bertscorer_score(cands, refs, *args, **kwargs):
            bertscores = []
            for candidate, ref in zip(cands, refs):
                key = f"{candidate} / {ref}"
                if key not in scores_cache:
                    scores_cache[key] = _rng.random()
                bertscores.append(scores_cache[key])
            return torch.rand(len(cands)), torch.rand(len(cands)), torch.tensor(bertscores)

        def expected_score_for_score_result(translations: List[str], result: ScoreResult, batch_builder: BatchBuilder) -> float:
            # `translations` is all the translations generated for the task's embedding set. The translations for
            # this task are just a sub-batch of all those translations. We get the translations for this task (with
            # `batchify()`) and then only score those.
            task = result.score_task
            task_translations = batch_builder.batchify(translations, task)
            batch = batch_builder.build_for_task(task)
            _, _, scores = mock_bertscorer_score(task_translations, batch.target)
            return scores.mean().item()

        def mock_get_sentences(data: List[str]) -> Callable:
            def func(domain):
                return [f"{domain} {s}" for s in data]
            return func

        with TemporaryDirectory() as tmp_dir_name, ExtendedSyncManager() as manager, \
                patch("couleurs.evaluate.with_domain_embedding.scoring.BERTScorer") as MockBERTScorer:
            manager = cast(ExtendedSyncManager, manager)  # cast(): only used for typing validation
            mock_scorer = Mock(BERTScorer)
            mock_scorer.score.side_effect = mock_bertscorer_score
            MockBERTScorer.return_value = mock_scorer

            tmp_dir = Path(tmp_dir_name)
            translations_dir = tmp_dir / "translations"
            translations_dir.mkdir()

            mock_data_source = Mock(DomainsDataSource)
            mock_data_source.get_source_sentences.side_effect = mock_get_sentences(base_source)
            mock_data_source.get_target_sentences.side_effect = mock_get_sentences(base_target)

            batch_builder = BatchBuilder(mock_data_source, 50, 1414)

            task_cache_file = tmp_dir / "scores.pkl"
            task_list = ScoreTaskList(task_cache_file)

            tasks = [
                ScoreTask("dom1", 62, 0, 0, ScoreTask.EMBEDDING_SET),
                # Same embedding set as above
                ScoreTask("dom1", 62, 0, 1, ScoreTask.EMBEDDING_SET),
                ScoreTask("dom2", 10, 0, 1, ScoreTask.EMBEDDING_SET),
                ScoreTask("dom2", 37, 0, 1, ScoreTask.TEST_SET),
            ]

            results = [
                ScoreResult(tasks[0], "BERTScore", np.NaN),
                ScoreResult(tasks[1], "BERTScore", np.NaN),
                ScoreResult(tasks[2], "BERTScore", np.NaN),
                # Already scored, should not be updated
                ScoreResult(tasks[3], "BERTScore", 1.1),
            ]
            expected_results = []

            for result in results:
                # Add the result to the task list
                task_list.update(result)

                # The translator outputs translations per embedding set (not per task). So we generate translations
                # for the task's embedding set (if it doesn't exist yet). In both cases, we then retrieve the content
                # of the file.
                file_path = ScoreTaskTranslator.generate_translation_path(translations_dir, result.score_task)
                if not file_path.exists():
                    file_content = lipsum.get_sentences_list(nb_sentences)
                    with open(file_path, "w") as file:
                        file.write("\n".join(file_content) + "\n")
                else:
                    with open(file_path, "r") as file:
                        file_content = file.read().strip().split("\n")

                # If the task is not yet scored (its score is np.NaN), we calculate what is the score we expect to be
                # calculated. If the task is already scored, its score is not supposed to change
                result_score = result.score
                if np.isnan(result_score):
                    result_score = expected_score_for_score_result(file_content, result, batch_builder)

                # We create a ScoreResult with the expected score
                expected_result = ScoreResult(result.score_task, result.metric, result_score)
                expected_results.append(expected_result)

            # We run the scorer
            scorer = TaskListBERTScorer(task_list, batch_builder, translations_dir, manager)
            scorer.run()

            # The scores in the task list should now be updated
            task_list.reload()
            stats = task_list.stats
            self.assertEqual(len(expected_results), stats.total)
            self.assertEqual(0, stats.incomplete)

            # The score results in the task list should now be equal to what we expected
            list_asserter = ScoreTaskListTestCase()
            for i, expected_result in enumerate(expected_results):
                list_asserter.assertScoreInList(task_list, expected_result)


if __name__ == '__main__':
    unittest.main()
