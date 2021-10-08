"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-03-22
"""
import logging
from contextlib import closing
from multiprocessing import Process
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, NamedTuple, Optional, Callable, Tuple, Any

import sacrebleu
import torch
from bert_score import BERTScorer
from tqdm import tqdm

from couleurs.utils.multiprocessing import ProcessList, StoppableQueue, \
    StoppedException, ExtendedSyncManager
from .batch import BatchBuilder
from .score_task import ScoreTask, ScoreTaskList
from .translation import TranslationResult, ScoreTaskTranslator


class ScoreResult(NamedTuple):
    """A score result of a translation"""
    score_task: ScoreTask
    metric: str
    score: float


class ScorersPool(Process):
    """
    Pool of scoring processes (instances of `ScorerProcess`). Distributes `TranslationResult` from a TranslationResult
    queue to scorers.

    Scoring processes can be divided in "groups" consisting of one or multiple `ScorerProcess` processes. All processes
    in the same group share a same `TranslationResult` queue. Different groups have different queues. Once a new
    `TranslationResult` comes from the `translations` queue, it is distributed to the queue of each group.

    For example, we could have 3 `BLEUScorerProcess` (the "BLEU" group) and 1 `BERTScorerProcess` (the "BERT" group).
    When a new `TranslatorResult` is retrieved, it is sent to the queue of both queue. One of the `BLEUScorerProcess`
    will calculate the BLEU score while the `BERTScorerProcess` will calculate the BERTScore.
    """

    MAX_WAITING_JOBS = 50
    MAX_SCORES = 20

    def __init__(self, translations: StoppableQueue, manager: ExtendedSyncManager, shutdown_event: EventType):
        """
        :param manager: `ExtendedSyncManager` to create queues
        :param shutdown_event: Event that, once set, should stop this process
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._queues: Dict[str, StoppableQueue] = {}
        self._scorer_groups: Dict[str, Tuple[int, Callable[..., "ScorerProcess"], Tuple[Any]]] = {}
        self._shutdown_event = shutdown_event
        self._translations = translations
        self.scores = manager.StoppableQueue(self.MAX_SCORES, stop_event=shutdown_event)
        self._manager = manager

    def add_scorers_group(self, group_name: str, target: Callable[..., "ScorerProcess"], nb_processes: int, args=tuple()):
        """
        Add a new group of processes.

        The actual process instance will be created as child processes once the pool process starts, not before. So
        instead of an instance of the process, this method accepts a `target` (ex: the process class) and initialization
        arguments (`args`).

        `nb_processes` instances will be created (the `target` will be called this number of times, each time with the
        `args` arguments).

        :param group_name: Name of this group
        :param target: Callable (or class) to create each process
        :param nb_processes: Number of scorer processes to create
        :param args: Arguments to pass to `target` when called to create the process
        """
        if self.is_alive():
            # We can't add new scorers group, because self._queues and self._processes are not multiprocess dict, so if
            #   the process is started, we can't update those dictionaries in the subprocess
            raise ValueError("Can't add scorers group once the process is started.")

        group_queue = self._manager.StoppableQueue(self.MAX_WAITING_JOBS, stop_event=self._shutdown_event)
        self._scorer_groups[group_name] = (nb_processes, target, args)
        self._queues[group_name] = group_queue

    def _complete_all(self, processes: Dict[str, ProcessList["ScorerProcess"]]):
        for group_name, group_processes in processes.items():
            for _ in group_processes:
                self._queues[group_name].watching_put(None)

    def _start_scorers(self) -> Dict[str, ProcessList["ScorerProcess"]]:
        """
        Create scorer process instances of all groups and start them.

        :return: All started processes, grouped by their group name
        """
        processes = {}
        for group_name, (nb_processes, target, args) in self._scorer_groups.items():
            processes[group_name] = ProcessList()
            for process_number in range(nb_processes):
                instance: ScorerProcess = target(*args, name=f"ScorerProcess ({group_name}, #{process_number})")
                instance.integrate(self._queues[group_name], self.scores, self._shutdown_event)
                processes[group_name].append(instance)
            processes[group_name].start()
        return processes

    def run(self):
        try:
            processes = self._start_scorers()
            translations = self._translations

            while not self._shutdown_event.is_set():
                try:
                    next_translation: TranslationResult = translations.watching_get()

                    # If we received None, it means there is no more translations to score. We send the None to each
                    #  scoring process
                    if next_translation is None:
                        self._complete_all(processes)
                        break

                    # We send the translation to score to each groups (even if it was None)
                    for group_queue in self._queues.values():
                        group_queue.watching_put(next_translation)

                except StoppedException:
                    break

            # If the shutdown_event was triggered, we stop all sub processes, else we wait for all of them to finish
            for group_processes in processes.values():
                if self._shutdown_event.is_set():
                    group_processes.stop()
                else:
                    try:
                        group_processes.join()
                    except StoppedException:
                        group_processes.stop()

            # If all sub processes stopped and the shutdown event is still not triggered, we add None to the `scores` queue
            if not self._shutdown_event.is_set():
                self.scores.watching_put(None, raise_exception=False)
        except KeyboardInterrupt:
            self._logger.warning("KeyboardInterrupt raised")


class ScorerProcess(Process):
    """
    Generic class for a scorer process
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger(self.name)
        self._shutdown_event: Optional[EventType] = None
        self._translations: Optional[StoppableQueue[TranslationResult]] = None
        self._scores: Optional[StoppableQueue[ScoreResult]] = None

    def integrate(self, translations: StoppableQueue[TranslationResult], scores: StoppableQueue[ScoreResult],
                  shutdown_event: EventType):
        """
        Method called by the ScorersPool to integrate this process into the pool.

        The `translations` is the queue where new translation results waiting for scoring will be put. `scores` is the
        queue where to output `ScoreResult` calculated. The `shutdown_event` is the event that, once triggered, should
        end this process.

        :param translations: Queue of `TranslationResult` to score
        :param scores: Queue where to put `ScoreResult` once calculated
        :param shutdown_event: `Event` that should stop this process if set
        """
        self._shutdown_event = shutdown_event
        self._translations = translations
        self._scores = scores

    @staticmethod
    def get_metric_name() -> str:
        raise NotImplementedError()

    def run(self):
        try:
            while not self._shutdown_event.is_set():
                try:
                    next_translation = self._translations.watching_get()

                    # If the `next_translation` is None, it means we don't have anymore scoring to do. We end the process
                    if next_translation is None:
                        break

                    self._process_translation(next_translation)
                except StoppedException:
                    break
        except KeyboardInterrupt:
            self._logger.warning("KeyboardInterrupt raised")
            # Do nothing else

    def _process_translation(self, translation: TranslationResult):
        """
        Abstract method that processes a `TranslationResult`

        :param translation: `TranslationResult` to process
        """
        raise NotImplementedError()


class BLEUScorerProcess(ScorerProcess):
    """
    Scorer process that calculates a BLEU score
    """

    @staticmethod
    def get_metric_name() -> str:
        return "BLEU"

    @staticmethod
    def _create_metric():
        return sacrebleu.BLEU()

    def _process_translation(self, translation: TranslationResult):
        metric = self._create_metric()
        bleu_score = metric.corpus_score(translation.translations, [translation.batch.target])
        metric_name = self.get_metric_name()
        self._logger.info(f"{metric_name} for {translation.score_task.key}: {bleu_score.score}")
        result = ScoreResult(translation.score_task, metric_name, bleu_score.score)
        self._scores.watching_put(result, False)


class NoopScorer(ScorerProcess):
    """
    Special scorer that does nothing, it only consume the translation result.

    It's used when we don't want to calculate any score, but we still want to
    keep the waiting translations queue empty.
    """

    @staticmethod
    def get_metric_name() -> str:
        return "No-OP"

    def _process_translation(self, translation: TranslationResult):
        # Do nothing
        pass


class CHRFScorerProcess(ScorerProcess):
    """
    Scorer process that calculates a chrF++ score
    """

    @staticmethod
    def get_metric_name() -> str:
        return "chrF++"

    @staticmethod
    def _create_metric():
        return sacrebleu.CHRF(word_order=2)

    def _process_translation(self, translation: TranslationResult):
        metric = self._create_metric()
        chrf_score = metric.corpus_score(translation.translations, [translation.batch.target])
        score = chrf_score.score
        metric_name = self.get_metric_name()
        self._logger.info(f"{metric_name} for {translation.score_task.key}: {score}")
        result = ScoreResult(translation.score_task, metric_name, score)
        self._scores.watching_put(result, False)


class TaskListBERTScorer:
    """
    Calculates BERTScore in a ScoreTaskList from already translated files.

    Updates all BERTScores in a ScoreTaskList by calculating the BERTScore for
    each task already translated.

    This class should be used after the ScoreTaskTranslator has already run.
    """
    SCORE_NAME = "BERTScore"

    def __init__(self, task_list: ScoreTaskList, batch_builder: BatchBuilder, translations_dir: Path,
                 manager: ExtendedSyncManager, device="cpu", verbose=False):
        """
        :param task_list: The task list to update
        :param batch_builder: Batch builder returning the source and target sentences
            for each task
        :param translations_dir: Directory containing the translated sentences for
            each task
        :param manager: Multiprocessing "extended" sync manager
        :param device: Device where to load the BERTScorer
        :param verbose: Show progress of scoring
        """
        self._task_list = task_list
        self._translations_dir = translations_dir
        self._batch_builder = batch_builder
        self._manager = manager
        self._verbose = verbose
        self._scorer = BERTScorer(lang="fr", device=device, nthreads=8, rescale_with_baseline=True)

    def run(self):
        """
        Starts the scoring for each incomplete task in the task list.

        For each incomplete task in the task list, calculates the BERTScore by
        averaging the BERTScore of each translated sentence for this task.

        The ScoreTaskList cache file will be periodically updated. Note that
        the update will happen in another process, so the task list passed to
        this instance won't be updated. Call task_list.reload() to update it
        from the cache file.

        Already scored tasks are not updated.

        :raises FileNotFoundError: If an incomplete task doesn't have its
            associated translation file.
        """
        stop_event = self._manager.Event()
        scores = self._manager.StoppableQueue(100, stop_event=stop_event)
        listen_process = self._task_list.listen(scores)
        listen_process.start()

        stats = self._task_list.stats
        try:
            with closing(tqdm(total=stats.incomplete)) as progress, TemporaryDirectory() as tmp_dir_name:
                tmp_dir = Path(tmp_dir_name)
                for task in self._task_list.incomplete_tasks:
                    progress.set_description(task.key)
                    sentences_scores = self._get_task_scores(task, tmp_dir)
                    task_score = sentences_scores.mean().item()
                    result = ScoreResult(task, self.SCORE_NAME, task_score)
                    scores.watching_put(result)
                    progress.update()
        except KeyboardInterrupt:
            stop_event.set()

        scores.watching_put(None)
        listen_process.join()

    def _get_task_scores(self, task: ScoreTask, cache_dir: Path) -> torch.Tensor:
        """
        For a ScoreTask, calculate and return the BERTScore per sentence.

        Given a ScoreTask, retrieves the already translated sentences for this task and calculates their BERTScore by
        comparing to the target sentences.

        Since bootstrap sets are used, a sentence may be found in multiple sets. For optimality, the ScoreTaskTranslator
        translates whole domain files for each embedding set instead of translating each bootstrap set independently.
        Thus each sentence is translated only one time, even if its present in multiple bootstrap sets.

        This BERTScore scorer uses the same improvement. It scores whole domain files for each embedding set and then
        retrieves only the sentence scores for the task's batch. Those "domain files" are exactly the sames generated by
        the ScoreTaskTranslator. This is done lazily: only when this method is called with a task that its domain
        file is scored.

        This method then loads the cached scores and returns only those for this task's batch as a Torch tensor.

        The calculated scores are saved in (and retrieved from) the `cache_dir`.

        :param task: ScoreTask
        :param cache_dir: Directory where to temporarily save the embedding set scores
        :return: Float torch tensor of shape (n,) where n is the number of sentences in this task's set
        """
        cache_path = cache_dir / f"{task.embedding_key}.pt"

        if cache_path.exists():
            embedding_set_scores = torch.load(cache_path)
        else:
            # We score all the translations for this task's embedding set'
            translations_path = ScoreTaskTranslator.generate_translation_path(self._translations_dir, task)

            if not translations_path.exists():
                raise FileNotFoundError(f"Translation file for a task not found: {translations_path}. Make sure you ran"
                                        f" `calculate_scores` before `bertscore`.")

            with open(translations_path, "r") as file:
                translations = [line.strip() for line in file.readlines()]

            target_sentences = self._batch_builder.data_source.get_target_sentences(task.domain)
            _, _, embedding_set_scores = self._scorer.score(translations, target_sentences, verbose=self._verbose)

            # We save in cache the calculated embedding_set_scores
            torch.save(embedding_set_scores, cache_path)

        return self._batch_builder.batchify(embedding_set_scores, task)
