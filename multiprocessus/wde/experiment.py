"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-03-15
"""
from multiprocessing.context import Process as ProcessType
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import tqdm

from couleurs.utils.multiprocessing import ExtendedSyncManager, StoppedException
from . import ScoreTaskList, DomainEmbedder, BatchBuilder, ScorersPool, BLEUScorerProcess, ScoreTaskTranslator, \
    CHRFScorerProcess, NoopScorer


class ExperimentsRunner:
    """
    Class that runs a set of experiments.

    An experiment is the process of translating for a task and scoring the translation with different metrics.
    """

    def __init__(self, manager: ExtendedSyncManager, shutdown_event: EventType, batch_builder: BatchBuilder,
                 embedder: Union[DomainEmbedder, None], base_model_path: Path, model_config: Dict,
                 special_token: Optional[str] = None, embeddings_integration: Optional[str] = None,
                 translations_dir: Optional[Path] = None, n_BLEU_jobs=1, n_chrF_jobs=1):
        """
        See `ScoreTaskTranslator` for documentation on `embeddings_integration`.
        """
        self._manager = manager
        self._shutdown_event = shutdown_event
        self._batch_builder = batch_builder
        self._embedder = embedder
        self._base_model_path = base_model_path
        self._model_config = model_config
        self._special_token = special_token
        self._embeddings_integration = embeddings_integration
        self._translations_dir = translations_dir
        self._n_BLEU_jobs = n_BLEU_jobs
        self._n_chrF_jobs = n_chrF_jobs

    def _add_scorers_to_pool(self, scorers_pool: ScorersPool):
        """
        Add BLEU and chrF++ scorers to the scorers pool.
        """
        created = False

        if self._n_BLEU_jobs > 0:
            created = True
            scorers_pool.add_scorers_group(BLEUScorerProcess.get_metric_name(), BLEUScorerProcess, self._n_BLEU_jobs)

        if self._n_chrF_jobs > 0:
            created = True
            scorers_pool.add_scorers_group(CHRFScorerProcess.get_metric_name(), CHRFScorerProcess, self._n_chrF_jobs)

        if not created:
            scorers_pool.add_scorers_group(NoopScorer.get_metric_name(), NoopScorer, 1)

    def _create_processes(self, manager: ExtendedSyncManager,
                          task_list: ScoreTaskList) -> Tuple[ScoreTaskTranslator, ScorersPool, ProcessType]:
        """
        Create all processes required for the experiments run.

        :param manager: Multiprocessing manager
        :param task_list: ScoreTaskList whose scores will be updated
        """
        translator = ScoreTaskTranslator(self._batch_builder, self._embedder, self._shutdown_event, manager,
                                         self._base_model_path, self._model_config, self._special_token,
                                         self._embeddings_integration, self._translations_dir)
        scorers_pool = ScorersPool(translator.translations, manager, self._shutdown_event)
        self._add_scorers_to_pool(scorers_pool)
        scores_writer = task_list.listen(scorers_pool.scores)
        return translator, scorers_pool, scores_writer

    def run(self, task_list: ScoreTaskList, progress: Optional[tqdm.tqdm] = None):
        """
        Run the experiments manager on incomplete tasks in `task_list`.

        Start the the scorers pool, a process that updates the scores cache file and then start translating incomplete
        tasks from the `ScoreTaskList`.

        Ends when all experiments are run.

        :param task_list: incomplete tasks of this ScoreTaskList will be processed
        :param progress: tqdm instance to update after each score
        """
        translator, scorers_pool, scores_writer = self._create_processes(self._manager, task_list)

        scorers_pool.start()
        scores_writer.start()

        for task in task_list.incomplete_tasks:
            if self._shutdown_event.is_set():
                break
            if progress:
                progress.set_postfix(task=task.key)
            try:
                translator.translate(task)
            except StoppedException:
                break
            if progress:
                progress.update()

        translator.close()

        # We wait for all processes to finish
        scorers_pool.join()
        scores_writer.join()
