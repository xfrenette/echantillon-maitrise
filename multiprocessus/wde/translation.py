"""
Created by: Xavier Frenette <xavier.frenette@umontreal.ca>
Created on: 2021-03-15
"""
import logging
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import NamedTuple, List, Optional, Tuple, cast, Union, Sequence

import torch
from fairseq.data import Dictionary
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel, TransformerEncoder

from couleurs.fairseq_plugins.transformer import CouleursTransformerEncoder
from couleurs.utils.multiprocessing import ExtendedSyncManager
from . import BatchBuilder, ScoreTask, DomainEmbedder, Batch


class TranslationResult(NamedTuple):
    """
    Represents a translation created by a TranslatorProcess
    """
    score_task: ScoreTask
    batch: Batch
    translations: List[str]


class ScoreTaskTranslatorClosedException(BaseException):
    """Exception raised when trying to call `translate()` on a closed `ScoreTaskTranslator`"""
    pass


class ScoreTaskTranslator:
    """
    Class that creates a translation model and translates batches for ScoreTask.

    The translation model will have it's `special_token` embedding updated with the embedding of the task's domain set.

    Domain embedding can be disabled by setting the `domain_embedder` to None. In that case, no domain embedding will be
    calculated for each task and the `special_token` will be ignored (the special token won't be updated in the
    translation model).

    Start a translation for a task with `translate(task)`. The result will be in tbe `translations` property, a
    StoppableQueue. Once all tasks are translated, call `close()` to notify processes listening to the queue that no
    more translations are expected.

    This class uses a translation cache (as files) to speed up re-translations of already translated tasks.
    """

    # Maximum number of generated translations in the `translations` queue
    MAX_TRANSLATIONS = 50

    def __init__(self, batch_builder: BatchBuilder, domain_embedder: Union[DomainEmbedder, None],
                 shutdown_event: EventType, manager: ExtendedSyncManager, base_model_path: Path, model_config: dict,
                 special_token: Optional[str] = None, embeddings_integration: Optional[str] = None, output_dir: Optional[Path] = None):
        """
        Set `domain_embedder` to None to disable domain embedding calculation for each task. `special_token` will be
        ignored.

        `embeddings_integration` determines how new domain embeddings will be integrated in the model. If None (default)
          the embedding will be used a the special token's embedding in the encoder. If `"foreign"`, embeddings will
          be used in the encoder's `embeddings_combiner` for the special token. In this last case, the encoder must be
          an instance of CouleursTransformerEncoder with an `embeddings_combiner`.

        :param output_dir: Directory where to output translations
        :param batch_builder: instance of a BatchBuilder to construct batches for tasks
        :param domain_embedder: instance of the DomainEmbedder to use to calculate domain embeddings
        :param shutdown_event: Event that, once set, will shutdown translations queue
        :param manager: Multiprocessing ExtendedSyncManager
        :param base_model_path: Base Transformer model that will be updated for each domain embedding
        :param model_config: Configuration to pass to all new translation model
        :param special_token: Token to prepend to source when translating (and for which to replace the embedding)
        :param embeddings_integration: How calculated embeddings are integrated in the model
        """
        super().__init__()

        if domain_embedder is not None and special_token is None:
            raise ValueError("special_token cannot be None if domain_embedder is not None")

        self._output_dir = output_dir
        self._batch_builder = batch_builder
        self._domain_embedder = domain_embedder
        self._shutdown_event = shutdown_event
        self._manager = manager
        self._base_model_path = base_model_path
        self._model_config = model_config
        self._special_token = None if domain_embedder is None else special_token
        self._embeddings_integration = embeddings_integration

        self._logger = logging.getLogger(self.__class__.__name__)
        self.translations = manager.StoppableQueue(self.MAX_TRANSLATIONS, stop_event=shutdown_event)
        self._model: Optional[GeneratorHubInterface] = None
        self._current_embedding: Optional[str] = None
        self._closed = False

        if self._output_dir is not None:
            self._output_dir.mkdir(exist_ok=True)

    def _create_source_sentences(self, sentences: Sequence[str]) -> List[str]:
        """
        Preprocess a sequence of sentences to be ready for translation.

        Prepend the special token to each sentence, if set.

        :return: The list of sentences preprocessed.
        """
        if self._special_token is not None:
            return [f"{self._special_token} {s}" for s in sentences]
        return list(sentences)

    def _update_model(self, score_task: ScoreTask):
        """
        Update the special token's embedding with the embedding for the task's domain
        :param score_task: Task for which to calculate the domain embedding
        """
        embedding = self._domain_embedder.calculate_for_task(score_task)
        if self._embeddings_integration == "foreign":
            self._logger.info(f"Updating the foreign embeddings with embedding {score_task.embedding_key}")
            self._update_foreign_embeddings(embedding)
        else:
            self._logger.info(f"Updating the special token with embedding {score_task.embedding_key}")
            self._update_special_token_embedding(embedding)

    def _update_special_token_embedding(self, embedding: torch.Tensor):
        """
        Update (in place) the `special_token` embedding of the `model`'s encoder.

        :param embedding: Its new embedding
        """
        encoder, special_token_index = self._get_encoder_and_special_token_index()
        with torch.no_grad():
            embedding = embedding.to(device=encoder.embed_tokens.weight.device)
            encoder.embed_tokens.weight[special_token_index] = embedding

    def _update_foreign_embeddings(self, embedding: torch.Tensor):
        """
        Update the foreign embeddings of the `model.encoder.embeddings_combiner`.

        :param embedding: The new embedding
        """
        encoder, special_token_index = self._get_encoder_and_special_token_index()
        encoder = cast(CouleursTransformerEncoder, encoder)
        with torch.no_grad():
            current_foreign_embeddings = encoder.embeddings_combiner.foreign_embeddings
            if current_foreign_embeddings is None:
                current_foreign_embeddings = {}
            current_foreign_embeddings[special_token_index] = embedding
            encoder.embeddings_combiner.foreign_embeddings = current_foreign_embeddings

    def _get_encoder_and_special_token_index(self) -> Tuple[TransformerEncoder, int]:
        model: TransformerModel = self._model.models[0]
        # noinspection PyTypeChecker
        encoder: TransformerEncoder = model.encoder
        dictionary: Dictionary = model.encoder.dictionary
        special_token_index = dictionary.index(self._special_token)
        if special_token_index == dictionary.unk_index:
            raise ValueError("Could not find the special token in the dictionary")
        return encoder, special_token_index

    def _load_model(self):
        """
        Loads the translation model in memory. Does nothing if the model is
        already loaded
        """
        if self._model is None:
            model_path = str(self._base_model_path.parent.absolute())
            self._logger.info(f"Loading the model from: {model_path}")
            checkpoint_file = self._base_model_path.name
            self._model = TransformerModel.from_pretrained(model_path, checkpoint_file, ".", **self._model_config)

            if torch.cuda.is_available() and not getattr(self._model_config, "cpu", False):
                self._model.cuda()

    def close(self):
        """
        Close this translator.

        Call this method once all tasks have been translated. `None` will be added to the `translations` queue to
        notify a listening process.

        You can't call `translate()` again on a close translator.
        """
        self.translations.watching_put(None, False)
        self._closed = True

    @staticmethod
    def generate_translation_path(translations_dir: Path, task: ScoreTask) -> Path:
        return translations_dir / f"{task.embedding_key}.txt"

    def translate(self, score_task: ScoreTask):
        """
        Starts a translation on the ScoreTask.

        The resulting translation will be put on the `self.translations` queue as a `TranslationResult` instance.

        The `self.translations` queue has a maximum size, so this method will block if the queue is full.

        :param score_task: The ScoreTask to translate
        """
        if self._closed:
            raise ScoreTaskTranslatorClosedException("You can't call translate on a closed translator")

        batch = self._batch_builder.build_for_task(score_task)

        # Use of the cache:
        # The domain embedding of this task will probably be used multiple times to translate different test sets
        # selected by bootstrapping from the domain data. If a lot of bootstrap sets are created, each sentence will
        # probably end up in multiple test sets, each time being translated again. It might be a bottleneck. To
        # improve speed, when we encounter a task with a new embedding set, we translate one time the whole domain
        # data set. Then, for each task using this embedding set (including the current task), we will select the
        # translations for this bootstrap from the translated domain data.
        #
        # To enable this feature, specify a `output_dir` in the initialisation. If it's not set, the model will
        # translate each bootstrap sets.

        use_cache = self._output_dir is not None
        translations: Optional[List[str]] = None
        sentences_to_translate: Optional[List[str]] = None

        # If we use the translation cache, we extract translations from the cache for the task's embedding. If the
        # cache file is created (we didn't yet translate for this embedding set), we extract all the sentences of the
        # domain to translate them later.
        #
        # If we don't use the translation cache, we extract the sentences for this task's batch to translate them later.
        if use_cache:
            translations_file_path = self.generate_translation_path(self._output_dir, score_task)

            if translations_file_path.exists():
                with open(translations_file_path, "r") as file:
                    translations = [line.strip() for line in file]
            else:
                domain_data = self._batch_builder.data_source.get_source_sentences(score_task.domain)
                sentences_to_translate = domain_data
        else:
            sentences_to_translate = batch.source

        # If we have sentences to translate, we load the model and translate them. If we use a cache, we then save
        # the translations
        if sentences_to_translate is not None:
            self._logger.info(f"Translating: {score_task.key}")

            sentences_to_translate = self._create_source_sentences(sentences_to_translate)
            embedding_key = score_task.embedding_key

            self._load_model()

            if self._domain_embedder is not None and embedding_key != self._current_embedding:
                self._update_model(score_task)
                self._current_embedding = embedding_key

            # No need for torch.no_grad() since the `translate` method does it
            translations = self._model.translate(sentences_to_translate)

            # If we use cache, we save the generated translations in the file
            if use_cache:
                translations_file_path = self.generate_translation_path(self._output_dir, score_task)
                with open(translations_file_path, "w") as file:
                    file.write("\n".join(translations) + "\n")

        # If we used cache, `translations` contain all the domain's data, we need to keep only the sentences for this
        # task's batch set.
        if use_cache:
            translations = self._batch_builder.batchify(translations, score_task)

        result = TranslationResult(score_task, batch, translations)
        self.translations.watching_put(result)
