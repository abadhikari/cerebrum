from cerebrum.infra.db import SqliteClient, SqliteSqlProducer, SqliteSchemaManager
from cerebrum.infra.repository import SqliteRepository
from cerebrum.infra.embedder import SentenceTransformerEmbedder
from cerebrum.infra.semantic_store import FaissClient
from cerebrum.application.service import Service
from cerebrum.infra.language_model import OllamaModel
from cerebrum.application.config import Config
from cerebrum.infra.language_model import LanguageModel

from typing import Optional


class Container:
    """
    Dependency container for the Cerebrum application.

    This class owns the lifecycle of all infrastructure components:
    the embedder, semantic map, database client, repository, and the
    high-level `Service`. Nothing is constructed until `start()` is
    called, ensuring explicit, predictable initialization.

    The container also handles cleanup via `stop()`, flushing the semantic 
    map to disk and closing database connections.
    """

    def __init__(self, config: Config):
        """
        Initialize the container with configuration parameters.

        Args:
            config (Config):
                Structured application settings loaded from environment
                (file paths, model names, hyperparameters, etc.).
        """
        self._config = config

        self._faiss_client: Optional[FaissClient] = None
        self._sql_client: Optional[SqliteClient] = None
        self._service: Optional[Service] = None
        self._embedder: Optional[SentenceTransformerEmbedder] = None
        self._language_model: Optional[OllamaModel] = None
        
        self._started = False
  
    def start(self) -> None:
        """Initialize all dependencies and build the application service."""
        if self._started:
            return

        embedder = SentenceTransformerEmbedder(self._config.embedding_model_name)
        faiss_client = FaissClient(self._config.faiss_filepath, embedder.get_dimensions())
        sql_client = SqliteClient(self._config.db_filepath)
        sql_client.connect()
        sql_producer = SqliteSqlProducer()
        SqliteSchemaManager(sql_client, sql_producer).init()
        repository = SqliteRepository(sql_client, sql_producer)
        service = Service(repository, embedder, faiss_client)

        language_model = OllamaModel(self._config.language_model_name, self._config.language_model_temperature)

        self._faiss_client = faiss_client
        self._sql_client = sql_client
        self._service = service
        self._embedder = embedder
        self._language_model = language_model

        self._started = True
    
    def stop(self) -> None:
        """Tear down all managed resources."""
        if not self._started:
            return

        try:
            if self._faiss_client:
                self._faiss_client.write_index()
        finally:
            if self._sql_client:
                self._sql_client.close()
            
            self._faiss_client = None
            self._sql_client = None
            self._service = None
            self._embedder = None
            self._language_model = None
            self._started = False
    
    def __enter__(self) -> "Container":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def service(self) -> Service:
        """
        Return the initialized `Service` instance.

        Raises:
            RuntimeError: If the container has not been started.

        Returns:
            Service: The fully constructed application service.
        """
        self._check_started()
        return self._service
    
    def _check_started(self) -> None:
        if not self._started:
            raise RuntimeError("Container has not been started. Call start() first.")
    
    @property
    def language_model(self) -> LanguageModel:
        """
        Return the initialized `LanguageModel` instance.

        Raises:
            RuntimeError: If the container has not been started.

        Returns:
            LanguageModel: The fully constructed language model.
        """
        self._check_started()
        return self._language_model
