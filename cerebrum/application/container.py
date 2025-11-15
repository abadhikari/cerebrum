from pathlib import Path

from cerebrum.infra.db import SqliteClient, SqliteSqlProducer, SqliteSchemaManager
from cerebrum.infra.repository import SqliteRepository
from cerebrum.infra.embedder import SentenceTransformerEmbedder
from cerebrum.infra.semantic_store import FaissClient
from cerebrum.application.service import Service


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

    def __init__(self, db_filepath: Path, faiss_filepath: Path, model_name: str):
        """
        Initialize the container with configuration parameters.

        Args:
            db_filepath (Path): Filesystem path to the SQLite database.
            faiss_filepath (Path): Filesystem path to the FAISS index.
            model_name (str): Name of the embedding model to load.
        """
        self._db_filepath = db_filepath
        self._faiss_filepath = faiss_filepath
        self._model_name = model_name

        self._faiss_client = None
        self._sql_client = None
        self._service = None
        self._embedder = None
        
        self._started = False
  
    def start(self) -> None:
        """Initialize all dependencies and build the application service."""
        if self._started:
          return

        embedder = SentenceTransformerEmbedder(self._model_name)
        faiss_client = FaissClient(self._faiss_filepath, embedder.get_dimensions())
        sql_client = SqliteClient(self._db_filepath)
        sql_client.connect()
        sql_producer = SqliteSqlProducer()
        SqliteSchemaManager(sql_client, sql_producer).init()
        repository = SqliteRepository(sql_client, sql_producer)
        service = Service(repository, embedder, faiss_client)

        self._faiss_client = faiss_client
        self._sql_client = sql_client
        self._service = service
        self._embedder = embedder

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
            self._started = False

    @property
    def service(self) -> Service:
        """
        Return the initialized `Service` instance.

        Raises:
            RuntimeError: If the container has not been started.

        Returns:
            Service: The fully constructed application service.
        """
        if not self._started:
            raise RuntimeError("Container has not been started. Call start() first.")
        return self._service
