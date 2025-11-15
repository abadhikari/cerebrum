import os
from pathlib import Path


class Config:
    """
    Application configuration loader.

    Reads required and optional environment variables at startup and exposes
    strongly-typed values (e.g., Path objects) for use during dependency
    injection. This isolates environment handling from the rest of the
    application and prevents config logic from leaking into core or infra layers.
    """

    def __init__(self):
        """
        Load and validate environment variables.

        Required:
            BASE_DIR – Root directory for all persistent storage.
        
        Optional:
            DB_FILE_NAME – SQLite filename (default: 'cerebrum.db')
            FAISS_FILE_NAME – FAISS index filename (default: 'semantic.faiss')
            LANGUAGE_MODEL_NAME – Embedding model name 
                                  (default: 'all-MiniLM-L6-v2')

        Raises:
            RuntimeError: If BASE_DIR is missing.
        """
        base_dir_env = os.getenv("BASE_DIR")
        if not base_dir_env:
          raise RuntimeError("BASE_DIR env variable not set.")
        self._base_dir = Path(base_dir_env)

        self._data_dir = self._base_dir / "data"
        self._db_dir = self._data_dir / "db"
        self._faiss_dir = self._data_dir / "faiss"

        self._db_file_name = os.getenv("DB_FILE_NAME", "cerebrum.db")
        self._faiss_file_name = os.getenv("FAISS_FILE_NAME", "semantic.faiss")

        self.language_model_name = os.getenv("LANGUAGE_MODEL_NAME", "all-MiniLM-L6-v2")
    
    @property
    def db_filepath(self) -> Path:
        """
        Full path to the SQLite database file.

        Returns:
            Path: `<DB_DIR>/<DB_FILE_NAME>`
        """
        return self._db_dir / self._db_file_name

    @property
    def faiss_filepath(self) -> Path:
        """
        Full path to the FAISS index file.

        Returns:
            Path: `<FAISS_DIR>/<FAISS_FILE_NAME>`
        """
        return self._faiss_dir / self._faiss_file_name