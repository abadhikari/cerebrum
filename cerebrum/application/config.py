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
            BASE_DIR – Root directory for all persistent data.
            DB_FILE_NAME – Database filename.
            FAISS_FILE_NAME – FAISS index filename.
            EMBEDDING_MODEL_NAME – Name of the embedding model to load.
            LANGUAGE_MODEL_NAME – Name of the language model to use.
            LANGUAGE_MODEL_TEMPERATURE – Sampling temperature for the language model.

        Raises:
            RuntimeError: If any required env vars are missing.
        """
        self._base_dir: Path = Path(self._required_env("BASE_DIR"))

        self._data_dir: Path = self._base_dir / "data"
        self._db_dir: Path = self._data_dir / "db"
        self._faiss_dir: Path = self._data_dir / "faiss"

        self._db_file_name: str = self._required_env("DB_FILE_NAME")
        self._faiss_file_name: str = self._required_env("FAISS_FILE_NAME")

        self.embedding_model_name: str = self._required_env("EMBEDDING_MODEL_NAME")
        self.language_model_name: str = self._required_env("LANGUAGE_MODEL_NAME")
        self.language_model_temperature: float = float(self._required_env("LANGUAGE_MODEL_TEMPERATURE"))
    
    def _required_env(self, env_key: str) -> str:
        env_value = os.getenv(env_key)
        if not env_value:
          raise RuntimeError(f"Required env variable not set: {env_key}")
        return env_value
    
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