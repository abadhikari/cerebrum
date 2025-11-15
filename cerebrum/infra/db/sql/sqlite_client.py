import sqlite3
from contextlib import contextmanager
from pathlib import Path

from cerebrum.infra.db.base_client import BaseClient
from cerebrum.infra.db.sql.sql_client import Row, Rows, SqlParams


class SqliteClient(BaseClient):
  """
  SQLite implementation of the SqlClient interface.

  Provides a lightweight, file-based SQL backend with safe parameterized
  execution and transaction management. Designed for local persistence or
  low-concurrency applications where a full client/server database would be
  overkill.
  """

  def __init__(self, db_filepath: Path, timeout: float = 5.0):
    """
    Initialize a new SqliteClient.

    Args:
        db_filepath (Path): Path to the SQLite database file.
        timeout (float): Seconds to wait for a locked database before raising an error.
    """
    self._db_filepath = db_filepath
    self._timeout = timeout
    self._connection: sqlite3.Connection | None = None
  
  def connect(self):
    """
    Establish a connection to the SQLite database if not already connected.

    Configures:
      - WAL journal mode for concurrent readers/writers.
      - NORMAL synchronous mode for balanced durability/performance.
      - Foreign key enforcement.

    Idempotent; repeated calls are no-ops once connected.
    """
    if self._connection:
      return
    
    self._db_filepath.parent.mkdir(parents=True, exist_ok=True) 

    self._connection = sqlite3.connect(
      self._db_filepath,
      timeout=self._timeout,
      isolation_level=None,  # autocommit ON
      detect_types=sqlite3.PARSE_DECLTYPES,
    )
    self._connection.row_factory = sqlite3.Row

    self._connection.execute("PRAGMA journal_mode = WAL;")
    self._connection.execute("PRAGMA synchronous = NORMAL;")
    self._connection.execute("PRAGMA foreign_keys = ON;")

  def close(self):
    """
    Close the active SQLite connection, if present.

    Idempotent; silently ignored if already closed.
    """
    if self._connection:
      self._connection.close()
      self._connection = None
  
  def execute(self, sql: str, params: SqlParams | None = None) -> int:
    """
    Execute a non-SELECT SQL statement.

    Args:
        sql: SQL statement with named placeholders (e.g., `UPDATE ... WHERE id = :id`).
        params: Optional mapping of placeholder names to values. Use `bytes` for BLOBs.

    Returns:
        int: Number of affected rows. SQLite may return -1 for statements where
             the count is undefined.

    Raises:
        sqlite3.Error: If execution fails.
    """
    cur = self.connection.execute(sql, params or {})
    return cur.rowcount
  
  def query(self, sql: str, params: SqlParams | None = None) -> Rows:
    """
    Execute a SELECT statement and return all results as a list of dicts.

    Args:
        sql: SQL SELECT with named placeholders.
        params: Optional mapping of placeholder names to values.

    Returns:
        Rows: List of rows, where each row is `dict[str, Any]`.

    Raises:
        sqlite3.Error: If execution fails.
    """
    cur = self.connection.execute(sql, params or {})
    return [dict(row) for row in cur.fetchall()]
  
  def query_one(self, sql: str, params: SqlParams | None = None) -> Row | None:
    """
    Execute a statement that returns a single row.

    Args:
        sql: SQL statement (SELECT or RETURNING ...).
        params: Optional mapping of placeholder names to values.

    Returns:
        Row: The first row as a dict, or None if no results.
    """
    rows = self.query(sql, params)
    return rows[0] if rows else None

  @property
  def connection(self) -> sqlite3.Connection:
      """
      Active SQLite connection.

      Returns:
          sqlite3.Connection: Current connection object.

      Raises:
          RuntimeError: If called before `connect()` is invoked.
      """
      if not self._connection:
          raise RuntimeError("Database not connected. Call connect() first.")
      return self._connection

  @contextmanager
  def transaction(self):
    """
    Context manager for explicit transaction control.

    Begins an `IMMEDIATE` transaction on entry, commits on success,
    and rolls back on exception or keyboard interrupt.

    Example:
        >>> with client.transaction():
        ...     client.execute("INSERT INTO t (a) VALUES (:a)", {"a": 1})
        ...     client.execute("UPDATE t SET a = :a WHERE id = :id", {"a": 2, "id": 5})
    """
    connection = self.connection
    try:
      connection.execute("BEGIN IMMEDIATE;")
      yield
      connection.execute("COMMIT;")
    except (Exception, KeyboardInterrupt):
      connection.execute("ROLLBACK;")
      raise
