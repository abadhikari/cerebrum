from typing import Any, ContextManager, Mapping, Protocol

# Type alias for a row returned from the database
Row = dict[str, Any]

# Type alias for Rows returned from the database
Rows = list[Row]

# Parameters for parameterized SQL. Keys are the named placeholders (":name"),
# values are values.
SqlParams = Mapping[str, Any]


class SqlClient(Protocol):
  """
  Minimal database client interface used by higher-level business logic.
  """

  def execute(self, sql: str, params: SqlParams | None = None) -> int:
    """
    Execute a non-SELECT statement (INSERT/UPDATE/DELETE/DDL).

    Args:
        sql: The SQL statement with named placeholders (e.g., `... WHERE id = :id`).
        params: Mapping of placeholder names to values to bind. Use `bytes` for BLOBs.

    Returns:
        int: Number of affected rows **when supported by the driver**.
             Note: SQLite may return -1 for statements where `rowcount` is undefined.

    Raises:
        Any driver-specific exception if execution fails (propagate; do not swallow).
    """
    ...

  def query(self, sql: str, params: SqlParams | None = None) -> Rows:
    """
    Execute a SELECT and return all rows as a list of dicts.

    Args:
        sql: The SQL SELECT with named placeholders.
        params: Mapping of placeholder names to values to bind.

    Returns:
        Rows: List of rows.

    Raises:
        Any driver-specific exception if execution fails (propagate; do not swallow).
    """
    ...

  def transaction(self) -> ContextManager[None]:
    """
    Return a context manager that wraps a database transaction.

    Semantics:
      - Enters with BEGIN (or BEGIN IMMEDIATE/EXCLUSIVE, per implementation).
      - Commits on normal exit.
      - Rolls back if an exception escapes the `with` block.

    Example:
        >>> with client.transaction():
        ...     client.execute("INSERT INTO t(a) VALUES(:a)", {"a": 1})
        ...     client.execute("UPDATE t SET a = :a WHERE id = :id", {"a": 2, "id": 5})

    Returns:
        ContextManager[None]: A context manager controlling the transaction boundary.
    """
    ...
