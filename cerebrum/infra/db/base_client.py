from abc import ABC, abstractmethod
from typing import Self


class BaseClient(ABC):
  """
  Abstract base class defining the minimal interface for a client
  that manages a connection lifecycle (e.g., database, API, socket).

  Subclasses must implement `connect()` and `close()` to handle
  resource initialization and teardown. This base class also enables
  context-manager semantics (`with` statements) for safe and predictable
  resource handling.
  """

  @abstractmethod
  def connect(self) -> None:
    """
    Establish a connection to the underlying resource.

    Implementations should:
      - Initialize any necessary internal handles or sessions.
      - Be idempotent (safe to call multiple times without side effects).
    """
    ...
  
  @abstractmethod
  def close(self) -> None:
    """
    Close the active connection and release all associated resources.

    Implementations should:
      - Handle repeated calls gracefully.
      - Ensure pending operations are safely finalized or rolled back.
    """
    ...

  def __enter__(self) -> Self:
    """
    Enter a context manager by establishing a connection.

    Returns:
        Self: The connected client instance for use within the `with` block.

    Example:
        >>> with SqliteClient("db.sqlite") as client:
        ...     client.execute("SELECT 1")
    """
    self.connect()
    return self
  
  def __exit__(self, exc_type, exc_value, traceback) -> bool:
    """
    Exit the context manager by closing the connection.

    Always closes the connection, regardless of whether an exception
    occurred inside the `with` block.

    Args:
        exc_type: The exception type, if raised.
        exc_value: The exception instance, if raised.
        traceback: The traceback object, if raised.

    Returns:
        bool: Always False so exceptions (if any) propagate.
    """
    self.close()
    return False