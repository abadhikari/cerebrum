from typing import Protocol


class SchemaManager(Protocol):
  """
  Protocol defining the interface for managing database schema lifecycle.

  Implementations are responsible for creating or migrating database tables,
  indexes, and constraints required by the application. 
  """

  def init(self) -> None:
    """
    Initialize the database schema.
    """
    ...