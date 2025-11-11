from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from cerebrum.core.thought import Thought
from cerebrum.infra.semantic_store.semantic_store import Ids
from cerebrum.embedder.embedder import EmbeddingRecord

@dataclass(frozen=True, slots=True)
class Index:
  """Metadata for a semantic index."""
  index_id: str
  index_name: str
  algorithm: str
  created_at: datetime

class ThoughtStatus(StrEnum):
  """Logical state of a thought record."""
  ACTIVE = 'active'
  ARCHIVED = 'archived'

@dataclass(frozen=True, slots=True)
class ThoughtRecord:
  """Persisted thought and its storage/indexing metadata."""
  embedding_id: str
  id64: int
  body: str
  tags: tuple[str, ...]
  status: ThoughtStatus
  created_at: datetime

class ThoughtRepository(Protocol):
  """
  Persistence contract for thoughts and indexes.

  Implementations must be idempotent where noted and preserve data integrity.
  """
  
  def create_index(self, index_name: str, algorithm: str) -> str:
    """
    Create a new index record.

    Args:
      index_name (str): Human-readable name.
      algorithm (str): Backend algorithm label (e.g., 'Flat', 'IVF', 'HNSW').

    Returns:
      str: Generated index_id (UUID or equivalent).
    """
    ...
  
  def insert_thought(self, thought: Thought, embedding: EmbeddingRecord, index_id: str) -> int:
    """
    Insert a thought and link it to an index.

    Args:
      thought (Thought): Thought payload (body, tags, etc.).
      embedding (EmbeddingRecord): Embedding matrix + model metadata for the thought.
      index_id (str): Target index identifier.

    Returns:
      int: Generated id64 linking row (used by the vector store).
    """
    ...

  def complete_thought_insert(self, id64: int) -> None:
    """
    Mark the index-link record as complete after vector store write.

    Args:
      id64 (int): Primary key of the index-link row.
    """
    ...
  
  def list_indexes(self) -> list[Index]:
    """
    List all indexes.

    Returns:
      list[Index]: Index records (ordering is implementation-defined).
    """
    ...
  
  def retrieve_thoughts(self, ids: Ids, index_id: str, status: ThoughtStatus) -> list[ThoughtRecord]:
    """
    Fetch thoughts by id64 within an index, filtered by status.

    Args:
      ids (Ids): Array-like of id64 values to retrieve.
      index_id (str): Index scope.
      status (ThoughtStatus): Thought status state.

    Returns:
      list[ThoughtRecord]: Matching records (ordering is implementation-defined).
    """
    ...
