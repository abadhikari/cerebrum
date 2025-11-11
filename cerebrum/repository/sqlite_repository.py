from datetime import datetime, timezone
from uuid import uuid4
import json

from cerebrum.repository.thought_repository import Index, ThoughtRecord, ThoughtStatus
from cerebrum.embedder.embedder import EmbeddingRecord
from cerebrum.infra.db.sql.sql_client import SqlClient, Row
from cerebrum.infra.db.sql.sqlite_sql_producer import SqliteSqlProducer
from cerebrum.core.thought import Thought
from cerebrum.infra.semantic_store.semantic_store import Ids


class SqliteRepository:
  """
  SQLite-backed implementation of the ThoughtRepository protocol.

  Handles persistence for thoughts, embeddings, and index metadata
  through parameterized SQL and managed transactions.
  """

  def __init__(self, sql_client: SqlClient, sql_producer: SqliteSqlProducer):
    """
    Args:
        sql_client (SqlClient): Executes SQL statements and manages transactions.
        sql_producer (SqliteSqlProducer): Generates parameterized SQL queries for the schema.
    """
    self._sql_client = sql_client
    self._sql_producer = sql_producer
  
  def create_index(self, index_name: str, algorithm: str) -> str:
    """
    Insert a new index record.

    Args:
        index_name (str): Human-readable name.
        algorithm (str): Index algorithm (e.g., 'Flat', 'IVF', 'HNSW').

    Returns:
        str: Generated index_id (UUID).
    """
    index_id = str(uuid4())
    sql, params = self._sql_producer.insert_indexes_row(index_name, index_id, algorithm)
    self._sql_client.execute(sql, params)
    return index_id
  
  def insert_thought(self, thought: Thought, embedding: EmbeddingRecord, index_id: str) -> int:
    """
    Insert a thought and link it to the specified index.

    Args:
        thought (Thought): The thought content and metadata.
        embedding (EmbeddingRecord): The computed embedding vector and model info.
        index_id (str): UUID of the target index.

    Returns:
        int: Generated id64 linking record in `index_embeddings`.
    """
    with self._sql_client.transaction():
      embedding_id = str(uuid4())
      sql, params = self._sql_producer.insert_embeddings_row(thought, embedding, embedding_id)
      self._sql_client.execute(sql, params)

      sql, params = self._sql_producer.insert_index_embedding_row(index_id, embedding_id)
      row = self._sql_client.query_one(sql, params)
      return self._read_id64(row)
  
  def _read_id64(self, row: Row):
    """
    Extract the `id64` column value from a query result.

    Args:
        row (Row): Row containing an `id64` key.

    Returns:
        int: Parsed `id64` value.

    Raises:
        ValueError: If the row is missing or malformed.
    """
    if not row or "id64" not in row:
      raise ValueError("Expected id64 but got no row/column")
    return int(row["id64"])

  def complete_thought_insert(self, id64: int) -> None:
    """
    Mark an index_embeddings record as complete after FAISS insertion.

    Args:
        id64 (int): Primary key of the index_embeddings row.
    """
    sql, params = self._sql_producer.update_index_embeddings_status(id64)
    self._sql_client.execute(sql, params)
  
  def list_indexes(self) -> list[Index]:
    """
    Retrieve all index records.

    Returns:
        list[Index]: Available index metadata, sorted by creation time.
    """
    sql, params = self._sql_producer.select_indexes()
    rows = self._sql_client.query(sql, params)
    return [self._hydrate_index(row) for row in rows]
  
  def _hydrate_index(self, row: Row) -> Index:
    """
    Convert a raw DB row into an Index dataclass.

    Args:
        row (Row): Row containing index metadata.

    Returns:
        Index: Hydrated index object.
    """
    created_at = self._timestamp_to_utc_datetime(row["created_at"])
    return Index(
        index_id=row["index_id"],
        index_name=row["index_name"],
        algorithm=row["algorithm"],
        created_at=created_at,
    )
  
  def _timestamp_to_utc_datetime(self, timestamp: datetime) -> datetime:
    """
    Ensure a timestamp is timezone-aware and UTC-based.

    Args:
        timestamp (datetime): Input timestamp.

    Returns:
        datetime: UTC-aware timestamp.
    """
    return timestamp.replace(tzinfo=timezone.utc)
  
  def retrieve_thoughts(self, ids: Ids, index_id: str, status: ThoughtStatus) -> list[ThoughtRecord]:
    """
    Fetch thoughts by id64 from a specific index.

    Args:
        ids (Ids): Array of id64 values to retrieve.
        index_id (str): Index identifier.
        status (ThoughtStatus): Thought status to filter by.

    Returns:
        list[ThoughtRecord]: Retrieved thought records.
    """
    sql, params = self._sql_producer.select_ids(ids, index_id, status.value)
    rows = self._sql_client.query(sql, params)
    return [self._hydrate_thought_record(row) for row in rows]
  
  def _hydrate_thought_record(self, row: Row) -> ThoughtRecord:
    """
    Convert a raw DB row into a ThoughtRecord dataclass.

    Args:
        row (Row): Row containing thought and embedding metadata.

    Returns:
        ThoughtRecord: Hydrated thought record object.
    """
    tags = tuple(json.loads(row['tags']))
    created_at = self._timestamp_to_utc_datetime(row["created_at"])
    return ThoughtRecord(
      tags=tags,
      body=row['body'],
      id64=int(row['id64']),
      embedding_id=row['embedding_id'],
      status=ThoughtStatus(row['status']),
      created_at=created_at
    )
