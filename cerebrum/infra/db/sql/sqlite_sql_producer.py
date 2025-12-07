import json

from cerebrum.core.embedder import EmbeddingRecord
from cerebrum.core.semantic_store import Ids
from cerebrum.core.thought import Thought
from cerebrum.infra.db.sql.sql_client import SqlParams

# Type alias representing a single SQL statement and its named parameters.
SqlStatement = tuple[str, SqlParams]

# Type alias representing one SQL statement + many parameter dicts
SqlBatchStatement = tuple[str, list[SqlParams]]


class SqliteSqlProducer:
    """Factory for generating parameterized Sqlite SQL statements used by the repository layer.

    This class does not execute SQL. It only constructs SQL strings and parameter
    dictionaries to be consumed by a client or repository. Each method returns
    either a single (sql, params) tuple or a list of such tuples.
    """

    def __init__(self): ...

    def insert_embeddings_row(
        self,
        thought: Thought,
        embedding: EmbeddingRecord,
        embedding_id: str,
    ) -> SqlStatement:
        """Create an INSERT statement for a new embedding record.

        Args:
            thought (Thought): The metadata for a thought in
            cerebrum.
            embedding (EmbeddingRecord): The computed vector representation to store and its metadata.
            embedding_id (str): Unique identifier for this embedding record.

        Returns:
            SqlStatement: Tuple of SQL string and parameter dict.
        """
        sql = """
            INSERT INTO embeddings (embedding_id, body, tags, model_name, embedding)
            VALUES (:embedding_id, :body, :tags, :model_name, :embedding);
        """
        params = {
            "embedding_id": embedding_id,
            "body": thought.body,
            "tags": json.dumps(thought.tags),
            "model_name": embedding.model_name,
            "embedding": embedding.vector.tobytes(),
        }
        return (sql, params)

    def insert_indexes_row(
        self,
        index_name: str,
        index_id: str,
        algorithm: str,
    ) -> SqlStatement:
        """Create an INSERT statement for a new semantic map index record.

        Args:
            index_name (str): Human-readable name of the index.
            index_id (str): Unique identifier for this index record.
            algorithm (str): Index algorithm (e.g., 'Flat', 'IVF', 'HNSW').

        Returns:
            SqlStatement: Tuple of SQL string and parameter dict.
        """
        sql = """
            INSERT INTO indexes (index_id, index_name, algorithm)
            VALUES (:index_id, :index_name, :algorithm);
        """
        params = {
            "index_id": index_id,
            "index_name": index_name,
            "algorithm": algorithm,
        }
        return (sql, params)

    def insert_index_embedding_row(
        self,
        index_id: str,
        embedding_id: str,
    ) -> SqlStatement:
        """Create an INSERT statement linking embeddings to indexes that include them
           and then returns the generated id64.

        Args:
            index_id (str): UUID of the index.
            embedding_id (str): UUID of the embedding being linked.

        Returns:
            SqlStatement: Tuple of SQL string and parameter dict.
        """
        sql = """
            INSERT INTO index_embeddings (index_id, embedding_id)
            VALUES (:index_id, :embedding_id)
            RETURNING id64;
        """
        params = {
            "index_id": index_id,
            "embedding_id": embedding_id,
        }
        return (sql, params)

    def insert_embedding_tags_rows(
        self,
        tag_ids: list[int],
        embedding_id: str,
    ) -> SqlBatchStatement:
        """
        Create a batch INSERT statement linking an embedding to multiple tags.

        Args:
            tag_ids: List of tag IDs to associate with the embedding.
            embedding_id: UUID of the embedding being tagged.

        Returns:
            SqlBatchStatement: Tuple of SQL string and list of parameter dicts.
        """
        sql = """
            INSERT OR IGNORE INTO embedding_tags (tag_id, embedding_id)
            VALUES (:tag_id, :embedding_id)
        """
        params = [
            {
                "tag_id": tag_id,
                "embedding_id": embedding_id,
            }
            for tag_id in tag_ids
        ]
        return (sql, params)

    def insert_tags_rows(
        self,
        tag_names: list[str],
    ) -> SqlBatchStatement:
        """
        Create a batch INSERT statement for multiple tags.

        Args:
            tag_names: List of tag names.

        Returns:
            SqlBatchStatement: Tuple of SQL string and list of parameter dicts.
        """
        sql = """
            INSERT OR IGNORE INTO tags (name)
            VALUES (:tag_name)
        """
        params = [
            {
                "tag_name": tag_name,
            }
            for tag_name in tag_names
        ]
        return (sql, params)

    def select_tag_ids(self, tag_names: list[str]) -> SqlStatement:
        """
        Build a SELECT query to retrieve tag_ids for the given tag names.

        Args:
            tag_names: List of tag names to look up.

        Returns:
            SqlStatement: Tuple of SQL string and parameter dict.

        Raises:
            ValueError: If tag_names is empty.
        """
        if not tag_names:
            raise ValueError("tag_names must be non-empty.")

        n = len(tag_names)
        placeholders = ", ".join(f":name{i}" for i in range(n))

        sql = f"""
            SELECT tag_id
            FROM tags
            WHERE name IN ({placeholders})
        """
        params = {f"name{i}": tag_names[i].strip().lower() for i in range(n)}
        return (sql, params)

    def select_ids(self, ids: Ids, index_id: str, status: str) -> SqlStatement:
        """Build a SELECT query to retrieve embeddings by semantic map IDs.

        Args:
            ids (Ids): Numpy array of semantic map integer IDs.
            index_id (str): ID of the semantic map index being queried.
            status (str): Filter value for embedding status ('active', 'archived').

        Returns:
            SqlStatement: Tuple of SQL string and parameter dict.

        Raises:
            ValueError: If the provided ID list is empty.
        """
        if ids.size == 0:
            raise ValueError("Ids must be non-empty.")

        n = len(ids)
        values = ", ".join(f":id{i}" for i in range(n))

        sql = f"""
            SELECT e.embedding_id, ie.id64, e.body, e.status, e.created_at, COALESCE(json_group_array(t.name), json('[]')) AS tags_json
            FROM index_embeddings ie
            JOIN embeddings e ON e.embedding_id = ie.embedding_id
            LEFT JOIN embedding_tags et ON et.embedding_id = e.embedding_id
            LEFT JOIN tags t ON t.tag_id = et.tag_id
            WHERE e.status = :status
                AND ie.index_id = :index_id
                AND ie.id64 IN ({values})
            GROUP BY e.embedding_id, ie.id64, e.body, e.status, e.created_at
            ORDER BY ie.id64
        """
        params = {
            "index_id": index_id,
            "status": status,
        }
        params.update({f"id{i}": int(ids[i]) for i in range(n)})
        return (sql, params)

    def select_indexes(self) -> SqlStatement:
        """Return a SELECT statement to fetch all available indexes.

        Returns:
            SqlStatement: Tuple of SQL string and empty parameter dict.
        """
        sql = """
            SELECT index_id, index_name, algorithm, created_at
            FROM indexes
            ORDER BY created_at DESC;
        """
        return (sql, {})

    def update_index_embeddings_status(self, id64: int) -> SqlStatement:
        """Return the SQL statement to mark an index_embedding row as complete.

        Updates the `status` field in the `index_embeddings` table to `'complete'`
        for the specified `id64`. Typically called after an embedding has been
        successfully integrated into its FAISS index.

        Args:
            id64 (int): The primary key identifier of the index_embeddings record.

        Returns:
            SqlStatement: A tuple containing the parameterized SQL query and
            its associated named parameters.
        """
        sql = """
            UPDATE index_embeddings
            SET status = 'complete'
            WHERE id64 = :id64;
        """
        params = {
            "id64": id64,
        }
        return (sql, params)
    
    def select_random_thoughts(self, index_id: str, limit: int) -> SqlStatement:
        """
        Select a random batch of active thoughts for drill/review mode.

        Args:
            index_id (str): ID of the semantic map index being queried.
            limit: Number of thoughts to return.

        Returns:
            SqlStatement: Tuple of SQL string and parameter dict.
        """
        if limit <= 0:
            raise ValueError("limit must be > 0")

        sql = """
            SELECT ie.id64, e.embedding_id, e.body, e.status, e.created_at, COALESCE(json_group_array(t.name), json('[]')) AS tags_json
            FROM index_embeddings ie
            JOIN embeddings e ON e.embedding_id = ie.embedding_id
            LEFT JOIN embedding_tags et ON et.embedding_id = e.embedding_id
            LEFT JOIN tags t ON t.tag_id = et.tag_id
            WHERE 
                ie.index_id = :index_id
                AND e.status = 'active'
            GROUP BY ie.id64, e.embedding_id, e.body, e.status, e.created_at
            ORDER BY RANDOM()
            LIMIT :limit
        """
        params = {
            "index_id": index_id,
            "limit": limit
        }
        return (sql, params)

    def create_tables(self) -> list[SqlStatement]:
        """Return the SQL statements required to create the database schema.

        Each statement is idempotent and safe to execute multiple times.
        The method returns a list of (sql, params) tuples for sequential execution.

        Returns:
            list[SqlStatement]: List of SQL statements to initialize the schema.
        """
        create_embeddings_table_sql = """
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                body TEXT NOT NULL,
                tags TEXT NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT CHECK(status IN ('active', 'archived')) DEFAULT 'active',
                embedding BLOB NOT NULL
            );
        """

        create_tags_table_sql = """
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
        """

        create_embedding_tags_table_sql = """
            CREATE TABLE IF NOT EXISTS embedding_tags (
                tag_id INTEGER NOT NULL,
                embedding_id TEXT NOT NULL,
                PRIMARY KEY (embedding_id, tag_id),
                FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE,
                FOREIGN KEY (embedding_id) REFERENCES embeddings(embedding_id) ON DELETE CASCADE
            );
        """

        create_indexes_table_sql = """
            CREATE TABLE IF NOT EXISTS indexes (
                index_id TEXT PRIMARY KEY,
                index_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                algorithm TEXT NOT NULL,
                UNIQUE (index_name)
            );
        """

        create_index_embeddings_table_sql = """
            CREATE TABLE IF NOT EXISTS index_embeddings (
                id64 INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT NOT NULL,
                embedding_id TEXT NOT NULL,
                status TEXT CHECK(status IN ('pending', 'complete')) DEFAULT 'pending',
                UNIQUE (index_id, embedding_id),
                FOREIGN KEY (index_id) REFERENCES indexes(index_id) ON DELETE CASCADE,
                FOREIGN KEY (embedding_id) REFERENCES embeddings(embedding_id) ON DELETE CASCADE
            );
        """

        create_table_commands = [
            create_embeddings_table_sql,
            create_tags_table_sql,
            create_embedding_tags_table_sql,
            create_indexes_table_sql,
            create_index_embeddings_table_sql,
        ]
        return [(sql, {}) for sql in create_table_commands]
