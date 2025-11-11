from cerebrum.infra.db.sql.sql_client import SqlClient
from cerebrum.infra.db.sql.sqlite_sql_producer import SqliteSqlProducer


class SqliteSchemaManager:
    """
    Initializes the SQLite schema using a provided SQL client and producer.

    Executes all idempotent DDL statements (tables, indexes) within a single
    transaction to ensure consistency.
    """

    def __init__(self, client: SqlClient, producer: SqliteSqlProducer):
        """
        Args:
            client (SqlClient): Database client used to execute SQL statements.
            producer (SqliteSqlProducer): Factory responsible for generating
                parameterized SQL statements for table and index creation.
        """
        self._client = client
        self._producer = producer

    def init(self) -> None:
        """
        Initialize the SQLite schema.

        Executes all DDL statements returned by the SQL producer within a
        transaction. Each statement is designed to be idempotent, ensuring that
        multiple calls do not modify existing schema objects.

        Raises:
            Exception: Propagates any database error encountered during execution.
        """
        with self._client.transaction():
            for sql, params in self._producer.create_tables():
                self._client.execute(sql, params)
