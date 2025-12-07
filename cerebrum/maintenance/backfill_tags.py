import json
import shutil
from pathlib import Path

from cerebrum.application.config import Config
from cerebrum.infra.db import SqlClient, SqliteClient, SqliteSqlProducer
from cerebrum.infra.db.sql.sql_client import Rows


def read_tag_ids(rows: Rows) -> list[int]:
    return [int(row["tag_id"]) for row in rows]


def read_legacy_tags(rows: Rows) -> list[tuple[str, list[str]]]:
    legacy_tags = []
    for row in rows:
        embedding_id = row["embedding_id"]
        raw_tags = row["tags"]
        if not raw_tags:
            continue
        tags = json.loads(raw_tags)
        legacy_tags.append((embedding_id, tags))
    return legacy_tags


def canonicalize_tag_names(tag_names: list[str]) -> list[str]:
    tags = {tag.strip().lower() for tag in tag_names if tag and tag.strip()}
    return sorted(tags)


def run_backfill_tags(sql_client: SqlClient, sql_producer: SqliteSqlProducer) -> None:
    """
    Perform the tags backfill.

    Selects all embeddings that (1) have legacy tags in the old JSON field, and
    (2) do not yet have rows in embedding_tags. Inserts canonicalized tags into
    the tags table and links them via embedding_tags. Runs inside a transaction.
    """

    sql = """
		SELECT e.embedding_id, e.tags
		FROM embeddings e
		LEFT JOIN embedding_tags et ON et.embedding_id = e.embedding_id
		WHERE e.tags IS NOT NULL AND et.embedding_id IS NULL;
	"""
    rows = sql_client.query(sql)
    legacy_tags = read_legacy_tags(rows)

    with sql_client.transaction():
        for embedding_id, raw_tags in legacy_tags:
            tags = canonicalize_tag_names(raw_tags)
            if not tags:
                continue

            sql, params = sql_producer.insert_tags_rows(tags)
            sql_client.execute_many(sql, params)

            sql, params = sql_producer.select_tag_ids(tags)
            tag_id_rows = sql_client.query(sql, params)
            tag_ids = read_tag_ids(tag_id_rows)

            sql, params = sql_producer.insert_embedding_tags_rows(tag_ids, embedding_id)
            sql_client.execute_many(sql, params)


def build_sql_client(db_filepath: Path) -> SqlClient:
    return SqliteClient(db_filepath)


def backup_sqlite_file(db_filepath: Path) -> None:
    backup_path = Path(str(db_filepath) + ".bak")
    shutil.copy2(db_filepath, backup_path)
    print(f"Created backup at {backup_path}")


def backfill_tags(config: Config):
    """
    High-level wrapper that backs up the DB, runs the tags backfill,
    and reports success or failure.

    This is the entrypoint invoked by the maintenance CLI.
    """
    db_filepath = config.db_filepath
    backup_sqlite_file(db_filepath)

    try:
        with build_sql_client(db_filepath) as sql_client:
            run_backfill_tags(sql_client, SqliteSqlProducer())
        print("Backfill successful")
    except Exception as e:
        print("Backfill failed:", e)
        raise
