"""
Cerebrum Maintenance CLI

Provides an internal command-line interface for running operational tasks
that mutate or inspect Cerebrum’s underlying data stores. This entrypoint is
not part of the public user-facing CLI; it is intended for one-off migrations,
data backfills, integrity checks, and other maintenance workflows.

Features:
    - Global logging controls (--verbose, --debug)
    - Subcommand dispatcher for individual maintenance operations
    - Current supported command:
        * backfill-tags: Migrate legacy tag JSON into the normalized tags schema

Future maintenance tasks (e.g., FAISS rebuilds, schema migrations, exports)
should be added as additional subcommands and routed through `main()`.
"""

import argparse
from enum import StrEnum

from dotenv import load_dotenv

from cerebrum.application.config import Config
from cerebrum.infra.logging.logging_config import init_logging
from cerebrum.maintenance.backfill_tags import backfill_tags


class CommandArgs(StrEnum):
    BACKFILL_TAGS = "backfill-tags"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse maintenance CLI arguments.

    Global flags:
        --verbose / -v : enable INFO logging
        --debug   / -d : enable DEBUG logging

    Subcommands:
        backfill-tags  : run the tags backfill
    """
    parser = argparse.ArgumentParser(
        prog="cerebrum-maintenance",
        description="Cerebrum internal maintenance utilities",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logs")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable DEBUG logs")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        CommandArgs.BACKFILL_TAGS.value,
        help="Run the tags backfill migration",
    )
    return parser.parse_args(argv)


def main() -> None:
    load_dotenv()
    args = parse_args()
    init_logging(verbose=args.verbose, debug=args.debug)

    config = Config()

    if args.command == CommandArgs.BACKFILL_TAGS:
        backfill_tags(config)


if __name__ == "__main__":
    raise SystemExit(main())
