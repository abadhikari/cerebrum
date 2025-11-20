"""
CLI entrypoint for Cerebrum.

Initializes logging, bootstraps the dependency container,
and dispatches control to the interactive CLI session.
"""

import argparse
import logging

from cerebrum.application.bootstrap import build_container
from cerebrum.presentation.cli.session import CliSession
from cerebrum.infra.logging.logging_config import init_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse CLI flags for log verbosity and debugging."""
	parser = argparse.ArgumentParser(prog="cerebrum")
	parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logs")
	parser.add_argument("--debug", "-d", action="store_true", help="Enable DEBUG logs")
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	"""
    Main CLI entrypoint.

    Configures logging, builds the application container,
    and runs the interactive CLI session. Returns a Unix-style
    exit code (0 success, 1 failure).
    """
	args = parse_args(argv)
	init_logging(verbose=args.verbose, debug=args.debug)

	logger.info("Starting Cerebrum CLI session")

	try:
		with build_container() as container:
			CliSession(
				container.service,
				container.language_model,
				container.speech_to_text
			).run_session()
	except Exception:
		logger.exception("Fatal error in Cerebrum CLI session:")
		return 1

	logger.info("Cerebrum CLI session ended")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
