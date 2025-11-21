"""
CLI entrypoint for Cerebrum.

Initializes logging, bootstraps the dependency container,
and dispatches control to the interactive CLI session.
"""

import argparse
import logging

from cerebrum.application.bootstrap import build_container, init_environment
from cerebrum.cli.session import CliSession
from cerebrum.infra.logging.logging_config import init_logging
from cerebrum.cli.spinner import typewriter_spinner
from cerebrum.application.container import Container
from cerebrum.application.config import Config

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for log verbosity and debugging."""
    parser = argparse.ArgumentParser(prog="cerebrum")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logs")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable DEBUG logs")
    return parser.parse_args(argv)


def run_cli(container: Container) -> int:
    """
    Run the interactive CLI session for Cerebrum.

    Handles:
    - session creation
    - session execution
    - guaranteed cleanup via container.stop()
    """
    try:
        CliSession(
            container.service, container.language_model, container.speech_to_text
        ).run_session()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Fatal error in Cerebrum CLI session:")
        return 1
    finally:
        logger.info("Cerebrum CLI session ended")
        container.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
	"""
	Main CLI entrypoint.

	Configures logging, builds the application container,
	and runs the interactive CLI session. Returns a Unix-style
	exit code (0 success, 1 failure).
	"""
	init_environment()

	args = parse_args(argv)
	init_logging(verbose=args.verbose, debug=args.debug)

	logger.info("Starting Cerebrum CLI session")
	config = Config()
	container = build_container(config)

	try:
		with typewriter_spinner("Booting Cerebrum (this might take awhile)"):
			container.start()
	except Exception:
		logger.exception("Error during Cerebrum bootstrap:")
		return 1

	return run_cli(container)


if __name__ == "__main__":
    raise SystemExit(main())
