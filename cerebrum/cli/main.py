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
from cerebrum.cli.input_reader import InputReader
from cerebrum.cli.cerebrum_chat import CerebrumChat
from cerebrum.cli.thought_coach import ThoughtCoach

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for log verbosity and debugging."""
    parser = argparse.ArgumentParser(prog="cerebrum")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logs")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable DEBUG logs")
    return parser.parse_args(argv)


def _build_cli_session(container: Container):
    language_model = container.language_model
    input_reader = InputReader(container.speech_to_text)
    cerebrum_chat = CerebrumChat(language_model, input_reader)
    thought_coach = ThoughtCoach(language_model, input_reader, container.service)
    return CliSession(
        container.service,
        input_reader,
        cerebrum_chat,
        thought_coach,
    )


def run_cli(container: Container) -> int:
    """
    Run the interactive CLI session for Cerebrum.

    Handles:
    - session creation
    - session execution
    - guaranteed cleanup via container.stop()
    """
    try:
        cli_session = _build_cli_session(container)
        cli_session.run_session()
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
        with typewriter_spinner(
            messages=["Booting Cerebrum ...", "(this might take a while)"]
        ):
            container.start()
    except Exception:
        logger.exception("Error during Cerebrum bootstrap:")
        return 1

    return run_cli(container)


if __name__ == "__main__":
    raise SystemExit(main())
