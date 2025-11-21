import os

from dotenv import load_dotenv

from cerebrum.application.config import Config
from cerebrum.application.container import Container


def init_environment() -> None:
    """
    One-time process bootstrap.

    - Set env knobs (like OpenMP threads)
    - Load .env file
    """
    # Prevent OpenMP thread contention
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Load env configuration
    load_dotenv()


def build_container(config: Config) -> Container:
    """
    Assemble and return the application's dependency container.

    This function acts as the composition root: it loads configuration,
    constructs infrastructure components, and wires them into a single
    `Container` instance. Callers use this to bootstrap the application
    without needing to know how individual dependencies are created.

    Returns:
        Container: A fully initialized dependency container.
    """
    return Container(
        config=config,
    )
