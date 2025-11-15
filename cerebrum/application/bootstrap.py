import os
from dotenv import load_dotenv

from cerebrum.application.config import Config
from cerebrum.application.container import Container

# Prevent OpenMP thread contention 
os.environ["OMP_NUM_THREADS"] = "1"

# Load env configuration
load_dotenv()

def build_container() -> Container:
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
        config=Config()
    )