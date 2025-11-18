import torch

# Prevent thread contention for pytorch dependencies
torch.set_num_threads(1)

from dotenv import load_dotenv

from cerebrum.application.config import Config
from cerebrum.application.container import Container


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