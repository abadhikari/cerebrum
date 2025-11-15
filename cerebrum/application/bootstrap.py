from cerebrum.application.config import Config
from cerebrum.application.container import Container


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
    config = Config()

    return Container(
        db_filepath=config.db_filepath,
        faiss_filepath=config.faiss_filepath,
        model_name=config.language_model_name
    )