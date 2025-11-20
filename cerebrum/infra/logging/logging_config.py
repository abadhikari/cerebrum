import logging


def init_logging(verbose: bool = False, debug: bool = False) -> None:
	"""
    Initialize global logging configuration.

    Args:
        verbose: If True, set level to INFO.
        debug: If True, set level to DEBUG (overrides verbose).
    """
	if debug:
		level = logging.DEBUG
	elif verbose:
		level = logging.INFO
	else:
		level = logging.WARNING
	
	logging.basicConfig(
		level=level,
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
	)
