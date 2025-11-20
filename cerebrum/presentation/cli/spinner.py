import sys
import time
import threading
import itertools
from contextlib import contextmanager


@contextmanager
def typewriter_spinner(message: str = "..."):
	"""
    Context manager that displays an animated “typewriter” spinner
    on stderr while the enclosed block executes.

    The animation progressively reveals the characters of `message`,
    then retracts them (forward + backward). This runs in a dedicated
    background thread and updates in-place using carriage returns.

    Usage:
        with typewriter_spinner("Loading"):
            do_work()

    Args:
        message: The text to animate. The animation cycles through
                 all prefixes of this string, then reverses.
    """
	stop_event = threading.Event()

	def build_prefixes():
		"""
        Build the forward + backward sequence of prefixes.
        Example for 'abc': ['', 'a', 'ab', 'abc', 'ab', 'a', ''].
        """
		prefixes = [""]
		buffer = []
		for letter in message:
			buffer.append(letter)
			prefixes.append("".join(buffer))
		return prefixes + prefixes[::-1]

	def _run() -> None:
		"""
        Background animation loop.
        Cycles through the prefix list, printing each frame,
        then clearing the line. Sleeps a bit to control speed.
        """
		prefixes = build_prefixes()
		symbols = itertools.cycle(prefixes)
		stream = sys.stderr

		print()
		while not stop_event.is_set():
			symbol = next(symbols)
			stream.write(f"\r{symbol}")
			stream.flush()
			time.sleep(0.05)
			if symbol == message:
				time.sleep(0.2)
			stream.write("\r\033[K")
			stream.flush()
	
	thread = threading.Thread(target=_run, daemon=True)
	thread.start()

	try:
		yield
	finally:
		stop_event.set()
		thread.join()
