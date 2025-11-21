import itertools
import sys
import threading
import time
from contextlib import contextmanager


@contextmanager
def typewriter_spinner(messages: list[str]):
    """
    Context manager that displays an animated “typewriter” spinner
    on stderr while the enclosed block executes.

    The animation progressively reveals the characters of `messages`,
    then retracts them (forward + backward). This runs in a dedicated
    background thread and updates in-place using carriage returns.

    Usage:
        with typewriter_spinner("Loading"):
            do_work()

    Args:
        message: A list of full strings to animate. The animation cycles through
                 all prefixes of the strings, then reverses.
    """
    stop_event = threading.Event()

    def build_prefixes():
        """
        Build the forward + backward sequence of prefixes for all messages.
        Example for ['abc']: ['', 'a', 'ab', 'abc', 'ab', 'a', ''].
        """
        total_prefixes = [""]
        for message in messages:
            prefixes = [""]
            buffer = []
            for letter in message:
                buffer.append(letter)
                prefixes.append("".join(buffer))
            total_prefixes += prefixes + prefixes[::-1]
        return total_prefixes

    def _run() -> None:
        """
        Background animation loop.
        Cycles through the prefix list, printing each frame,
        then clearing the line. Sleeps a bit to control speed.
        """
        full_messages = set(messages)
        prefixes = build_prefixes()
        symbols = itertools.cycle(prefixes)
        stream = sys.stderr

        print()
        while not stop_event.is_set():
            symbol = next(symbols)
            stream.write(f"\r{symbol}")
            stream.flush()
            time.sleep(0.05)
            if symbol in full_messages:
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
