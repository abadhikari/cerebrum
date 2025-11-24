RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"


def success(message: str) -> str:
    """
    Wrap a message in ANSI green formatting for success output.

    Args:
        message: The text to style.

    Returns:
        A string with green coloring applied and reset code appended.
    """
    return f"{GREEN}{message}{RESET}"


def error(message: str) -> str:
    """
    Wrap a message in ANSI red formatting for error output.

    Args:
        message: The text to style.

    Returns:
        A string with red coloring applied and reset code appended.
    """
    return f"{RED}{message}{RESET}"
