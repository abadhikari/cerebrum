from enum import StrEnum


class Command(StrEnum):
    """
    Canonical CLI control commands.

    These special tokens are recognized across multiple CLI components
    to provide out-of-band control signals such as:
            - VOICE: trigger speech-to-text input
            - QUIT:  terminate an ongoing multi-turn interaction
    """

    VOICE = "/v"
    QUIT = "/q"
    THOUGHT = "/t"
