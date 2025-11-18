from typing import Protocol


class SpeechToText(Protocol):
    """
    Interface for speech-to-text backends.

    Defines the minimal surface required for converting captured audio
    into text. Concrete implementations may choose any recording strategy
    (e.g., keypress-triggered, time-bounded, stream-driven).
    """

    def transcribe(self) -> str:
        """
        Capture audio and return the transcribed text.

        Returns:
            str: The recognized speech content.
        """
        ...
