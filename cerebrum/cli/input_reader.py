from cerebrum.core.speech_to_text import SpeechToText
from cerebrum.cli.command import Command


class InputReader:
    """
    Unified wrapper for all user input in the CLI.

    Centralizes text and integer input handling, including optional
    speech-to-text activation when the VOICE command is supplied.
    This keeps raw `input()` calls out of higher-level logic and
    ensures consistent prompt formatting and validation across the
    CLI layer.
    """

    def __init__(self, speech_to_text: SpeechToText):
        """
        Initialize the input reader.

        Args:
            speech_to_text: STT engine used when voice input is triggered.
        """
        self._speech_to_text = speech_to_text

    def text(self, prompt: str, allow_voice: bool = False) -> str:
        """
        Read a line of text from the user.

        Behaviors:
          - If allow_voice=True and the user enters Command.VOICE,
            STT is invoked and its transcription is returned.
          - Leading/trailing whitespace is stripped.

        Args:
            prompt: Text shown to the user before input.
            allow_voice: Enables voice input via the /v command.

        Returns:
            The final text string entered or transcribed.
        """
        command_text = f" [{Command.VOICE} for voice input]" if allow_voice else ""
        raw = input(f"{prompt}{command_text}: ")

        if allow_voice and raw == Command.VOICE:
            raw = self._speech_to_text.transcribe()
            print(f"Recorded: {raw}")
        return raw.strip()

    def integer(self, prompt: str) -> int:
        while True:
            raw = self.text(prompt)
            try:
                return int(raw)
            except ValueError:
                print("Please enter a valid integer")
