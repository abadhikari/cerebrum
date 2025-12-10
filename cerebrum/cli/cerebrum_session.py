from enum import StrEnum
from datetime import datetime, timezone

from cerebrum.cli.command import Command
from cerebrum.cli.input_reader import InputReader
from cerebrum.cli.prompts import CEREBRUM_SESSION_SYSTEM_PROMPT
from cerebrum.cli.spinner import typewriter_spinner
from cerebrum.cli.views import print_box_text
from cerebrum.core.language_model import ChatMessage, LanguageModel


class CerebrumSessionResult(StrEnum):
    """
    Terminal outcomes for a Cerebrum Session.

    These values indicate how the interactive session concluded and which
    high-level flow should execute next (exit vs transitioning into thought
    creation).
    """
    DONE = "done"
    ADD_THOUGHT = "add_thought"


class CerebrumSession:
    """
    Orchestrates the interactive "Cerebrum Session" loop.

    This class owns *only* the LLM-driven session experience.
    """

    def __init__(self, model: LanguageModel, input_reader: InputReader):
        """
        Initialize the CerebrumSession controller.

        Args:
            model:
                The language model used to generate assistant messages.
            input_reader:
                Provider for user input (text/voice).
        """
        self._model = model
        self._input_reader = input_reader

    def run(self) -> CerebrumSessionResult:
        """Run an interactive Cerebrum Session and return the final result."""
        return self._cerebrum_session_loop()

    def _cerebrum_session_loop(self) -> CerebrumSessionResult:
        print("\n---- START OF CEREBRUM SESSION ----\n")
        print_box_text("(Enter /q to quit and /t to go to add thought)")
        messages = self._init_context()
        while True:
            user_input = self._input_reader.text("\nYou", allow_voice=True)

            lowercase_input = user_input.lower()
            if lowercase_input == Command.QUIT:
                print()
                print("---- END OF SESSION ----\n")
                return CerebrumSessionResult.DONE
            if lowercase_input == Command.THOUGHT:
                print()
                print("---- Moving to add thought ----")
                return CerebrumSessionResult.ADD_THOUGHT

            messages.append({"role": "user", "content": user_input})

            with typewriter_spinner(messages=["Thinking..."]):
                response = self._model.call(messages)
            print()
            print_box_text("Cerebrum:")
            print(response)
            messages.append({"role": "assistant", "content": response})

    def _init_context(self) -> list[ChatMessage]:
        now = datetime.now(timezone.utc)
        system_context = (
            CEREBRUM_SESSION_SYSTEM_PROMPT
            + f"\nCurrent date (for reference): {now.isoformat()}\n"
        )
        messages = [
            {"role": "system", "content": system_context},
        ]
        return messages
