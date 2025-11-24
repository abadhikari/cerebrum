from datetime import datetime, timezone

from cerebrum.cli.command import Command
from cerebrum.cli.input_reader import InputReader
from cerebrum.cli.prompts import CEREBRUM_CHAT_SYSTEM_PROMPT
from cerebrum.cli.spinner import typewriter_spinner
from cerebrum.core.language_model import LanguageModel
from cerebrum.core.search import SearchResult


class CerebrumChat:
    """
    Orchestrates the interactive "Ask Cerebrum" chat loop.

    This class owns *only* the LLM-driven chat experience.
    """

    def __init__(self, model: LanguageModel, input_reader: InputReader):
        """
        Initialize the CerebrumChat controller.

        Args:
            model:
                The language model used to generate assistant messages.
            input_reader:
                Provider for user input (text/voice), keeping I/O isolated
                from chat logic.
        """
        self._model = model
        self._input_reader = input_reader

    def run(self, query: str, search_result: SearchResult):
        """Start a full Cerebrum chat session for a single user query."""
        formatted_search_result = self._format_search_results_for_context(search_result)
        result_context = (
            "=== Retrieved Thoughts ===\n"
            f"{formatted_search_result}\n"
            "=== End Thoughts ==="
        )
        now = datetime.now(timezone.utc)
        system_context = (
            CEREBRUM_CHAT_SYSTEM_PROMPT
            + f"\nCurrent date (for reference): {now.isoformat()}\n"
        )
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": result_context},
            {"role": "user", "content": query},
        ]
        self._ask_cerebrum_chat_loop(messages)

    def _format_search_results_for_context(self, search_result: SearchResult) -> str:
        lines = []
        for hit in search_result.hits:
            record = hit.record
            line = (
                f"[rank={hit.rank}] score={hit.score:.3f}\n"
                f"tags: {record.tags}\n"
                f"thought: {record.body}\n"
                f"created_at: {record.created_at.isoformat()}"
            )
            lines.append(line)
        return "\n\n".join(lines)

    def _ask_cerebrum_chat_loop(self, messages: list):
        print("\n---- START OF CEREBRUM CHAT ----\n")
        while True:
            with typewriter_spinner(messages=["Thinking..."]):
                response = self._model.call(messages)
            print(f"Cerebrum: {response}")
            messages.append({"role": "assistant", "content": response})

            user_input = self._input_reader.text("\nYou", allow_voice=True)
            if user_input.lower() == Command.QUIT:
                print("\n---- END OF CHAT ----\n")
                break
            messages.append({"role": "user", "content": user_input})
