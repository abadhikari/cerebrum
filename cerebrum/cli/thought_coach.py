import click

from cerebrum.core.language_model import LanguageModel
from cerebrum.core.repository import Index
from cerebrum.cli.prompts import (
    THOUGHT_COACH_SYSTEM_PROMPT,
)
from cerebrum.application.service import Service
from cerebrum.cli.spinner import typewriter_spinner
from cerebrum.cli.command import Command
from cerebrum.cli.input_reader import InputReader


class ThoughtCoach:
    """
    High-level orchestrator for refining a user’s thought before storage.

    It guides input collection, invokes the LLM for iterative feedback,
    surfaces potential duplicates, and manages optional rewrites. The final
    result is a polished thought or None if the user cancels.
    """

    def __init__(
        self, language_model: LanguageModel, input_reader: InputReader, service: Service
    ):
        """
        Initialize the ThoughtCoach.

        Args:
            language_model:
                Model used to generate coaching feedback.
            input_reader:
                Provider for user input (text/voice), keeping raw I/O
                out of coach logic.
            service:
                Application service used for semantic similarity checks
                against the active index.
        """
        self._model = language_model
        self._input_reader = input_reader
        self._service = service

    def run(self, index: Index) -> str | None:
        """
        Run the iterative Cerebrum Thought Coach refinement loop.

        This enforces a minimum quality bar and reduces noise in stored thoughts.
        The loop is capped at max_num_loops and exits early if the user accepts.

        Args:
                index: The active Index to query for potential collisions.

        Returns:
                The final thought text after zero or more refinement steps or None.
        """
        body = self._input_reader.text(
            f"Enter your thought ({Command.QUIT} to cancel)",
            allow_voice=True,
        )
        if body in {Command.QUIT, ""}:
            return None

        max_num_loops = 3
        for _ in range(max_num_loops):
            self._run_thought_coach_round(body, index)

            rewrite_decision = self._input_reader.text(
                f"\nWould you like to rewrite? (y/n, {Command.QUIT} to cancel)"
            ).lower()
            if rewrite_decision == Command.QUIT:
                return None
            if rewrite_decision != "y":
                break

            edited = click.edit(text=body, require_save=True)
            if edited is None:
                return None

            edited = edited.strip()
            if edited == "" or edited == body:
                break
            body = edited
        return body

    def _run_thought_coach_round(self, body: str, index: Index) -> str:
        """
        Run a single Thought Coach round: generate feedback, show it,
        check for similar thoughts, and print the current draft.

        Returns:
                The feedback text (useful if you ever want to reuse/reprint).
        """
        user_thought = f"user thought: {body}"
        messages = [
            {"role": "system", "content": THOUGHT_COACH_SYSTEM_PROMPT},
            {"role": "user", "content": user_thought},
        ]
        print("\n==== Thought Coach Feedback ====")
        with typewriter_spinner(["Producing feedback..."]):
            feedback = self._model.call(messages)
        print(feedback)
        self._similar_thought_check(body, index)

        print("\n---- Current Draft ----\n")
        print(body)
        print("-----------------------")
        return feedback

    def _similar_thought_check(self, body: str, index: Index) -> None:
        """
        Surface the nearest semantic neighbor for a candidate thought.

        Args:
                body: The text of the candidate thought (post-Coach).
                index: The active Index to query for potential collisions.
        """
        search_result = self._service.query(body, index.index_id, 1)
        hits = search_result.hits
        if len(hits):
            similar_thought = hits[0]
            print(f"\nClosest match: {similar_thought.record.body}")
            print(f"Similarity score: {similar_thought.score:.3f}")
            return
        print("No similar thoughts!")
