import logging
from typing import Optional
from datetime import datetime, timezone
from enum import StrEnum

import click

from cerebrum.application.service import Service
from cerebrum.core.language_model import LanguageModel
from cerebrum.core.repository import Index
from cerebrum.core.search import SearchResult, SearchStatus
from cerebrum.core.speech_to_text import SpeechToText
from cerebrum.core.thought import Thought
from cerebrum.cli.prompts import (
    CEREBRUM_CHAT_SYSTEM_PROMPT,
    THOUGHT_COACH_SYSTEM_PROMPT,
)
from cerebrum.cli.spinner import typewriter_spinner
from cerebrum.cli.views import (
    print_banner,
    print_indexes,
    print_menu,
    print_search_result,
    print_box_text,
    print_duck,
)
from cerebrum.cli.colors import (
    success,
    error,
)

logger = logging.getLogger(__name__)


class Command(StrEnum):
    VOICE = "/v"
    QUIT = "/q"


class CliSession:
    """
    Interactive CLI session for Cerebrum.

    This is the presentation layer that:
      - manages index selection and creation,
      - captures and refines thoughts via the thought coach,
      - runs semantic queries,
      - asks the LLM to synthesize answers from retrieved thoughts.
    """

    def __init__(
        self,
        service: Service,
        model: LanguageModel,
        speech_to_text: SpeechToText,
    ):
        """
        Initialize a new CLI session.

        Args:
            service: Application service used for indexes and thought storage/query.
            model: Language model used for both Cerebrum and the thought coach.
        """
        self._service = service
        self._model = model
        self._speech_to_text = speech_to_text

        self._selected_index: Optional[Index] = None
        self._should_exit = False

        self._menu_actions = {
            "1": ("Add Thought", self._action_add_thought),
            "2": ("Ask Cerebrum", self._action_ask_cerebrum),
            "3": ("Talk to Duck", self._action_talk_to_duck),
            "4": ("Exit Cerebrum", self._action_close_cerebrum),
        }

    def run_session(self) -> None:
        """
        Start the interactive CLI loop.

        Ensures an index is selected, then repeatedly:
          - shows the menu,
          - reads user input,
          - dispatches to the corresponding action.
        """
        print_banner()
        while True:
            if not self._selected_index:
                index = self._select_index()
                print_box_text(
                    f"Selected Index: {index.index_name}, created: {index.created_at.isoformat()}"
                )

            print_menu(self._menu_actions)
            choice = input("\nEnter choice (number): ").strip()
            entry = self._menu_actions.get(choice)

            if entry is None:
                print(error("Invalid Option.\n"))
            else:
                _, menu_action = entry
                menu_action()

            if self._should_exit:
                break

    def _select_index(self) -> Index:
        indexes = self._service.get_indexes()
        if not indexes:
            print(error("No indexes currently exist! You will need to create one."))
            index_id = self._create_index()
            return self._select_index_with_id(index_id)

        if len(indexes) == 1:
            index = indexes[0]
            self._selected_index = index
            return index

        indexes_map = print_indexes(indexes)
        while True:
            index_key = input("\nPlease select an index out of the above: ")
            index = indexes_map.get(index_key)

            if index:
                self._selected_index = index
                return index
            print(error("Invalid index selected. Try again"))

    def _create_index(self) -> str:
        print("\n==== INDEX CREATION ====\n")
        index_name = input("Enter the name of the index to create: ")
        algorithm = input("Enter the name of the algorithm of the index: ")
        return self._service.create_index(index_name, algorithm)

    def _select_index_with_id(self, index_id: str) -> Index:
        index = self._service.get_index_by_id(index_id)
        if index is None:
            raise RuntimeError(f"Index not found: {index_id}")
        self._selected_index = index
        return index

    def _require_index(self) -> Index:
        """
        Return the currently selected index.

        This enforces the invariant that all CLI actions must operate
        on a valid, user-chosen index.
        """
        if self._selected_index is None:
            print(error("No index selected. Select an index first."))
            self._select_index()
        return self._selected_index

    def _action_add_thought(self) -> None:
        print("\n==== ADD THOUGHT ====\n")
        index = self._require_index()

        body = self._thought_coach_loop(index)
        if body is None:
            print("\nAdd thought canceled.\n")
            return

        tags = self._read_tags_input()
        print()
        thought = Thought(body, tags)

        try:
            self._service.add_thought(thought, index.index_id)
            print(success("\nThought saved.\n"))
        except Exception:
            error_msg = "Failed to save thought"
            logger.exception(error_msg)
            print(error(f"\n{error_msg}\n"))

    def _thought_coach_loop(self, index: Index) -> str | None:
        """
        Run the iterative Cerebrum Thought Coach refinement loop.

        This enforces a minimum quality bar and reduces noise in stored thoughts.
        The loop is capped at max_num_loops and exits early if the user accepts.

        Args:
            index: The active Index to query for potential collisions.

        Returns:
            The final thought text after zero or more refinement steps or None.
        """
        body = self._capture_text_input(
            f"Enter your thought ({Command.QUIT} to cancel)"
        )
        if body in {Command.QUIT, ""}:
            return None

        max_num_loops = 3
        for _ in range(max_num_loops):
            self._run_thought_coach_round(body, index)

            rewrite_decision = (
                input(f"\nWould you like to rewrite? (y/n, {Command.QUIT} to cancel): ")
                .strip()
                .lower()
            )
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

    def _capture_text_input(self, input_text: str) -> str:
        command_options = f"{Command.VOICE} for voice input"
        thought = input(f"{input_text} [{command_options}]: ")

        if thought == Command.VOICE:
            thought = self._speech_to_text.transcribe()
            print(f"Recorded: {thought}")
        return thought.strip()

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

    def _read_tags_input(self) -> list[str]:
        """
        Prompt for comma-separated tags and return a de-duplicated list.

        Returns:
            A list of unique, stripped tag strings.
        """
        raw_tags = input("Enter your comma separated tags: ").split(",")

        tags = []
        seen = set()
        for raw_tag in raw_tags:
            tag = raw_tag.strip()
            if tag and tag not in seen:
                tags.append(tag)
                seen.add(tag)
        return tags

    def _action_query_thoughts(self) -> None:
        print("\n==== QUERY THOUGHTS ====\n")
        query = input("Enter your query: ")
        k = int(input("Enter your k value: "))

        index = self._require_index()
        search_result = self._service.query(query, index.index_id, k)

        error_msg = self._validate_search_result(search_result)
        if error_msg:
            print(error(f"\n{error_msg}"))
            return

        print_search_result(search_result)

    def _validate_search_result(self, search_result: SearchResult) -> Optional[str]:
        if search_result.status == SearchStatus.NO_EMBEDDINGS:
            return "No embeddings exist yet. Add thoughts first."

        if search_result.status == SearchStatus.NO_MATCHES:
            return "No matching thoughts found."

        return None

    def _action_create_index(self) -> None:
        index_id = self._create_index()
        self._select_index_with_id(index_id)

    def _action_ask_cerebrum(self) -> None:
        """
        Run a semantic query and ask the LLM to synthesize an answer
        from the retrieved thoughts.

        Optionally prints the raw semantic search hits before the answer.
        """
        print("\n==== ASK CEREBRUM ====\n")
        query = self._capture_text_input("Enter your query")
        k = int(input("Enter your k value: "))
        see_semantic_results = input(
            "Would you like to see the semantic results? (y/n): ",
        )

        index = self._require_index()
        search_result = self._service.query(query, index.index_id, k)

        error_msg = self._validate_search_result(search_result)
        if error_msg:
            print(error(f"\n{error_msg}"))
            return

        if see_semantic_results.strip().lower() == "y":
            print_search_result(search_result)

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

            user_input = self._capture_text_input("\nYou").strip().lower()
            if user_input == Command.QUIT:
                print("\n---- END OF CHAT ----\n")
                break
            messages.append({"role": "user", "content": user_input})

    def _action_talk_to_duck(self) -> None:
        print("\n==== TALK TO DUCK ====\n")
        total_text = ""
        while True:
            print_duck()
            text = self._capture_text_input(
                "Talk to duck (or say 'thanks duck' to finish)"
            )
            if text == "thanks duck":
                break
            total_text += f"{text}\n"
        print("\n---- Session ----\n")
        print(total_text)
        print("-----------------")

    def _action_close_cerebrum(self):
        print("Exiting Cerebrum...")
        self._should_exit = True
