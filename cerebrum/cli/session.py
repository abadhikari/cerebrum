import logging
from typing import Optional

from cerebrum.application.service import Service
from cerebrum.cli.cerebrum_chat import CerebrumChat
from cerebrum.cli.colors import (
    error,
    success,
)
from cerebrum.cli.input_reader import InputReader
from cerebrum.cli.thought_coach import ThoughtCoach
from cerebrum.cli.views import (
    print_banner,
    print_box_text,
    print_duck,
    print_indexes,
    print_menu,
    print_search_result,
)
from cerebrum.core.repository import Index
from cerebrum.core.search import SearchResult, SearchStatus
from cerebrum.core.thought import Thought

logger = logging.getLogger(__name__)


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
        input_reader: InputReader,
        cerebrum_chat: CerebrumChat,
        thought_coach: ThoughtCoach,
    ):
        """
        Initialize a new CLI session.

        Args:
            service: Application service used for indexes and thought storage/query.
        """
        self._service = service
        self._input_reader = input_reader
        self._cerebrum_chat = cerebrum_chat
        self._thought_coach = thought_coach

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
            self._require_index()

            print_menu(self._menu_actions)
            choice = self._input_reader.text("\nEnter choice (number)")
            entry = self._menu_actions.get(choice)

            if entry is None:
                print(error("Invalid Option.\n"))
            else:
                _, menu_action = entry
                menu_action()

            if self._should_exit:
                break

    def _require_index(self) -> Index:
        """
        Return the currently selected index.

        This enforces the invariant that all CLI actions must operate
        on a valid, user-chosen index.
        """
        if self._selected_index is None:
            index = self._select_index()
            print_box_text(
                f"Selected Index: {index.index_name}, created: {index.created_at.isoformat()}",
            )
            self._selected_index = index
            return index
        return self._selected_index

    def _select_index(self) -> Index:
        indexes = self._service.get_indexes()
        if not indexes:
            print(error("No indexes currently exist! You will need to create one."))
            index_id = self._create_index()
            return self._select_index_with_id(index_id)

        if len(indexes) == 1:
            return indexes[0]

        indexes_map = print_indexes(indexes)
        while True:
            index_key = self._input_reader.text(
                "\nPlease select an index out of the above",
            )
            index = indexes_map.get(index_key)

            if index:
                return index
            print(error("Invalid index selected. Try again"))

    def _create_index(self) -> str:
        print("\n==== INDEX CREATION ====\n")
        index_name = self._input_reader.text("Enter the name of the index to create")
        algorithm = self._input_reader.text(
            "Enter the name of the algorithm of the index",
        )
        return self._service.create_index(index_name, algorithm)

    def _select_index_with_id(self, index_id: str) -> Index:
        index = self._service.get_index_by_id(index_id)
        if index is None:
            raise RuntimeError(f"Index not found: {index_id}")
        return index

    def _action_add_thought(self) -> None:
        print("\n==== ADD THOUGHT ====\n")
        index = self._require_index()

        body = self._thought_coach.run(index)
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

    def _read_tags_input(self) -> list[str]:
        """
        Prompt for comma-separated tags and return a de-duplicated list.

        Returns:
            A list of unique, stripped tag strings.
        """
        while True:
            raw_tags = self._input_reader.text("Enter your comma separated tags").split(
                ","
            )
            tags = self._parse_tags(raw_tags)
            tags_text = ", ".join(tags)
            confirmation = self._input_reader.text(
                f"Do these tags ({tags_text}) look good? (y/n)"
            )
            if confirmation.lower() == "y":
                return tags

    def _parse_tags(self, raw_tags: str) -> list[str]:
        tags = []
        seen = set()
        for raw_tag in raw_tags:
            tag = raw_tag.strip()
            if tag and tag not in seen:
                tags.append(tag)
                seen.add(tag)
        return tags

    def _validate_search_result(self, search_result: SearchResult) -> Optional[str]:
        if search_result.status == SearchStatus.NO_EMBEDDINGS:
            return "No embeddings exist yet. Add thoughts first."

        if search_result.status == SearchStatus.NO_MATCHES:
            return "No matching thoughts found."

        return None

    def _action_ask_cerebrum(self) -> None:
        """
        Run a semantic query and ask the LLM to synthesize an answer
        from the retrieved thoughts.

        Optionally prints the raw semantic search hits before the answer.
        """
        print("\n==== ASK CEREBRUM ====\n")
        query = self._input_reader.text("Enter your query")
        k = self._input_reader.integer("Enter your k value")
        see_semantic_results = self._input_reader.text(
            "Would you like to see the semantic results? (y/n)",
        )

        index = self._require_index()
        search_result = self._service.query(query, index.index_id, k)

        error_msg = self._validate_search_result(search_result)
        if error_msg:
            print(error(f"\n{error_msg}"))
            return

        if see_semantic_results.lower() != "n":
            print_search_result(search_result)

        self._cerebrum_chat.run(query, search_result)

    def _action_talk_to_duck(self) -> None:
        print("\n==== TALK TO DUCK ====\n")
        total_text = ""
        while True:
            print_duck()
            text = self._input_reader.text(
                "Talk to duck (or say 'thanks duck' to finish)",
            )
            if text == "thanks duck":
                break
            total_text += f"{text}\n"
        print("\n---- Session ----\n")
        print(total_text)
        print("-----------------")

    def _action_close_cerebrum(self) -> None:
        print("Exiting Cerebrum...")
        self._should_exit = True
