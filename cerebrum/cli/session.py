from typing import Optional

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
)

VOICE_COMMAND = "/v"


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

        self._menu_actions = {
            "1": ("Add Thought", self._action_add_thought),
            "2": ("Ask Cerebrum", self._action_ask_cerebrum),
            "3": ("Create Index", self._action_create_index),
            "4": ("Select Index", self._select_index),
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
                self._select_index()

            index = self._require_index()
            print(
                f"\n----> Selected Index: name - {index.index_name}, id - {index.index_id}, created - {index.created_at.isoformat()} <----",
            )

            print_menu(self._menu_actions)
            choice = input("\nEnter choice: ").strip()
            entry = self._menu_actions.get(choice)

            if entry is None:
                print("Invalid Option.\n")
            else:
                _, menu_action = entry
                menu_action()

    def _select_index(self) -> None:
        indexes = self._service.get_indexes()
        if not indexes:
            print("No indexes currently exist! You will need to create one.")
            index_id = self._create_index()
            self._select_index_with_id(index_id)
            return

        if len(indexes) == 1:
            self._selected_index = indexes[0]
            return

        indexes_map = print_indexes(indexes)
        index_key = input("\nPlease select an index out of the above: ")
        index = indexes_map.get(index_key)

        if index:
            self._selected_index = index
        else:
            print("Invalid index selected. Try again")

    def _create_index(self) -> str:
        print("\n==== INDEX CREATION ====\n")
        index_name = input("Enter the name of the index to create: ")
        algorithm = input("Enter the name of the algorithm of the index: ")
        return self._service.create_index(index_name, algorithm)

    def _select_index_with_id(self, index_id: str) -> None:
        index = self._service.get_index_by_id(index_id)
        self._selected_index = index

    def _require_index(self) -> Index:
        """
        Return the currently selected index.

        Raises:
                RuntimeError: If no index has been selected.

        This enforces the invariant that all CLI actions must operate
        on a valid, user-chosen index.
        """
        if self._selected_index is None:
            raise RuntimeError("No index selected. Call _select_index() first.")
        return self._selected_index

    def _action_add_thought(self) -> None:
        print("\n==== ADD THOUGHT ====\n")
        raw_body = self._read_thought_input("Enter your thought")
        body = self._thought_coach_loop(raw_body)
        tags = self._read_tags_input()
        thought = Thought(body, tags)

        index = self._require_index()
        self._service.add_thought(thought, index.index_id)

    def _read_thought_input(self, input_text: str) -> str:
        command_options = f"{VOICE_COMMAND} for voice input"
        thought = input(f"{input_text} [{command_options}]: ").strip()

        if thought == VOICE_COMMAND:
            thought = self._speech_to_text.transcribe()
            print(f"Recorded thought: {thought}")
        return thought

    def _thought_coach_loop(self, initial_body: str) -> str:
        """
                Run the iterative Cerebrum Thought Coach refinement loop.

                This enforces a minimum quality bar and reduces noise in stored thoughts.
                The loop is capped at max_num_loops and exits early if the user accepts.

        Args:
            initial_body: The original, raw thought text.

        Returns:
            The final thought text after zero or more refinement steps.
        """
        body = initial_body
        max_num_loops = 3
        for _ in range(max_num_loops):
            user_thought = f"user thought: {body}"
            messages = [
                {"role": "system", "content": THOUGHT_COACH_SYSTEM_PROMPT},
                {"role": "user", "content": user_thought},
            ]
            print("\n==== Thought Coach Feedback ====\n")
            print(self._model.call(messages))

            rewrite = self._read_thought_input(
                "\nRewrite (press enter if no change required)",
            )
            if rewrite:
                body = rewrite
            else:
                break
        return body

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
            print(f"\n{error_msg}")
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
        query = input("Enter your query: ")
        k = int(input("Enter your k value: "))
        see_semantic_results = input(
            "Would you like to see the semantic results? (y/n): ",
        )

        index = self._require_index()
        search_result = self._service.query(query, index.index_id, k)

        error_msg = self._validate_search_result(search_result)
        if error_msg:
            print(f"\n{error_msg}")
            return

        if see_semantic_results.strip().lower() == "y":
            print_search_result(search_result)

        user_context = f"user_query: {query}\n" f"search_hits: {search_result.hits}"
        messages = [
            {"role": "system", "content": CEREBRUM_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_context},
        ]
        self._ask_cerebrum_chat_loop(messages)

    def _ask_cerebrum_chat_loop(self, messages: list):
        print("\n---- START OF CEREBRUM CHAT ----\n")
        while True:
            with typewriter_spinner(message="Thinking ..."):
                response = self._model.call(messages)
            print(f"Cerebrum: {response}")
            messages.append({"role": "assistant", "content": response})

            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"/quit", "/q"}:
                print("\n---- END OF CHAT ----\n")
                break
            messages.append({"role": "user", "content": user_input})
