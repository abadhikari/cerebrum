from cerebrum.application.service import SearchHit, Service
from cerebrum.core.language_model import LanguageModel
from cerebrum.core.thought import Thought
from cerebrum.core.repository import Index

from typing import Optional

CEREBRUM_CHAT_ASCII = r"""
_________                     ___.                         _________ .__            __   
\_   ___ \  ___________   ____\_ |_________ __ __  _____   \_   ___ \|  |__ _____ _/  |_ 
/    \  \/_/ __ \_  __ \_/ __ \| __ \_  __ \  |  \/     \  /    \  \/|  |  \\__  \\   __\
\     \___\  ___/|  | \/\  ___/| \_\ \  | \/  |  /  Y Y  \ \     \___|   Y  \/ __ \|  |  
 \______  /\___  >__|    \___  >___  /__|  |____/|__|_|  /  \______  /___|  (____  /__|  
        \/     \/            \/    \/                  \/          \/     \/     \/      

"""

class CliSession:
	"""
    Interactive CLI session for Cerebrum.

    This is the presentation layer that:
      - manages index selection and creation,
      - captures and refines thoughts via the thought coach,
      - runs semantic queries,
      - asks the LLM to synthesize answers from retrieved thoughts.
    """

	def __init__(self, service: Service, model: LanguageModel):
		"""
        Initialize a new CLI session.

        Args:
            service: Application service used for indexes and thought storage/query.
            model: Language model used for both Cerebrum Chat and the thought coach.
        """
		self._service = service
		self._model = model

		self._selected_index: Optional[Index] = None

		self._menu_actions = {
			"1": ("Add Thought", self._action_add_thought),
			"2": ("Query Thoughts", self._action_query_thoughts),
			"3": ("Create Index", self._action_create_index),
			"4": ("Select Index", self._select_index),
			"5": ("Ask Cerebrum", self._action_ask_cerebrum)
		}

	def run_session(self) -> None:
		"""
        Start the interactive CLI loop.

        Ensures an index is selected, then repeatedly:
          - shows the menu,
          - reads user input,
          - dispatches to the corresponding action.
        """
		print(CEREBRUM_CHAT_ASCII)
		while True:
			if not self._selected_index:
				self._select_index()
			
			index = self._require_index()
			print(f"\n----> Selected Index: name - {index.index_name}, id - {index.index_id}, created - {index.created_at.isoformat()} <----")

			self._print_menu()
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

		indexes_map = self._print_indexes(indexes)
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

	def _print_indexes(self, indexes: list[Index]) -> dict[str, Index]:
		indexes_map = {}
		print("\n=== Indexes List ===\n")
		for i, index in enumerate(indexes):
			print(f"{i + 1}. {index}")
			indexes_map[str(i + 1)] = index
		return indexes_map

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

	def _print_menu(self) -> None:
		print("\n=== MENU ===")
		for key, (label, _) in self._menu_actions.items():
			print(f"{key}. {label}")

	def _action_add_thought(self) -> None:
		print("\n==== ADD THOUGHT ====\n")
		raw_body = input("Enter your thought: ")
		body = self._thought_coach_loop(raw_body)
		tags = self._get_tags()
		thought = Thought(body, tags)

		index = self._require_index()
		self._service.add_thought(thought, index.index_id)

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
		max_num_loops =  3
		system_context = self._get_thought_coach_system_context()
		for _ in range(max_num_loops):
			user_thought = f"user thought: {body}"
			messages = [
				{"role": "system", "content": system_context},
				{"role": "user", "content": user_thought}
			]
			print("\n==== Thought Coach Feedback ====\n")
			print(self._model.call(messages))

			rewrite = input("\nRewrite (press enter if no change required): ").strip()
			if rewrite:
				body = rewrite
			else:
				break
		return body
	
	def _get_thought_coach_system_context(self) -> str:
		"""
		Build the system prompt for the Cerebrum Thought Coach.

		The returned text defines the required output format (Verdict, Suggestion, Tags)
		and the quality bar used to decide whether a thought should be stored.
		"""
		return (
			"You're Cerebrum Coach. Evaluate if the thought is worth storing and propose a few useful tags. "
			"A strong thought has: (1) a concrete situation, (2) a clear realization, "
			"(3) an optional pattern, and (4) some reusable insight (implicit is fine). "
			"Output must be concise. No paragraphs. No filler. "
			"Do NOT restate the thought.\n"
			"Format:\n"
			"Verdict: weak/ good/ strong\n"
			"Suggestion: <one short clause about how the thought could be improved/ which aspect is lacking. " 
			"If already addressed, then can put 'Looks good!' instead'>\n"
			"Tags: 2–5 lowercase, hyphen-separated identity keywords \n"
			"Tag Example: the-count-of-monte-cristo, alexandre-dumas, revenge\n"
			"Do NOT collapse multiword concepts into one word. "
			"Do NOT shorten titles for tags. "
			"Do NOT rewrite or expand the thought. "
		)

	def _get_tags(self) -> list[str]:
		"""
        Prompt for comma-separated tags and return a de-duplicated list.

        Returns:
            A list of unique, stripped tag strings.
        """
		raw_tags = input("Enter your comma separated tags: ").split(',')

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
		search_hits = self._service.query(query, index.index_id, k)
		self._print_search_hits(search_hits)

	def _print_search_hits(self, search_hits: list[SearchHit]) -> None:
		print("\n===== Results =====\n")
		for hit in search_hits:
			print(
				f"thought: {hit.record.body}\n"
				f"tags: {hit.record.tags}\n"
				f"score: {hit.score}\n"
			)

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
		see_semantic_results = input("Would you like to see the semantic results? (y/n): ")

		index = self._require_index()
		search_hits = self._service.query(query, index.index_id, k)

		if see_semantic_results == "y":
			self._print_search_hits(search_hits)

		system_context = (
			"You're called cerebrum chat. You'll receive my query to the cerebrum semantic map and then give an "
			"answer based on the results. You are my personal memory assistant. "
			"Given a list of retrieved thoughts, answer in concise, but not short sentences. "
			"Avoid fluff like ‘based on your query’ or ‘users’. Speak directly and concretely. "
			"If older thoughts conflict with newer ones, treat newer ACTIVE thoughts as my current view "
			"and older ones as historical context."
		)
		user_context = (
			f"user_query: {query}\n"
			f"search_hits: {search_hits}"
		)
		messages = [
			{"role": "system", "content": system_context},
			{"role": "user", "content": user_context}
		]
		print("\n==== Cerebrum Response ====\n")
		print(self._model.call(messages))
