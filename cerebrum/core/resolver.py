from typing import Protocol

from cerebrum.core.search import SearchHit, SearchResult


class Resolver(Protocol):
    """
    Protocol for interpreting raw semantic search output.

    A Resolver takes the unfiltered list of SearchHit objects returned from
    the semantic store and maps them into a structured SearchResult.
    This may include applying thresholds, removing low-quality hits, or
    classifying the result into different status categories.
    """

    def resolve(self, search_hits: list[SearchHit]) -> SearchResult:
        """
        Convert raw search hits into a finalized SearchResult.

        Args:
            search_hits: The ordered list of SearchHit objects returned
                         directly from the semantic store.

        Returns:
            SearchResult: The structured outcome of the search.
        """
        ...
