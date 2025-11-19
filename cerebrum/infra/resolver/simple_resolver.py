from cerebrum.core.search import SearchHit, SearchResult, SearchStatus


class SimpleResolver:
	"""
    A minimal resolver that applies a fixed similarity threshold.

    This implementation filters out any SearchHit whose score is below
    `min_allowed_score`, and classifies the result as:
    """

	def __init__(self, min_allowed_score):
		"""
        Initialize the resolver.

        Args:
            min_allowed_score: The minimum similarity score a SearchHit must satisfy
                               in order to be included in the final result.
        """
		self._min_allowed_score = min_allowed_score
	
	def resolve(self, search_hits: list[SearchHit]) -> SearchResult:
		"""
        Filter and classify raw search hits.

        Args:
            search_hits: The ordered list of SearchHit objects returned 
                         directly from the semantic store.

        Returns:
            SearchResult: The structured outcome of the search.
        """
		filtered_search_hits = []
		for search_hit in search_hits:
			if search_hit.score < self._min_allowed_score:
				continue
			filtered_search_hits.append(search_hit)

		search_status = SearchStatus.NO_MATCHES if len(filtered_search_hits) == 0 else SearchStatus.OK
		return SearchResult(
			status=search_status,
			hits=filtered_search_hits
		)