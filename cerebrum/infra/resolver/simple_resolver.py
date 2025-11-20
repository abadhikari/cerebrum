from cerebrum.core.search import SearchHit, SearchResult, SearchStatus


class SimpleResolver:
    """
    A minimal resolver that applies a fixed similarity threshold.

    This implementation filters out any SearchHit whose score is below
    `min_allowed_score`, and classifies the result as:
    """

    def __init__(self, min_allowed_score, relative_cutoff_ratio):
        """
        Initialize the resolver.

        Args:
            min_allowed_score: The minimum similarity score a SearchHit must satisfy
                               in order to be included in the final result.
                        relative_cutoff_ratio: Ratio of the top hit’s score used as a relative threshold.
                                   Example: 0.5 means “only keep hits scoring at least half of the
                                   best score.” Must be in (0, 1].
        """
        self._min_allowed_score = min_allowed_score
        self._relative_cutoff_ratio = relative_cutoff_ratio

    def resolve(self, search_hits: list[SearchHit]) -> SearchResult:
        """
        Filter and classify raw search hits.

        Args:
            search_hits: The ordered list of SearchHit objects returned
                         directly from the semantic store.

        Returns:
            SearchResult: The structured outcome of the search.
        """
        if not search_hits:
            return SearchResult(
                status=SearchStatus.NO_MATCHES,
                hits=[],
            )
        max_score = self._find_max_score(search_hits)
        filtered_search_hits = []
        for search_hit in search_hits:
            score = search_hit.score
            if (
                score < self._min_allowed_score
                or score < max_score * self._relative_cutoff_ratio
            ):
                continue
            filtered_search_hits.append(search_hit)

        search_status = (
            SearchStatus.NO_MATCHES
            if len(filtered_search_hits) == 0
            else SearchStatus.OK
        )
        return SearchResult(
            status=search_status,
            hits=filtered_search_hits,
        )

    def _find_max_score(self, search_hits: list[SearchHit]) -> float:
        """
        Compute the maximum similarity score in the hit list.

        Assumes the list is non-empty.

        Args:
            search_hits: List of SearchHit objects.

        Returns:
            float: The highest score observed.
        """
        max_score = search_hits[0].score
        for i in range(1, len(search_hits)):
            search_hit = search_hits[i]
            max_score = max(max_score, search_hit.score)
        return max_score
