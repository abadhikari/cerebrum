"""
CLI view helpers for Cerebrum.

This module defines small, stateless functions used by the interactive
CLI to display banners, menus, index lists, and query results. These
functions contain no business logic — they only handle text formatting
and printing to the console.
"""

from cerebrum.core.repository import Index
from cerebrum.core.search import SearchResult

CEREBRUM_ASCII = r"""
_________                     ___.
\_   ___ \  ___________   ____\_ |_________ __ __  _____
/    \  \/_/ __ \_  __ \_/ __ \| __ \_  __ \  |  \/     \
\     \___\  ___/|  | \/\  ___/| \_\ \  | \/  |  /  Y Y  \
 \______  /\___  >__|    \___  >___  /__|  |____/|__|_|  /
        \/     \/            \/    \/                  \/

"""

DUCK_ASCII = r"""
.__(.)< (MEOW)
 \___)   
"""


def print_banner() -> None:
    print(CEREBRUM_ASCII)


def print_duck() -> None:
    print(DUCK_ASCII)


def print_indexes(indexes: list[Index]) -> dict[str, Index]:
    indexes_map = {}
    print("\n=== Indexes List ===\n")
    for i, index in enumerate(indexes):
        print(f"{i + 1}. {index}")
        indexes_map[str(i + 1)] = index
    return indexes_map


def print_box_text(text: str) -> None:
    n = len(text)
    horizontal = f"+{'-' * (n + 2)}+"
    box_text = f"{horizontal}\n| {text} |\n{horizontal}"
    print(box_text)


def print_menu(menu_actions: dict[str, tuple[str, object]]) -> None:
    print("\n=== MENU ===")
    for key, (label, _) in menu_actions.items():
        print(f"[{key}] {label}")


def print_search_result(search_result: SearchResult) -> None:
    print("\n===== Results =====\n")
    for hit in search_result.hits:
        print(
            f"thought: {hit.record.body}\n"
            f"tags: {hit.record.tags}\n"
            f"score: {hit.score:.3f}\n",
        )
