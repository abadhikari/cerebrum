"""
CLI view helpers for Cerebrum.

This module defines small, stateless functions used by the interactive
CLI to display banners, menus, index lists, and query results. These
functions contain no business logic — they only handle text formatting
and printing to the console.
"""

from cerebrum.core.repository import Index, ThoughtRecord
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
    _
.__(.)< (MEOW)
 \___)
"""


def print_banner() -> None:
    print(CEREBRUM_ASCII)


def print_duck() -> None:
    print(DUCK_ASCII)


def print_indexes(indexes: list[Index]) -> dict[str, Index]:
    indexes_map = {}
    print()
    print("=== Indexes List ===")
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
    print()
    print("=== MENU ===")
    for key, (label, _) in menu_actions.items():
        print(f"[{key}] {label}")


def print_search_result(search_result: SearchResult) -> None:
    print()
    print("===== Results =====")

    hits = search_result.hits
    if not hits:
        print("(no results)")
        return

    for hit in hits:
        tags = ", ".join(hit.record.tags)

        print()
        print(f"thought: {hit.record.body}")
        print(f"tags: {tags}")
        print(f"score: {hit.score:.3f}")


def print_thought_records(thought_records: list[ThoughtRecord]) -> None:
    print()
    print("===== Thoughts =====")

    if not thought_records:
        print("(no records)")
        return

    for record in thought_records:
        print()
        print(f"thought: {record.body}")
        print(f"tags: {record.tags}")


def print_section(section_name: str, body: str) -> None:
    header = f"---- {section_name} ----"
    bar = len(header) * "-"

    print()
    print(header)
    print(body)
    print(bar)
