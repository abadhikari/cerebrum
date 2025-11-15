from dataclasses import dataclass


@dataclass
class Thought:
  body: str
  tags: list[str]
