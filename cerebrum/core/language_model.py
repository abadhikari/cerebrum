from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class ChatMessage(TypedDict):
    """
    Represents a single message exchanged in a chat conversation.

    Attributes:
        role (str): The role of the message author — typically "user",
            "assistant", or "system".
        content (str): The text content of the message.
    """

    role: str
    content: str


@dataclass(slots=True)
class CallOptions:
    """
    Generic options for a language model call.

    Backends are free to ignore unsupported fields.
    """
    format: Any = None


class LanguageModel(Protocol):
    """
    Interface defining the contract for a language model.

    All concrete implementations must accept a list of chat messages
    and return the generated text response as a string.
    """

    def call(self, messages: list[ChatMessage], options: CallOptions | None = None) -> str:
        """
        Generate a text response given a list of chat messages.

        Args:
            messages (list[ChatMessage]): The conversation history,
                ordered chronologically.
            options (CallOptions | None): Optional model call configuration.

        Returns:
            str: The generated model response.
        """
        ...
