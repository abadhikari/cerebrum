from typing import Protocol, TypedDict

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


class LanguageModel(Protocol):
  """
  Interface defining the contract for a language model.

  All concrete implementations must accept a list of chat messages
  and return the generated text response as a string.
  """

  def call(self, messages: list[ChatMessage]) -> str:
    """
    Generate a text response given a list of chat messages.

    Args:
        messages (list[ChatMessage]): The conversation history,
            ordered chronologically.

    Returns:
        str: The generated model response.
    """
    ...