import ollama

from .language_model import ChatMessage

MESSAGE_KEY = "message"
CONTENT_KEY = "content"
TEMPERATURE_KEY = "temperature"


class OllamaModel:
  """
  Concrete implementation of the LanguageModel interface using the local Ollama API.

  This class provides a lightweight wrapper around the Ollama Python SDK, allowing
  the model to generate responses from a list of chat messages. It supports
  configurable temperature settings for sampling variability.
  """

  def __init__(self, model: str, temperature: float):
    """
    Initialize an Ollama language model.

    Args:
        model (str): The model identifier (e.g., 'llama3:8b').
        temperature (float): Sampling temperature controlling randomness.
            Higher values (e.g., 1.0) produce more diverse responses, while
            lower values (e.g., 0.2) make output more deterministic.
    """
    self._model = model
    self._temperature = temperature
  
  def call(self, messages: list[ChatMessage]) -> str:
    """
    Generate a response using the Ollama model.

    Args:
        messages (list[ChatMessage]): A list of chat messages forming the
            conversation history. Each message must contain a role and content.

    Returns:
        str: The generated text response from the model.

    Raises:
        KeyError: If the Ollama API response structure changes or expected
            keys ('message', 'content') are missing.
    """
    response = ollama.chat(
      model=self._model,
      messages=messages,
      options={TEMPERATURE_KEY: self._temperature}
    )
    return response[MESSAGE_KEY][CONTENT_KEY]
