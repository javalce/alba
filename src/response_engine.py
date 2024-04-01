from typing import Any
from ollama import chat
from config.config import Config
from src.templates.template_manager import TemplateManager


class ResponseEngine:
    def __init__(self, model_name: str) -> None:
        """
        Initializes the response engine with a specific model type.

        Parameters:
        - model_name (str): Identifier for the type of model or response generation strategy to use.
        """
        self._model = model_name

    def _load_model(self, model_name: str) -> Any:
        """
        Loads a model based on the specified model type.

        Parameters:
        - model_name (str): The type of model to load.
        """
        self._model = model_name

    def generate_response(self, llm_prompt: str) -> str:
        """
        Generates a response based on the enriched query (llm_prompt) using the loaded model or strategy.

        Parameters:
        - llm_prompt (str): The enriched query that combines the user prompt with context from recent messages and long-term memory.

        Returns:
        - str: The generated response.
        """

        messages = [
            {
                "role": "system",
                "content": TemplateManager.get("system_message", model=self._model),
            },
            {
                "role": "user",
                "content": llm_prompt,
            },
        ]

        # Send the messages to the chat model
        response = chat(
            Config.get("default_inference_model"), messages=messages, stream=False
        )

        return response["message"]["content"]

    def _create_query(self, user_prompt: str, recent_messages: str) -> str:
        """
        Combines the user prompt with recent chat history to create a self-contained query.

        Parameters:
        - user_prompt (str): The latest user prompt.
        - recent_messages (str): Concatenated string of recent messages for context.

        Returns:
        - str: A self-contained query that combines the prompt with context.
        """
        # TODO: Implement this method
        return user_prompt
