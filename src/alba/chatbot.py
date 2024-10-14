# Local application imports
import logging
import re
from typing import Any, Iterator

from alba.config import Config
from alba.memory.long_term_memory import LongTermMemory
from alba.memory.short_term_memory import ShortTermMemory
from alba.response_engine import ResponseEngine
from alba.templates.template_manager import TemplateManager
from alba.utils.utils import setup_logging

setup_logging()


class Chatbot:
    """
    A class representing a chatbot that uses long-term and short-term memory to respond to user prompts.
    """

    def __init__(self, config: Config, long_term_mem: LongTermMemory):
        """
        Initialize the Chatbot object with long-term memory, short-term memory, and a response engine.
        """

        self.long_term_mem = long_term_mem
        self.short_term_mem = ShortTermMemory()
        model = config.INFERENCE_MODEL
        self.resp_engine = ResponseEngine(model)

    def recall_messages(self):
        """
        Recall messages from short-term memory.

        Returns:
            list: A list of messages recalled from short-term memory.
        """
        return self.short_term_mem.recall_messages()

    def forget_messages(self):
        """
        Forget all messages in short-term memory.
        """
        self.short_term_mem.forget_messages()

    def memorize_info(self, files, type):
        """
        Learn new facts from documents and store them in long-term memory.

        Args:
            files (list): A list of file paths to memorize information from.
            type (str): The type of information being memorized.
        """
        self.long_term_mem.add_documents(files, type)

    def forget_info(self, collections):
        """
        Forget information from specified collections in long-term memory.

        Args:
            collections (str): The collections to forget information from.
        """
        self.long_term_mem.delete_documents(collections)

    def respond_w_sources(self, user_prompt: str) -> Iterator[str]:
        """
        Respond to a user prompt with sources.

        Args:
            user_prompt (str): The user's prompt.

        Returns:
            str: The response with sources.
        """
        self.short_term_mem.add_message({"role": "user", "content": user_prompt})

        logging.info("Processing user prompt...")

        # Create a self-contained query from recent chat history and the user's prompt
        recent_messages = self.short_term_mem.recall_messages(limit=5, to_str=True)
        query = self._create_query(user_prompt, recent_messages)

        logging.info("Retrieving context...")

        # Enrich the query with relevant facts from long-term memory
        context, sources = self.long_term_mem.get_context(query)
        llm_prompt = TemplateManager.get("llm_prompt", query=query, context=context)

        logging.info("Generating response...")

        response_n_sources = ""

        for response in self.resp_engine.generate_response(llm_prompt):
            response_n_sources += response
            yield response

        yield from re.split(r"(\S+)", f"\n\n{sources}")

        response_n_sources += f"\n\n{sources}"

        logging.info(f"Response with sources: {response_n_sources}")

        self.short_term_mem.add_message({"role": "assistant", "content": response_n_sources})

    def respond_w_messages(self, messages: dict[str, Any]) -> Iterator[str]:
        """
        Respond to a user chat history.

        Args:
            messages (dict[str, Any]): The user's chat history.

        Returns:
            str: The response.
        """

        self.short_term_mem.add_message(messages[-1])

        message = ""

        for response in self.resp_engine.generate_response(messages):
            message += response
            yield response

        self.short_term_mem.add_message(message)

    def respond_w_context(self, user_prompt):
        """
        Respond to a user prompt with context.

        Args:
            user_prompt (str): The user's prompt.

        Returns:
            str: The response with context.
        """
        # Enrich the query with relevant facts from long-term memory
        context, sources = self.long_term_mem.get_context(user_prompt)
        llm_prompt = TemplateManager.get("llm_prompt", query=user_prompt, context=context)

        response_n_context = "RESPONSE: "
        for response in self.resp_engine.generate_response(llm_prompt):
            response_n_context += response
        response_n_context += f"\n\nCONTEXT: {context}"

        return response_n_context

    def _create_query(self, prompt, recent_messages):
        """
        Create a self-contained query from the user prompt and recent chat history.

        Args:
            prompt (str): The user's prompt.
            recent_messages (str): The recent chat history as a string.

        Returns:
            str: The self-contained query.
        """
        # This method should combine the prompt with recent chat history to create a self-contained query
        # However, for now, we will simply return the user prompt as is due to latency issues
        # observed with local models currently (expected to be resolved in future updates)
        return prompt
