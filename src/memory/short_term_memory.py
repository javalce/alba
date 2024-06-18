from typing import Union


class ShortTermMemory:
    """
    A class representing short-term memory for storing chat messages.
    """

    def __init__(self):
        """
        Initialize the ShortTermMemory object with an empty list of messages.
        """
        self.messages = []

    def add_message(self, message: dict) -> None:
        """
        Add a message to the conversation history.

        Args:
            message (dict): A message object containing 'role' and 'content' keys.
        """
        self.messages.append(message)

    def recall_messages(
        self, limit: int = None, to_str: bool = False
    ) -> Union[list, str]:
        """
        Retrieve the conversation history.

        Args:
            limit (int, optional): The number of recent messages to retrieve. If None, retrieve all messages.
            to_str (bool, optional): If True, return the messages as a formatted string. If False, return a list of message objects.

        Returns:
            Union[list, str]: A list of message objects or a formatted string of the most recent messages.
        """
        if to_str:
            recent_messages = self.recall_messages(limit=limit)
            return "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_messages]
            )
        return self.messages[-limit:] if limit else self.messages

    def forget_messages(self) -> None:
        """
        Clear the conversation history.
        """
        self.messages.clear()
