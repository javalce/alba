class ShortTermMemory:
    def __init__(self):
        self.messages = []  # A list to store chat messages

    def add_message(self, message):
        """
        Adds a message to the conversation history.

        Parameters:
        - message (dict): A message object containing role and content keys.
        """
        self.messages.append(message)

    def recall_messages(self, limit=None, to_str=False):
        """
        Retrieves the conversation history.

        Parameters:
        - limit (int): Optional parameter to specify the number of recent messages to retrieve.

        Returns:
        - list: A list of message objects.
        """
        # if to_str is True, return a string of the most recent messages (both role and content)
        if to_str:
            recent_messages = self.recall_messages(limit=limit)
            return "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_messages]
            )

        # Otherwise, return a list of message objects
        return self.messages[-limit:] if limit else self.messages

    def forget_messages(self):
        """
        Clears the conversation history.
        """
        self.messages.clear()
