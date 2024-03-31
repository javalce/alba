from config.config import Config
from src.chatbot.templates.template_manager import TemplateManager
from src.chatbot.memory.long_term_memory import LongTermMemory
from src.chatbot.memory.short_term_memory import ShortTermMemory
from src.chatbot.response_engine import ResponseEngine
from src.chatbot.document_engine import DocumentEngine


class Chatbot:
    def __init__(self, model_name, initial_files=None):
        self.doc_engine = DocumentEngine()
        self.long_term_mem = LongTermMemory()
        self.short_term_mem = ShortTermMemory()
        self.resp_engine = ResponseEngine(model_name)
        if initial_files:
            self.memorize_info(initial_files)

    def recall_messages(self):
        return self.short_term_mem.recall_messages()

    def forget_messages(self):
        self.short_term_mem.forget_messages()

    # Learn new facts from a document and store them in long-term memory
    def memorize_info(self, files, type="decrees"):

        documents = self.doc_engine.generate_documents(files, type)
        self.long_term_mem.add_documents(documents)

    def forget_info(self, collections):
        self.long_term_mem.delete_documents(collections)

    # Respond to user prompt
    def respond(self, user_prompt):
        self.short_term_mem.add_message({"role": "user", "content": user_prompt})

        # Create a self-contained query from recent chat history and the user's prompt
        recent_messages = self.short_term_mem.recall_messages(limit=5, to_str=True)
        query = self._create_query(user_prompt, recent_messages)

        # Enrich the query with relevant facts from long-term memory
        context = self.long_term_mem.get_context(query)
        llm_prompt = TemplateManager.get("llm_prompt", query=query, context=context)

        response = self.resp_engine.generate_response(llm_prompt)
        self.short_term_mem.add_message({"role": "assistant", "content": response})

        return response

    def _create_query(self, prompt, recent_messages):
        # TODO: Implement this method
        # Combine the user prompt with recent chat history to create a self-contained query
        return prompt
