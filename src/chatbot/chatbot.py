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
        self.memorize_files(initial_files)

    def recall_messages(self):
        return self.short_term_mem.recall_messages()

    # Forget messages (i.e., clear short-term memory)
    def forget_messages(self):
        self.short_term_mem.forget_messages()

    # Learn new facts from a document and store them in long-term memory
    def memorize_files(self, files, type="decrees"):

        # Read files and generate documents
        documents = self.doc_engine.generate_documents(files, type)

        # Store documents in long-term memory
        # See LongTermMemory class for storing details
        self.long_term_mem.add_documents(documents)

    # Respond to user prompt
    def respond(self, user_prompt):
        # Store user message in short-term memory
        self.short_term_mem.add_message({"role": "user", "content": user_prompt})

        # Create a self-contained query from recent chat history and the user's prompt
        recent_messages = self.short_term_mem.recall_messages(limit=5, to_str=True)
        query = self._create_query(user_prompt, recent_messages)

        # Enrich the query with relevant facts from long-term memory
        context = self.long_term_mem.recall_docs(query)
        llm_prompt = TemplateManager.get("llm_prompt", query=query, context=context)

        # Use the responder to generate a response based on the enriched query
        response = self.resp_engine.generate_response(llm_prompt)

        # Store the chatbot's response in short-term memory
        self.short_term_mem.add_message({"role": "assistant", "content": response})

        return response

    def _create_query(self, prompt, recent_messages):
        # TODO: Implement this method
        return prompt
