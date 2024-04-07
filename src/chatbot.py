from src.templates.template_manager import TemplateManager
from src.memory.long_term_memory import LongTermMemory
from src.memory.short_term_memory import ShortTermMemory
from src.response_engine import ResponseEngine


class Chatbot:
    def __init__(self, model_name):
        self.long_term_mem = LongTermMemory()
        self.short_term_mem = ShortTermMemory()
        self.resp_engine = ResponseEngine(model_name)

    def recall_messages(self):
        return self.short_term_mem.recall_messages()

    def forget_messages(self):
        self.short_term_mem.forget_messages()

    # Learn new facts from a document and store them in long-term memory
    def memorize_info(self, files, type):
        self.long_term_mem.add_documents(files, type)

    def forget_info(self, collections):
        self.long_term_mem.delete_documents(collections)

    # Respond to user prompt
    def respond(self, user_prompt):
        self.short_term_mem.add_message({"role": "user", "content": user_prompt})

        # Create a self-contained query from recent chat history and the user's prompt
        recent_messages = self.short_term_mem.recall_messages(limit=5, to_str=True)
        query = self._create_query(user_prompt, recent_messages)

        # Enrich the query with relevant facts from long-term memory
        context, sources = self.long_term_mem.get_context(query)
        llm_prompt = TemplateManager.get("llm_prompt", query=query, context=context)

        response = self.resp_engine.generate_response(llm_prompt)
        response_n_sources = f"{response}\n\n{sources}"
        self.short_term_mem.add_message(
            {"role": "assistant", "content": response_n_sources}
        )

        return response_n_sources

    def _create_query(self, prompt, recent_messages):
        # This method should combine the prompt with recent chat history to create a self-contained query
        # However, for now, we will simply return the user prompt as is due to latency issues
        return prompt
