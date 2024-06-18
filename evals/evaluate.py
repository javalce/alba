import pandas as pd
from config.config import Config
from src.templates.template_manager import TemplateManager
from src.memory.long_term_memory import LongTermMemory
from src.memory.short_term_memory import ShortTermMemory
from src.response_engine import ResponseEngine
from src.chatbot import Chatbot

# ---------------------------------------------------------------------
# Load the Excel file with questions and contexts for evaluation
# and generate/evaluate responses using the chatbot
# ---------------------------------------------------------------------


# Load the Excel file
df = pd.read_excel("evaluation_triplets.xlsx")

# Initialize the chatbot
chatbot = Chatbot()

# Iterate through each question in the DataFrame
for index, row in df.iterrows():
    question = row["Question"]  # Adjust the column name if necessary
    context = row["Context"]  # Adjust the column name if necessary

    # Get the response and context from the chatbot
    response_n_context = chatbot.respond_w_context(question)

    # Parse the response and context
    response, context = response_n_context.split("\n\nCONTEXT: ")
    response = response.replace("RESPONSE: ", "")

    # Append the response and context to the DataFrame
    df.at[index, "Answer"] = response
    df.at[index, "Chatbot Context"] = context

# Save the updated DataFrame to a new Excel file
df.to_excel("updated_responses.xlsx", index=False)
