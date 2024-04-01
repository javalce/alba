import os
import streamlit as st
from streamlit import session_state as ss
from config.config import Config
from src.chatbot import Chatbot


# TODO: Empty database, load more data, connect backend-frontend
def main():
    # Configure your page with a logo
    st.set_page_config(
        page_title="Búsqueda Privada", page_icon=":shark:", layout="wide"
    )
    # Initialize chatbot if necessary
    if "chatbot" not in ss:
        # Chatbot initialization
        # All files in the raw folder will be loaded during initialization
        initial_files = []
        folder = Config.get("raw_data_folder")
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                initial_files.append(file_path)

        model = Config.get("inference_model")
        ss.chatbot = Chatbot(model, initial_files)

    # Display chat messages from history on app rerun
    messages = ss.chatbot.recall_messages()
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Escribe tu mensaje..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response and display it in chat
        response = ss.chatbot.respond(prompt)
        with st.chat_message("assistant"):
            st.write(response)


if __name__ == "__main__":
    # TODO: Move this back into main function and empty database
    # as part of the initialization in DEV mode
    # TODO: Add all embeddings/documents to the database

    model = Config.get("inference_model")
    chatbot = Chatbot(model)

    response = chatbot.respond(
        "¿Qué se les concedió a a Doña Eva Esperanza Iniesta Blázquez y Dª Eloisa Martínez Rubio?"
    )

    print(response)
