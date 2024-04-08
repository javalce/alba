import os
import streamlit as st
from streamlit import session_state as ss
from config.config import Config
from src.chatbot import Chatbot


def main():
    # Configure page
    st.set_page_config(
        page_title="Alba - Asistente de Búsqueda Local y Privado",
        page_icon="👩",
        layout="wide",
    )
    st.title("Alba 👩 (Asistente de Búsqueda Local y Privado)")
    # Initialize chatbot if necessary
    if "chatbot" not in ss:
        model = Config.get("inference_model")
        ss.chatbot = Chatbot(model)
        st.write("Chatbot initialized!")  # Debug message

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
    main()
