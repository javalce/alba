
# Private Offline LLM-Powered RAG Chatbot

## Introduction

This repository contains the implementation of a fully private, fully offline, Large Language Model (LLM) powered Retrieval-Augmented Generation (RAG) Chatbot. This chatbot is designed to leverage the powerful capabilities of LLMs while ensuring complete privacy and offline functionality.

## Features

- **Fully Offline**: Operates independently of cloud services, ensuring that all interactions remain private and locally processed.
- **LLM-Powered**: Utilizes a state-of-the-art Large Language Model to understand and generate human-like responses.
- **Retrieval-Augmented Generation (RAG)**: Combines the benefits of retrieval-based and generative approaches for nuanced and context-aware conversations.
- **Privacy-First**: Built with privacy as a core principle, ensuring that all data stays on your device.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pablovazquezg/alba.git
   ```
2. Navigate to the cloned directory:
   ```bash
   cd alba
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```

## Usage

Launch the chatbot by running:

```bash
streamlit run main.py
```

Interact with the chatbot through the streamlit GUI.

## Configuration

To configure the chatbot settings, refer to the `config/config.json` file where you can adjust various parameters such as model settings, privacy options, etc.


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

