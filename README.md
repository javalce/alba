# Private Offline LLM-Powered RAG Chatbot

## Introduction

This repository contains the implementation of a fully private, fully offline, Large Language Model (LLM) powered Retrieval-Augmented Generation (RAG) Chatbot. This chatbot is designed to leverage the powerful capabilities of LLMs while ensuring complete privacy and offline functionality.

## Features

- **Fully Offline**: Operates independently of cloud services, ensuring that all interactions remain private and locally processed.
- **LLM-Powered**: Utilizes a state-of-the-art Large Language Model to understand and generate human-like responses.
- **Retrieval-Augmented Generation (RAG)**: Combines the benefits of retrieval-based and generative approaches for nuanced and context-aware conversations.
- **Privacy-First**: Built with privacy as a core principle, ensuring that all data stays on your device.

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- [Docker](https://www.docker.com/)

## Installation

> [!IMPORTANT]
> Make sure you are using the branch `sedipualba-chat`.

```bash
poetry install
```

### Post-Installation

After installing the dependencies, you will need to download the pre-trained model weights. To do this, run the following command:

```bash
python -m spacy download es_core_news_lg
```

This will download the pre-trained Spanish language model for spaCy.

```bash
python -m nltk.downloader punkt
```

This will download the punkt tokenizer for NLTK.

> [!IMPORTANT]
> You will need to have a milvus database running and a ollama server with the model loaded in order to use the chatbot.
>
> The ollama model is specified in the `config/config.py` file under the `DEFAULT_INFERENCE_MODEL` key. It defaults to `llama3.2`.

### Environment Variables

You can create a `.env` file in the root directory of the project to customize the behavior of the chatbot. The following environment variables are required:

- `JWT_SECRET`: The secret key used to sign the JWT tokens. Use `openssl rand -hex 32` to generate a random key.

You can check the `src/alba/config.py` file to see the available environment variables.

## Usage

Launch the chatbot by running:

```bash
# Activate the virtual environment
poetry shell

# Run the chatbot
alba run
```

Interact with the chatbot through the API endpoints. The chatbot docs will be available at `http://localhost:8000/docs` or `http://localhost:8000/redoc` for the Swagger or ReDoc documentation, respectively.

## Deployment

To deploy the chatbot, you must use the provided `docker-compose.yaml` file. This file contains the necessary configuration to run the chatbot in a production environment.

### Prerequisites

Create a `.env.production` file in the root directory of the project with the following environment variables:

- `JWT_SECRET`: The secret key used to sign the JWT tokens. Use `openssl rand -hex 32` to generate a random key.
- `MODEL_API_URL`: The URI of the Ollama server. If using docker compose, use `http://ollama:11434/api/generate`.
- `MILVUS_URI`: The URI of the Milvus database. If using docker compose, use `http://standalone:19530`.

### Pre-Deployment

Before deploying the chatbot, you must create the persistent volumes for the Milvus, Ollama and Chatbot services. To do this, run the following command:

```bash
mkdir -p volumes.prod/{etcd,minio,milvus,ollama,logs}
touch volumes.prod/{db.sqlite,logs/log.log}
```

After creating the volumes, you have to build the Docker images. To do this, run the following command:

```bash
docker-compose -f docker/docker-compose.yaml build
```

### Deployment

To deploy the chatbot, run the following command:

```bash
docker-compose -f docker/docker-compose.yaml up -d
```

To access the chatbot container, run the following command:

```bash
docker compose -f docker/docker-compose.yaml exec alba_api bash
```

### Post-Deployment

After deploying the chatbot, you will need to execute the ollama model. To do this, run the following command:

```bash
docker compose -f docker/docker-compose.yaml exec -d ollama ollama run llama3.2
```

> [!NOTE]
> You will have to wait a few minutes for the ollama model to download the necessary files and be ready to generate responses.

## Makefile

The project includes a `Makefile` with several useful commands to simplify the development process. To see the available commands, run:

```bash
make help # or just make
```

The commands for the deployment process are:

- `make prepare`: Prepares the environment for deployment.
- `make build`: Builds the Docker images.
- `make up`: Deploys the chatbot.
- `make ollama`: Executes the ollama model.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
