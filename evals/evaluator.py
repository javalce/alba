import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from ragas import evaluate
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.config import Config


class Evaluator:
    def generate_testset(self, input_file):
        loader = PyPDFLoader(input_file)
        documents = loader.load_and_split()

        for document in documents:
            document.metadata["filename"] = document.metadata["source"]

        # Initialize generator and critic with OpenAI models
        generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        critic_llm = ChatOpenAI(model="gpt-4")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm, critic_llm, embeddings
        )

        # Generate the testset
        testset = generator.generate_with_langchain_docs(
            documents,
            test_size=10,
            distributions={"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25},
        )

        return testset

    def save_testset(self, testset, testset_path):
        with open(testset_path, "w") as file:
            json.dump(testset, file)

    def load_testset(self, testset_path):
        with open(testset_path, "r") as file:
            return json.load(file)

    def evaluate_testset(self, testset):
        evals = evaluate(
            testset["eval"],
            metrics=[
                "context_precision",
                "faithfulness",
                "answer_relevancy",
                "context_recall",
            ],
        )

        return evals
