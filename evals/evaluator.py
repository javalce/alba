import os
import csv
import json
import fitz  # PyMuPDF for handling PDF files
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# JSON Schemas for expected responses
JSON_SCHEMAS = {
    "context_question_answer": {
        "type": "object",
        "properties": {
            "context": {"type": "string"},
            "question": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["context", "question", "answer"],
    },
    "evaluation_response": {
        "type": "object",
        "properties": {
            "has_context": {"type": "integer", "enum": [0, 1]},
            "is_correct": {"type": "integer", "enum": [-1, 0, 1]},
        },
        "required": ["has_context", "is_correct"],
    },
}


class Evaluator:
    def __init__(self, pdf_path, num_pages=None, chatbot=None):
        self.pdf_path = pdf_path
        self.num_pages = num_pages
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chatbot = chatbot

    def read_pdf(self):
        pages = []
        try:
            with fitz.open(self.pdf_path) as doc:
                for page in doc:
                    if self.num_pages and len(pages) >= self.num_pages:
                        break
                    pages.append(page.get_text())
        except Exception as e:
            print(f"Failed to read PDF: {e}")
        return pages

    def generate_context_question_answer(self, page_text):
        json_schema_str = json.dumps(JSON_SCHEMAS["context_question_answer"])
        system_message = (
            "You are a helpful assistant designed to create read a text and create triplets 1. a self-contained question that can be answered by the text, 2. the answer and 3. the piece of context from the text that is relevant to answer the question. Write the output in Spanish and in json format according to the following schema: "
            + json_schema_str
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": page_text},
        ]
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=messages,
            response_format={"type": "json_object"},
        )
        c_q_a_triplet = response.choices[0].message.content
        return c_q_a_triplet

    def get_responses(self, c_q_a_triplets):
        responses = []
        for triplet in c_q_a_triplets:
            # Ensure the triplet is a dictionary
            if isinstance(triplet, str):
                triplet = json.loads(triplet)  # Convert string to JSON (dictionary)

            question = triplet.get("question")  # Now we can safely get the "question"
            response = self.chatbot.respond_w_context(
                question
            )  # Process the question through the chatbot
            responses.append(response)
        return responses

    def evaluate_responses(self, c_q_a_triplets, chatbot_responses):
        results = []
        for triplet, chatbot_response in zip(c_q_a_triplets, chatbot_responses):
            # Ensure the triplet is a dictionary
            if isinstance(triplet, str):
                triplet = json.loads(triplet)
            gpt_context = triplet["context"]
            gpt_question = triplet["question"]  # Extract the question
            gpt_answer = triplet["answer"]

            json_schema_str = json.dumps(JSON_SCHEMAS["evaluation_response"])
            system_message = (
                "Assess the chatbot's response based on the GPT model's output. Score 'has_context' as 1 if the chatbot's context "
                "accurately reflects the same information necessary to understand the question, otherwise score as 0. "
                "Score 'is_correct' as 1 if the chatbot's context is correct and the answer is also correct; score -1 if the "
                "context is correct but the answer is incorrect; score 0 if the context does not adequately support the question. "
                "Please evaluate according to this JSON schema: " + json_schema_str
            )
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Question: {gpt_question}, GPT Context: {gpt_context}, GPT Answer: {gpt_answer}, Chatbot Response: {chatbot_response}",
                },
            ]
            eval_response = self.client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",
                messages=messages,
                response_format={"type": "json_object"},
            )
            result = eval_response.choices[0].message.content
            results.append(result)
        return results

    def save_scores(
        self,
        response_scores,
        c_q_a_triplets,
        chatbot_responses,
        file_path="evaluation_summary.csv",
    ):
        total_responses = len(response_scores)

        # Ensure the evaluation results are dictionaries if they are in string format
        if all(isinstance(result, str) for result in response_scores):
            response_scores = [json.loads(result) for result in response_scores]

        total_has_context = sum(
            1 for result in response_scores if result["has_context"] == 1
        )
        total_correct = sum(
            1 for result in response_scores if result["is_correct"] == 1
        )

        # Calculate percentages
        percent_has_context = (
            (total_has_context / total_responses * 100) if total_responses else 0
        )
        percent_correct = (
            (total_correct / total_has_context * 100) if total_has_context else 0
        )

        # Create or overwrite the CSV file
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)

            # Write summary row
            writer.writerow(
                [
                    "Total Responses",
                    "Total Has Context",
                    "Percentage Has Context",
                    "Total Correct",
                    "Percentage Correct",
                ]
            )
            writer.writerow(
                [
                    total_responses,
                    total_has_context,
                    f"{percent_has_context:.2f}%",
                    total_correct,
                    f"{percent_correct:.2f}%",
                ]
            )

            # Write headers for detailed table
            writer.writerow(
                [
                    "GPT Context",
                    "GPT Question",
                    "GPT Answer",
                    "Chatbot Context",
                    "Chatbot Answer",
                    "Has Context",
                    "Is Correct",
                ]
            )

            # Write each response detail row
            for triplet, response, eval_result in zip(
                c_q_a_triplets, chatbot_responses, response_scores
            ):
                if isinstance(triplet, str):
                    triplet = json.loads(triplet)
                if isinstance(eval_result, str):
                    eval_result = json.loads(eval_result)

                gpt_context = triplet["context"]
                gpt_question = triplet["question"]
                gpt_answer = triplet["answer"]
                chatbot_context = (
                    response["context"] if "context" in response else "N/A"
                )  # Adjust as per chatbot response structure
                chatbot_answer = (
                    response["answer"] if "answer" in response else "N/A"
                )  # Adjust as per chatbot response structure

                writer.writerow(
                    [
                        gpt_context,
                        gpt_question,
                        gpt_answer,
                        chatbot_context,
                        chatbot_answer,
                        eval_result["has_context"],
                        eval_result["is_correct"],
                    ]
                )

    def run_evaluation(self):
        pages = self.read_pdf()
        c_q_a_triplets = [
            self.generate_context_question_answer(page)
            for page in pages
            if page.strip()
        ]
        chatbot_responses = self.get_responses(c_q_a_triplets)
        response_scores = self.evaluate_responses(c_q_a_triplets, chatbot_responses)
        self.save_scores(response_scores, c_q_a_triplets, chatbot_responses)
        return response_scores
