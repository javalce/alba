import os
import csv
import json
from tqdm import tqdm
import fitz  # PyMuPDF for handling PDF files
import pandas as pd
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

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=60),
    )
    def generate_context_question_answer(self, page_text):
        json_schema_str = json.dumps(JSON_SCHEMAS["context_question_answer"])
        system_message = (
            "You are a helpful assistant designed to read a text and create a json structure containing 1. a self-contained question in Spanish that can be answered by the text, 2. the answer in Spanish and 3. the piece of context from the text that is relevant to answer the question. The output should be in json format and match the following schema: "
            + json_schema_str
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": page_text},
        ]
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )
        c_q_a_triplet = response.choices[0].message.content
        return c_q_a_triplet

    def get_responses(self, c_q_a_triplets):
        responses = []
        for triplet in tqdm(c_q_a_triplets, desc="Getting Chatbot Responses"):
            # Ensure the triplet is a dictionary
            if isinstance(triplet, str):
                triplet = json.loads(triplet)  # Convert string to JSON (dictionary)

            question = triplet.get("question")  # Now we can safely get the "question"
            response = self.chatbot.respond_w_context(
                question
            )  # Process the question through the chatbot
            responses.append(response)
        return responses

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=60),
    )
    def evaluate_responses(self, c_q_a_triplets, chatbot_responses):
        results = []
        for triplet, chatbot_response in tqdm(
            zip(c_q_a_triplets, chatbot_responses), desc="Evaluating Responses"
        ):
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
                max_tokens=1024,
                temperature=0,
            )
            result = eval_response.choices[0].message.content
            results.append(result)
        return results

    def save_scores(
        self,
        response_scores,
        c_q_a_triplets,
        chatbot_responses,
        file_path="evaluation_summary.xlsx",
    ):
        # Ensure the evaluation results are dictionaries if they are in string format
        if all(isinstance(result, str) for result in response_scores):
            response_scores = [json.loads(result) for result in response_scores]

        # Prepare data for DataFrame
        data = []
        for triplet, response, eval_result in zip(
            c_q_a_triplets, chatbot_responses, response_scores
        ):
            if isinstance(triplet, str):
                triplet = json.loads(triplet)
            if isinstance(eval_result, str):
                eval_result = json.loads(eval_result)

            data.append(
                [
                    triplet["context"],
                    triplet["question"],
                    triplet["answer"],
                    response,
                    eval_result["has_context"],
                    eval_result["is_correct"],
                ]
            )

        # Create a DataFrame
        df = pd.DataFrame(
            data,
            columns=[
                "GPT Context",
                "GPT Question",
                "GPT Answer",
                "Chatbot Response",
                "Has Context",
                "Is Correct",
            ],
        )

        # Calculate and append summary
        total_responses = len(response_scores)
        total_has_context = sum(row[4] for row in data)
        total_correct = sum(row[5] == 1 for row in data)
        percent_has_context = (
            (total_has_context / total_responses * 100) if total_responses else 0
        )
        percent_correct = (
            (total_correct / total_has_context * 100) if total_has_context else 0
        )

        summary_df = pd.DataFrame(
            [
                [
                    total_responses,
                    total_has_context,
                    f"{percent_has_context:.2f}%",
                    total_correct,
                    f"{percent_correct:.2f}%",
                ]
            ],
            columns=[
                "Total Responses",
                "Total Has Context",
                "Percentage Has Context",
                "Total Correct",
                "Percentage Correct",
            ],
        )

        # Write DataFrames to Excel
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Detailed Responses", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

    def save_triplets(self, c_q_a_triplets, file_path="triplets.xlsx"):
        # First, we need to unpack the JSON into separate columns
        # Initialize lists to hold the unpacked data
        contexts = []
        questions = []
        answers = []

        # Unpack each triplet
        for triplet in c_q_a_triplets:
            # Convert string to JSON (dictionary) if it's not already
            if isinstance(triplet, str):
                triplet = json.loads(triplet)

            # Append data to lists
            contexts.append(triplet.get("context", ""))
            questions.append(triplet.get("question", ""))
            answers.append(triplet.get("answer", ""))

        # Create a DataFrame using the unpacked data
        df = pd.DataFrame(
            {"Context": contexts, "Question": questions, "Answer": answers}
        )

        # Save DataFrame to an Excel file
        df.to_excel(file_path, index=False)

    def run_evaluation(self):
        pages = self.read_pdf()
        c_q_a_triplets = []
        for page in tqdm(pages, desc="Generating C-Q-A Triplets"):
            if page.strip():
                triplet = self.generate_context_question_answer(page)
                c_q_a_triplets.append(triplet)

        # Save the triplets to an Excel file
        self.save_triplets(c_q_a_triplets, "triplets.xlsx")

        # Now read the triplets from the Excel file to proceed
        c_q_a_triplets = []
        df_triplets = pd.read_excel("triplets.xlsx")
        for _, row in df_triplets.iterrows():
            c_q_a_triplets.append(
                {
                    "context": row["Context"],
                    "question": row["Question"],
                    "answer": row["Answer"],
                }
            )

        # Get responses from the chatbot
        chatbot_responses = self.get_responses(c_q_a_triplets)

        # Evaluate responses
        response_scores = self.evaluate_responses(c_q_a_triplets, chatbot_responses)

        # Save the final scores
        self.save_scores(response_scores, c_q_a_triplets, chatbot_responses)

        return response_scores
