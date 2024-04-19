import os
import json
import fitz  # PyMuPDF for handling PDF files
import jsonschema  # Include jsonschema for validation
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
}


class Evaluator:
    def __init__(self, pdf_path, num_pages=None):
        self.pdf_path = pdf_path
        self.num_pages = num_pages
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            "You are a helpful assistant designed to output structured json according to the following schema: "
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
        return response.choices[0].message.content

    def run_evaluation(self):
        pages = self.read_pdf()
        results = []
        for page in pages:
            if page.strip():
                response = self.generate_context_question_answer(page)
                results.append(response)
        return results


def main():
    pdf_path = "./data/processed/decretos_N_5060_2023.pdf"
    evaluator = Evaluator(pdf_path, num_pages=2)
    try:
        results = evaluator.run_evaluation()
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
