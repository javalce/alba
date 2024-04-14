import fitz  # PyMuPDF for handling PDF files
import aiohttp  # For asynchronous HTTP requests
import asyncio
import json
import os


class Evaluator:
    def __init__(self, pdf_path, gpt4_endpoint, num_pages=None):
        self.pdf_path = pdf_path
        self.gpt4_endpoint = gpt4_endpoint
        self.num_pages = num_pages
        self.headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}

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

    async def generate_context_question_answer(self, session, page_text):
        prompt = {
            "model": "gpt-4-turbo",
            "prompt": f"Generate a detailed response in JSON format containing the context, a question, and an answer based on the following text: {page_text}",
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            "temperature": 0.5,
        }
        async with session.post(
            self.gpt4_endpoint, json=prompt, headers=self.headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Error with GPT-4 API: {await response.text()}")
                return None

    async def evaluate_responses(self, session, data):
        if not data:
            return {"context_had_info": 0, "answer_evaluation": 0}
        verify_prompt = {
            "model": "gpt-4-turbo",
            "prompt": f"Evaluate if the context provided in '{data['context']}' contains the necessary information for the question '{data['question']}' and assess the correctness of the answer.",
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
            "temperature": 0.5,
        }
        async with session.post(
            self.gpt4_endpoint, json=verify_prompt, headers=self.headers
        ) as verification_response:
            if verification_response.status == 200:
                result = await verification_response.json()
                context_had_info = 1 if result.get("context_had_info", False) else 0
                answer_evaluation = (
                    (1 if result.get("answer_correct", False) else -1)
                    if context_had_info
                    else 0
                )
                return {
                    "context_had_info": context_had_info,
                    "answer_evaluation": answer_evaluation,
                }
            else:
                print(
                    f"Error during verification: {await verification_response.text()}"
                )
                return {"context_had_info": 0, "answer_evaluation": 0}

    async def run_evaluation(self):
        pages = self.read_pdf()
        async with aiohttp.ClientSession() as session:
            results = [
                await self.evaluate_responses(
                    session, await self.generate_context_question_answer(session, page)
                )
                for page in pages
                if page.strip()
            ]
            return results


# Asynchronous main function to run the evaluation
async def main():
    pdf_path = "path_to_your_pdf.pdf"
    gpt4_endpoint = (
        "https://api.openai.com/v1/chat/completions"  # Adjust with actual API endpoint
    )
    evaluator = Evaluator(
        pdf_path, gpt4_endpoint, num_pages=10
    )  # Process only the first 10 pages
    results = await evaluator.run_evaluation()
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
