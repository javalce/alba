import asyncio
import json
from evaluator import Evaluator


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


# Execute the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())
