import os
import json
from evaluator import Evaluator
from dotenv import load_dotenv


def main():
    load_dotenv()  # Load environment variables from .env file
    openai_api_key = os.getenv(
        "OPENAI_API_KEY"
    )  # Get OpenAI API key from environment variables

    # Paths and endpoints
    pdf_path = "./data/processed/decretos_N_5060_2023.pdf"
    gpt4_endpoint = "https://api.openai.com/v1/chat/completions"

    # Create an Evaluator instance with the necessary headers
    evaluator = Evaluator(
        pdf_path,
        gpt4_endpoint,
        num_pages=2,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",  # Ensure headers include the Authorization token
        },
    )

    # Run the evaluation and handle potential errors
    try:
        results = evaluator.run_evaluation()
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
