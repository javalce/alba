import json
from src.chatbot import Chatbot
from evals.evaluator import Evaluator


def main():
    # Path to the PDF file
    pdf_path = "data/raw/decretos_N_5060_2023.pdf"

    # Initialize the Chatbot
    chatbot = Chatbot()

    # Initialize the Evaluator with the PDF path, number of pages to read, and chatbot instance
    evaluator = Evaluator(pdf_path, num_pages=25, chatbot=chatbot)

    # Run the evaluation and get the results
    results = evaluator.run_evaluation()

    # Print the results as a formatted JSON string
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
