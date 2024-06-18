import os
import json
from src.chatbot import Chatbot
from evals.evaluator import Evaluator

# ------------------------------------------------------------
# Main function to create the Question/Context/Answer triplets
# that will be used for evaluation.
# ------------------------------------------------------------


def main():
    # Path to the PDF file
    pdf_path = ""

    # Path to the triplets Excel file
    triplets_file_path = "evaluation_triplets.xlsx"

    # Check if the triplets file exists
    use_existing_triplets = os.path.exists(triplets_file_path)

    # Initialize the Chatbot
    chatbot = Chatbot()

    # Initialize the Evaluator with the PDF path, number of pages to read, and chatbot instance
    evaluator = Evaluator(pdf_path, num_pages=5, chatbot=chatbot)

    if use_existing_triplets:
        print("Using existing triplets file for evaluation.")
    else:
        print("Generating new triplets from PDF for evaluation.")

    # Run the evaluation and get the results
    results = evaluator.run_evaluation(
        use_existing_triplets=use_existing_triplets,
        triplets_file_path=triplets_file_path,
    )

    # Print the results as a formatted JSON string
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
