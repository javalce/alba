import json
from src.chatbot import Chatbot
from evals.evaluator import Evaluator


def main():
    pdf_path = "data/raw/Ex Machina.pdf"
    chatbot = Chatbot()
    evaluator = Evaluator(pdf_path, num_pages=1, chatbot=chatbot)
    results = evaluator.run_evaluation()
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
