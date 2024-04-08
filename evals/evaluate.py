import os
from dotenv import load_dotenv
from evaluator import Evaluator
from config.config import Config
import pandas as pd


def main():
    load_dotenv()
    evaluator = Evaluator()

    # Load or generate the testset
    testset_path = Config.get("testset_path")
    if os.path.exists(testset_path):
        testset = evaluator.load_testset(testset_path)
    else:
        testset_sources_path = Config.get("testset_sources_path")
        testset = evaluator.generate_testset(testset_sources_path)
        evaluator.save_testset(testset, testset_path)

    # Evaluate the testset
    evals = evaluator.evaluate_testset(testset)

    # Convert evals to pandas DataFrame and print
    df = evals.to_pandas()
    print(df)


if __name__ == "__main__":
    main()
