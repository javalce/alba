import time
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from milvus_model.hybrid import BGEM3EmbeddingFunction
from milvus_model.sparse import BM25EmbeddingFunction
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
import torch
import PyPDF2

# ------------------------------------------------------------
# This script is used to compare the performance of different
# embedding methods and configurations
# ------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Create the analyzer for BM25 embeddings
analyzer = build_default_analyzer(language="sp")


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def split_text(text, num_parts):
    words = text.split()
    part_size = len(words) // num_parts
    parts = [words[i : i + part_size] for i in range(0, len(words), part_size)]
    return parts


def create_batches(words, batch_size):
    batches = [
        " ".join(words[i : i + batch_size]) for i in range(0, len(words), batch_size)
    ]
    return batches


def create_embeddings(text):
    # Create dense embeddings using BGE-M3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_fp16 = True if device.startswith("cuda") else False
    bgeM3_ef = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3",
        device=device,
        use_fp16=use_fp16,
    )
    dense_embeddings = bgeM3_ef.encode_documents([text])["dense"]

    # Create sparse embeddings using BM25
    bm25_ef = BM25EmbeddingFunction(analyzer)
    raw_sparse_embeddings = bm25_ef.encode_documents([text])
    sparse_embeddings = [
        sparse_embedding.toarray().tolist()[0]
        for sparse_embedding in raw_sparse_embeddings
    ]

    return dense_embeddings, sparse_embeddings


def process_batch(batch):
    dense_embeddings, sparse_embeddings = create_embeddings(batch)
    logging.info(f"Processed batch: {batch[:50]}...")
    logging.info(
        f"Dense embeddings shape: {len(dense_embeddings)}, {len(dense_embeddings[0])}"
    )
    logging.info(
        f"Sparse embeddings shape: {len(sparse_embeddings)}, {len(sparse_embeddings[0])}"
    )


def process_batches_parallel(batches, max_workers, batch_size):
    start_time = time.time()
    total_batches = len(batches)
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]

        # Print progress updates
        for future in concurrent.futures.as_completed(futures):
            completed_batches += 1
            progress_percent = (completed_batches / total_batches) * 100
            print(
                f"Progress: {completed_batches}/{total_batches} ({progress_percent:.2f}%)"
            )

            # Ensure to catch any exceptions raised by the future
            try:
                future.result()
            except Exception as e:
                print(f"Error processing batch: {e}")

    end_time = time.time()
    processing_time = end_time - start_time
    print(
        f"Processing with {max_workers} workers and batch size {batch_size} took {processing_time:.2f} seconds."
    )
    return processing_time


def process_text_sequential(text):
    start_time = time.time()
    dense_embeddings, sparse_embeddings = create_embeddings(text)
    end_time = time.time()
    logging.info(f"Sequential processing took {end_time - start_time:.2f} seconds.")


def main():
    # Set the path to the smaller test file
    test_file_path = "/Users/pablo/Library/CloudStorage/OneDrive-Personal/software/projects/alba/data/testset/decretos_N_5060_2023.pdf"

    # Extract text from the PDF file
    text = extract_text_from_pdf(test_file_path)

    # Define the number of workers and the top-5 most promising batch sizes
    num_workers = 4
    batch_sizes = [100, 500, 1000, 2000, 5000]

    # Split the text into parts based on the number of workers
    parts = split_text(text, num_workers)

    results = []

    for batch_size in batch_sizes:
        logging.info(f"Processing with batch size: {batch_size}")

        # Create batches for each part
        batches = []
        for part in parts:
            batches.extend(create_batches(part, batch_size))

        # Process the batches in parallel and record processing time
        processing_time = process_batches_parallel(batches, num_workers, batch_size)
        results.append((num_workers, batch_size, processing_time))

    # Print summary table
    print("\nSummary Table:")
    print("{:<15} {:<15} {:<15}".format("Workers", "Batch Size", "Processing Time"))
    for result in results:
        print("{:<15} {:<15} {:<15.2f}".format(result[0], result[1], result[2]))

    # Run the non-parallel configuration
    logging.info("Running non-parallel configuration...")
    process_text_sequential(text)


if __name__ == "__main__":
    main()
