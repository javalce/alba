# Custom file splitter to split the large decrees file into smaller chunks for processing.

from pathlib import Path

import PyPDF2


def split_pdf(file_path, output_dir, splits):
    pdf = PyPDF2.PdfReader(file_path)
    num_pages = len(pdf.pages)
    splits = sorted(splits)
    start_page = 0
    chunk_paths = []

    for end_page in splits:
        if end_page > num_pages:
            print(
                f"Warning: End page {end_page} exceeds the total number of pages {num_pages}. Adjusting to {num_pages}."
            )
            end_page = num_pages
        if start_page >= num_pages:
            break
        writer = PyPDF2.PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(pdf.pages[page_num])
        chunk_path = output_dir / f"{Path(file_path).stem}_chunk_{start_page + 1}_to_{end_page}.pdf"
        with open(chunk_path, "wb") as chunk_file:
            writer.write(chunk_file)
        chunk_paths.append(chunk_path)
        start_page = end_page

    if start_page < num_pages:
        writer = PyPDF2.PdfWriter()
        for page_num in range(start_page, num_pages):
            writer.add_page(pdf.pages[page_num])
        chunk_path = (
            output_dir / f"{Path(file_path).stem}_chunk_{start_page + 1}_to_{num_pages}.pdf"
        )
        with open(chunk_path, "wb") as chunk_file:
            writer.write(chunk_file)
        chunk_paths.append(chunk_path)

    return chunk_paths


# Example usage
file_path = "/Users/pablo/Library/CloudStorage/OneDrive-Personal/software/projects/alba/data/testset/Impresion_libro_de_decretos_N_1_2023___N_5060_2023.pdf"
output_dir = Path(
    "/Users/pablo/Library/CloudStorage/OneDrive-Personal/software/projects/alba/data/testset/"
)
output_dir.mkdir(parents=True, exist_ok=True)
splits = [150, 300, 450, 600, 750]

chunk_paths = split_pdf(file_path, output_dir, splits)
print(f"Created {len(chunk_paths)} chunks:")
for chunk_path in chunk_paths:
    print(chunk_path)
