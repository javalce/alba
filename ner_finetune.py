import fitz  # PyMuPDF
import tqdm
import spacy
from spacy.cli import download
from spacy.training import Example
from spacy.util import minibatch, compounding
import re
from sklearn.model_selection import train_test_split


# Function to download spaCy model if not already downloaded
def download_spacy_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        download(model_name)
        spacy.load(model_name)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        print(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


# Function to annotate text
def annotate_text(text):
    annotations = []

    # Regular expression patterns for different entities
    date_pattern = (
        r"\d{1,2} de \w+ de \d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}|\d{2}-\d{2}-\d{4}"
    )
    law_pattern = (
        r"Real Decreto \d+/\d+|R\.D\.L\. \d+/\d+|Decreto [Nº ]*\d+ de \d+/\d+/\d+"
    )
    contract_pattern = (
        r"\breferencia \d+/\d+|\bEXPEDIENTE SEGEX: \d+[A-Z]*\b|\bSEGEX \d+[A-Z]\b"
    )
    id_pattern = r"\b(?:NIF|CIF|DNI)[: ]?\d{8}[A-HJ-NP-TV-Z]\b|\b(?:NIF|CIF|DNI)[: ]?[A-HJ-NP-SUVW]\d{7}[A-J]\b|\b\d{8}[A-HJ-NP-TV-Z]\b"

    # Find all matches in the text
    dates = [(m.start(), m.end(), "DATE") for m in re.finditer(date_pattern, text)]
    laws = [(m.start(), m.end(), "LAW") for m in re.finditer(law_pattern, text)]
    contracts = [
        (m.start(), m.end(), "CONTRACT") for m in re.finditer(contract_pattern, text)
    ]
    ids = [(m.start(), m.end(), "ID") for m in re.finditer(id_pattern, text)]

    annotations.extend(dates)
    annotations.extend(laws)
    annotations.extend(contracts)
    annotations.extend(ids)

    return annotations


# Function to convert annotated text to spaCy's training format
def create_training_data(text, annotations):
    entities = [(start, end, label) for start, end, label in annotations]
    return [(text, {"entities": entities})]


# Ensure the spaCy model is downloaded
download_spacy_model("es_core_news_sm")

# Load the pre-trained Spanish model
nlp = spacy.load("es_core_news_sm")

# Extract text from PDF
pdf_path = "/Users/pablo/Library/CloudStorage/OneDrive-Personal/software/projects/alba/data/testset/Impresion_libro_de_decretos_N_1_2023___N_5060_2023_chunk_1_to_150.pdf"
text = extract_text_from_pdf(pdf_path)

if text:
    # Annotate text
    annotations = annotate_text(text)

    # Create training data
    TRAIN_DATA = create_training_data(text, annotations)

    # Log the number of annotations and sample size
    print(f"Number of training samples: {len(TRAIN_DATA)}")
    if len(TRAIN_DATA) == 0:
        print("No annotations found. Exiting.")
        exit()

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(TRAIN_DATA, test_size=0.2, random_state=42)

    # Add new labels to the NER pipeline
    ner = nlp.get_pipe("ner")
    for label in ["DATE", "LAW", "CONTRACT", "ID"]:
        ner.add_label(label)

    # Disable other pipelines during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        # Create an optimizer
        optimizer = nlp.resume_training()

        # Convert training data into spaCy's format
        train_examples = [
            Example.from_dict(nlp.make_doc(text), annotations)
            for text, annotations in train_data
        ]
        val_examples = [
            Example.from_dict(nlp.make_doc(text), annotations)
            for text, annotations in val_data
        ]

        # Early stopping parameters
        patience = 5
        best_loss = float("inf")
        no_improvement = 0

        # Train the model
        for i in tqdm.tqdm(range(30)):
            losses = {}
            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses)
            print(f"Iteration {i}, Losses: {losses}")

            # Evaluate on validation set
            val_losses = {}
            for val_batch in minibatch(
                val_examples, size=compounding(4.0, 32.0, 1.001)
            ):
                nlp.update(val_batch, sgd=optimizer, losses=val_losses, drop=0.0)
            print(f"Iteration {i}, Validation Losses: {val_losses}")

            # Check for early stopping
            if val_losses["ner"] < best_loss:
                best_loss = val_losses["ner"]
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print("Early stopping due to no improvement in validation loss.")
                    break

    # Save the fine-tuned model
    model_path = "fine_tuned_ner_model"
    nlp.to_disk(model_path)
    print(f"Model saved to {model_path}")

    # Test the trained model
    nlp = spacy.load(model_path)
    doc = nlp(
        "Visto el escrito presentado por el contratista ELECTROMONTAJES MAYE SOCIEDAD LIMITADA con NIF: B02487296, con fecha 12 de junio de 2023, y nº registro de entrada: 15416, sobre ampliación del plazo de ejecución de la obra que más abajo se indica (Instalación de sistemas fotovoltaicos en 2 escuelas infantiles y en la Casa de la Cultura de Aguas Nuevas en el municipio de Albacete y de un sistema fotovoltaico en el municipio TOMADOR: Ayuntamiento de Fuentealbilla DNI: P0203400G POLIZA: 049412072 ANTONIO MOLINA TORRES, CON DNI nº 44,.389.722K en la Vía Verde Sierra de Alcaraz al circular con un vehículo a motor, matrícula 1941 - KCP con DNI 05162741T, solicitando auto TITULAR: PLUS ULTRA SEGUROS (número de póliza: ES68111A3001481). Asegurada: Elena López Marín. DNI: 74510365-W CIF/NIF: A30014831 DOMICILIO: Plaza de Las Cortes, N.º"
    )
    for ent in doc.ents:
        print(ent.text, ent.label_)
else:
    print("No text extracted from the PDF.")
