from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Maximum length of input tokens
max_length = 512

# Use a pre-trained Spanish NER model
model_name = "mrm8488/bert-spanish-cased-finetuned-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create the NER pipeline with truncation and padding
ner_pipeline = pipeline(
    "ner",
    aggregation_strategy="simple",
    model=model,
    tokenizer=tokenizer,
)


def split_text(text):
    # Tokenize the text and split into chunks of max_length
    tokenized_text = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )["input_ids"][0]
    chunks = [
        tokenized_text[i : i + max_length]
        for i in range(0, len(tokenized_text), max_length)
    ]
    return chunks


# Load the NER model
def extract_entities(text):

    chunks = split_text(text)
    all_entities = []

    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        ner_results = ner_pipeline(chunk_text)
        entities = []
        for ent in ner_results:
            word = ent["word"]
            if word.startswith("##"):
                if entities:
                    entities[-1]["text"] += word[2:]
            else:
                entities.append({"text": word, "label": ent["entity_group"]})
        # Filter out subwords and irrelevant entities
        filtered_entities = [
            {"text": ent["text"], "label": ent["label"]}
            for ent in entities
            if not ent["text"].startswith("##") and len(ent["text"].strip()) > 1
        ]
        all_entities.extend(filtered_entities)
    return all_entities
