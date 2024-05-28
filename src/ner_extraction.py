from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Use a pre-trained Spanish NER model
model_name = "mrm8488/bert-spanish-cased-finetuned-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)


def extract_entities(text):
    ner_results = ner_pipeline(text)
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
    return filtered_entities
