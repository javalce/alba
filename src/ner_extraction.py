from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Use a pre-trained Spanish NER model
model_name = "mrm8488/bert-spanish-cased-finetuned-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)


def extract_entities(text):
    ner_results = ner_pipeline(text)
    entities = [
        {"text": ent["word"], "label": ent["entity_group"]} for ent in ner_results
    ]
    return entities
