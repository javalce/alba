import time
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import re

# Texto largo y sofisticado en español para prueba
texto = """
JOSE MANUEL HERREROS PEÑAFIEL es un ingeniero en telecomunicaciones que trabaja en TELEFONICA.
MARIA JOSE LOPEZ trabaja en la misma empresa como analista de datos. Ambos viven en Madrid, España.
JOSE FERNANDO GONZALEZ MARTINEZ es su compañero de trabajo y reside en Barcelona.
MARIA ISABEL FERNANDEZ PEREZ también es parte del equipo y vive en Valencia.
TELEFONICA es una de las empresas de telecomunicaciones más grandes de España.
El CEO de TELEFONICA, LUIS MIGUEL GARCIA, ha anunciado una nueva estrategia para el 2023.
El equipo planea asistir a la conferencia de tecnología en Lisboa el próximo mes de septiembre.
Además, recientemente, JUAN CARLOS MARTINEZ GOMEZ y ANA MARIA RODRIGUEZ RAMIREZ recibieron premios por sus contribuciones a proyectos internacionales.
Las oficinas de TELEFONICA se encuentran en el Paseo de la Castellana, una de las arterias principales de Madrid.
Se espera que el impacto de la nueva estrategia incremente los ingresos en un 15% para el final del año.
"""

# Configuración del modelo basado en transformers
max_length = 512
model_name = "mrm8488/bert-spanish-cased-finetuned-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline(
    "ner", aggregation_strategy="simple", model=model, tokenizer=tokenizer
)


def split_text(text):
    tokenized_text = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )["input_ids"][0]
    chunks = [
        tokenized_text[i : i + max_length]
        for i in range(0, len(tokenized_text), max_length)
    ]
    return chunks


def preprocess_text(text):
    pattern = re.compile(
        r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b"
    )
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(match, match.replace(" ", "_"))
    return text, matches


def postprocess_entities(entities, matches):
    for entity in entities:
        for match in matches:
            entity["text"] = entity["text"].replace(match.replace(" ", "_"), match)
    return entities


def extract_entities_transformers(text):
    preprocessed_text, matches = preprocess_text(text)
    chunks = split_text(preprocessed_text)
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
        filtered_entities = [
            {"text": ent["text"], "label": ent["label"]}
            for ent in entities
            if not ent["text"].startswith("##") and len(ent["text"].strip()) > 1
        ]
        all_entities.extend(filtered_entities)

    final_entities = postprocess_entities(all_entities, matches)
    return final_entities


# Configuración del modelo basado en spaCy
nlp = spacy.load("es_core_news_md")


def extract_entities_spacy(text):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities


# Medir tiempo de procesamiento y extracción de entidades con transformers
start_time = time.time()
entities_transformers = extract_entities_transformers(texto)
transformers_time = time.time() - start_time

# Medir tiempo de procesamiento y extracción de entidades con spaCy
start_time = time.time()
entities_spacy = extract_entities_spacy(texto)
spacy_time = time.time() - start_time

# Comparación de resultados
print("Resultados con Transformers:")
for ent in entities_transformers:
    print(f"Texto: {ent['text']}, Etiqueta: {ent['label']}")

print("\nResultados con spaCy:")
for ent in entities_spacy:
    print(f"Texto: {ent['text']}, Etiqueta: {ent['label']}")

print(f"\nTiempo de procesamiento con Transformers: {transformers_time:.4f} segundos")
print(f"Tiempo de procesamiento con spaCy: {spacy_time:.4f} segundos")
