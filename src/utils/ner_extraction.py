import json
import re
import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from typing import List, Tuple


class EntityExtractor:
    def __init__(self, locations_file: str):
        # Cargar modelo de spaCy
        self.nlp = spacy.load("es_core_news_lg")

        # Inicializar PhraseMatcher
        self.location_matcher = PhraseMatcher(self.nlp.vocab)

        # Definir patrones de expresiones regulares
        self.law_patterns = [
            re.compile(r"p\.d\. decreto nº (\d+/\d+)", re.IGNORECASE),
            re.compile(r"decreto nº (\d+/\d+)", re.IGNORECASE),
            re.compile(r"real decreto (\d+/\d+)", re.IGNORECASE),
            re.compile(r"ley (\d+/\d+)", re.IGNORECASE),
            re.compile(r"(\d+/\d+)", re.IGNORECASE),
            re.compile(r"decreto presidencial nº (\d+)[,.\s]*", re.IGNORECASE),
        ]
        self.id_patterns = [
            re.compile(r"\bdni[:\s]*?(\d{8}[a-zA-Z])\b", re.IGNORECASE),
            re.compile(r"\b(cif|nif)[:\s]*?([a-zA-Z]\d{8})\b", re.IGNORECASE),
        ]

        # Cargar y configurar ubicaciones
        self.load_locations(locations_file)

    def load_locations(self, locations_file: str):
        # Cargar ubicaciones desde el archivo JSON
        with open(locations_file, "r", encoding="utf-8") as f:
            locations_data = json.load(f)

        # Combina todas las ubicaciones en una lista
        locations = []
        if "municipios" in locations_data:
            locations.extend(locations_data["municipios"])
        if "comarcas" in locations_data:
            locations.extend(locations_data["comarcas"])
        if "pedanias" in locations_data:
            locations.extend(locations_data["pedanias"])

        location_patterns = [self.nlp(loc.lower()) for loc in locations]

        # Añadir patrones al PhraseMatcher
        self.location_matcher.add("LOC", None, *location_patterns)

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        # Procesar el texto con spaCy
        doc = self.nlp(text)
        entities = []

        # Extraer entidades "PER"
        persons = [(ent.label_, ent.text) for ent in doc.ents if ent.label_ == "PER"]
        unique_persons = list(set(persons))
        entities.extend([("PER", person) for _, person in unique_persons])

        # Extraer entidades "LOC"
        location_matches = self.location_matcher(doc)
        locations = [
            (self.nlp.vocab.strings[match_id], doc[start:end].text)
            for match_id, start, end in location_matches
        ]
        unique_locations = list(set(locations))
        entities.extend([("LOC", loc) for _, loc in unique_locations])

        # Extraer entidades "LAW" con expresiones regulares
        laws = []
        for pattern in self.law_patterns:
            matches = pattern.finditer(doc.text)
            laws.extend([("LAW", match.group(1)) for match in matches])
        unique_laws = list(set(laws))
        entities.extend(unique_laws)

        # Extraer entidades "ID" con expresiones regulares
        for pattern in self.id_patterns:
            matches = pattern.finditer(doc.text)
            for match in matches:
                if pattern.pattern.startswith(r"\bdni"):
                    entities.append(("ID", match.group(1)))
                else:
                    entities.append(("ID", match.group(2)))

        # Devolver todas las entidades extraídas
        return entities
