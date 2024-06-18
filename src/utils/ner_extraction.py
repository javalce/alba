import json
import re
import spacy
from spacy.matcher import PhraseMatcher
from typing import List, Tuple
from config.config import Config


class EntityExtractor:
    def __init__(self, locations_file: str):
        # Cargar modelo de spaCy
        false_positives_file = Config().get("false_positives_file")
        self.nlp = spacy.load("es_core_news_lg")

        # Inicializar PhraseMatcher
        self.location_matcher = PhraseMatcher(self.nlp.vocab)

        # Definir patrones de expresiones regulares
        self.law_patterns = [
            re.compile(
                r"(?i)(?:p\.d\.|r\.d\.|real decreto legislativo|decreto presidencial|real decreto|decreto|ley|art\.|p\.a\.|proceso administrativo|proceso\s*administrativo)\s*(?:nº|número)?\s*(\d+/\d+|\d+)"
            )
        ]

        self.id_patterns = [
            re.compile(r"\bdni[:\s]*?(\d{8}[a-zA-Z])\b", re.IGNORECASE),
            re.compile(r"\b(cif|nif)[:\s]*?([a-zA-Z]\d{8})\b", re.IGNORECASE),
            re.compile(
                r"\bSEGEX[:\s]*?(\d+)\b", re.IGNORECASE
            ),  # Patrón para números de contrato después de "SEGEX" o "SEGEX:"
            re.compile(
                r"\b([A-Z]-\d{2}-\d{3}-\d{4})\b", re.IGNORECASE
            ),  # Patrón para códigos de trabajo en el formato "F-21-421-0060"
        ]

        # Cargar y configurar ubicaciones
        self.load_locations(locations_file)

        # Cargar false positives
        self.false_positives = self.load_false_positives(false_positives_file)

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

    def load_false_positives(self, false_positives_file: str) -> set:
        # Cargar false positives desde el archivo
        with open(false_positives_file, "r", encoding="utf-8") as f:
            false_positives = {line.strip().lower() for line in f}
        return false_positives

    def clean_text(self, text: str) -> str:
        # Remove newline characters and multiple spaces
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        # Remove question and exclamation marks
        text = text.replace("?", "").replace("!", "").replace("¿", "").replace("¡", "")

        # Procesar el texto con spaCy
        doc = self.nlp(text)

        # Extraer entidades "ID" con expresiones regulares y marcarlas para removerlas
        id_entities = []
        for pattern in self.id_patterns:
            matches = pattern.finditer(doc.text)
            for match in matches:
                if pattern.pattern.startswith(r"\bdni"):
                    id_entities.append(("ID", self.clean_text(match.group(1))))
                elif pattern.pattern.startswith(r"\b(cif|nif)"):
                    id_entities.append(("ID", self.clean_text(match.group(2))))
                else:
                    id_entities.append(("ID", self.clean_text(match.group(1))))

        entities = id_entities

        # Extraer entidades "PER"
        persons = [
            (ent.label_, self.clean_text(ent.text))
            for ent in doc.ents
            if ent.label_ == "PER"
        ]

        # Filter out PER entities that contain false positives
        persons = [
            (label, text)
            for label, text in persons
            if not any(fp in text.lower() for fp in self.false_positives)
        ]

        # Remove duplicates and add to the entities list
        unique_persons = list(set(persons))
        entities.extend([("PER", person) for _, person in unique_persons])

        # Extraer entidades "LOC" usando spaCy y PhraseMatcher
        locations = [
            (ent.label_, self.clean_text(ent.text))
            for ent in doc.ents
            if ent.label_ == "LOC"
        ]
        location_matches = self.location_matcher(doc)
        locations.extend(
            [
                ("LOC", self.clean_text(doc[start:end].text))
                for _, start, end in location_matches
            ]
        )

        # Filter out LOC entities that contain false positives
        locations = [
            (label, text)
            for label, text in locations
            if not any(fp in text.lower() for fp in self.false_positives)
        ]

        # Remove duplicates and add to the entities list
        unique_locations = list(set(locations))
        entities.extend(unique_locations)

        # Extraer entidades "LAW" con expresiones regulares
        laws = []
        for pattern in self.law_patterns:
            matches = pattern.finditer(doc.text)
            laws.extend([("LAW", self.clean_text(match.group(1))) for match in matches])
        unique_laws = list(set(laws))
        entities.extend(unique_laws)

        # Devolver todas las entidades extraídas
        return entities
