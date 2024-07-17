# Standard library imports
import json
import re
from typing import List, Tuple

# Third-party imports
import spacy

# Local application imports
from alba.config import Config
from spacy.matcher import PhraseMatcher


class EntityExtractor:
    """
    A class for extracting entities from text using spaCy and regular expressions.
    """

    def __init__(self, config: Config, locations_file: str):
        """
        Initialize the EntityExtractor with a spaCy model, location matcher, and regex patterns.

        Args:
            locations_file (str): Path to the JSON file containing location data.
        """
        # Load spaCy model
        false_positives_file = config.FALSE_POSITIVES_FILE
        self.nlp = spacy.load("es_core_news_lg")

        # Initialize PhraseMatcher
        self.location_matcher = PhraseMatcher(self.nlp.vocab)

        # Define regex patterns
        self.law_patterns = [
            re.compile(
                r"(?i)(?:p\.d\.|r\.d\.|real decreto legislativo|decreto presidencial|real decreto|decreto|ley|art\.|p\.a\.|proceso administrativo|proceso\s*administrativo)\s*(?:nº|número)?\s*(\d+/\d+|\d+)"
            ),
            re.compile(r"\b(\d{1,5}/\d{1,5})\b"),
        ]

        self.id_patterns = [
            re.compile(r"\bdni[:\s]*?(\d{8}[a-zA-Z])\b", re.IGNORECASE),
            re.compile(r"\b(cif|nif)[:\s]*?([a-zA-Z]\d{8})\b", re.IGNORECASE),
            re.compile(
                r"\bSEGEX[:\s]*?(\d+)\b", re.IGNORECASE
            ),  # Pattern for contract numbers after "SEGEX" or "SEGEX:"
            re.compile(
                r"\b([A-Z]-\d{2}-\d{3}-\d{4})\b", re.IGNORECASE
            ),  # Pattern for job codes in the format "F-21-421-0060"
        ]

        # Load and configure locations
        self.load_locations(locations_file)

        # Load false positives
        self.false_positives = self.load_false_positives(false_positives_file)

    def load_locations(self, locations_file: str):
        """
        Load locations from the JSON file and add them to the location matcher.

        Args:
            locations_file (str): Path to the JSON file containing location data.
        """
        # Load locations from the JSON file
        with open(locations_file, encoding="utf-8") as f:
            locations_data = json.load(f)

        # Combine all locations into a list
        locations = []
        if "municipios" in locations_data:
            locations.extend(locations_data["municipios"])
        if "comarcas" in locations_data:
            locations.extend(locations_data["comarcas"])
        if "pedanias" in locations_data:
            locations.extend(locations_data["pedanias"])

        location_patterns = [self.nlp(loc.lower()) for loc in locations]

        # Add patterns to the PhraseMatcher
        self.location_matcher.add("LOC", None, *location_patterns)

    def load_false_positives(self, false_positives_file: str) -> set:
        """
        Load false positives from the file and return them as a set.

        Args:
            false_positives_file (str): Path to the file containing false positives.

        Returns:
            set: A set of false positive strings.
        """
        # Load false positives from the file
        with open(false_positives_file, encoding="utf-8") as f:
            false_positives = {line.strip().lower() for line in f}
        return false_positives

    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing newline characters and multiple spaces.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Remove newline characters and multiple spaces
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        # Remove question and exclamation marks
        text = text.replace("?", "").replace("!", "").replace("¿", "").replace("¡", "")
        # Remove leading and trailing punctuation
        text = text.strip(".,;:¡!¿?()[]{}")
        # Remove leading and trailing whitespace and hyphens
        text = text.strip().strip("-")
        # Remove Don, Doña, Sr., Sra., Dr., Dra., D., Da., Dª., etc.
        text = re.sub(r"\b(Don|Doña|Sr\.|Sra\.|Dr\.|Dra\.|D\.|Da\.|Dª\.)\s+", "", text)
        return text

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from the given text using spaCy and regex patterns.

        Args:
            text (str): The input text to extract entities from.

        Returns:
            List[Tuple[str, str]]: A list of extracted entities, each represented as a tuple of (entity_type, entity_text).
        """

        # Process the text with spaCy
        doc = self.nlp(text)

        # Extract "ID" entities using regex patterns and mark them for removal
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

        # Extract "PER" entities
        persons = [
            (ent.label_, self.clean_text(ent.text)) for ent in doc.ents if ent.label_ == "PER"
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

        # Extract "LOC" entities using spaCy and PhraseMatcher
        locations = [
            (ent.label_, self.clean_text(ent.text)) for ent in doc.ents if ent.label_ == "LOC"
        ]
        location_matches = self.location_matcher(doc)
        locations.extend(
            [("LOC", self.clean_text(doc[start:end].text)) for _, start, end in location_matches]
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

        # Extract "LAW" entities using regex patterns
        laws = []
        for pattern in self.law_patterns:
            matches = pattern.finditer(doc.text)
            laws.extend([("LAW", self.clean_text(match.group(1))) for match in matches])
        unique_laws = list(set(laws))
        entities.extend(unique_laws)

        return entities
