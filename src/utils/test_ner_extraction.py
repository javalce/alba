import unittest

import fitz  # PyMuPDF
from ner_extraction import EntityExtractor


class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        # Crear un archivo JSON temporal con ubicaciones de ejemplo
        self.locations_file = "/Users/pablo/Library/CloudStorage/OneDrive-Personal/software/projects/alba/config/locations.json"
        self.extractor = EntityExtractor(self.locations_file)

    def test_specific_entity_extraction(self):
        text = "El decreto nº 123/2021 establece la normativa de la región. Art. 156. Decreto nº 456/2021. Ley 789/2021. decreto número 012/2021. Real Decreto 345/2021. R.D. 678/2021. Real Decreto Legislativo 901/2021. Ley 123/2021. 123/2021. R.D. 456/2021. Ley 456/2021. ley 432/2021. Ley 123. ¿A quién corresponde el DNI 4438791R? No lo sabemos. CIF: A12345678. NIF: B87654321. SEGEX 123. F-21-421-0060. ¿Dónde está Madrid? ¿Dónde está la comarca de la Sierra de Gredos"
        entities = self.extractor.extract_entities(text)
        for entity in entities:
            print(entity)

    def test_standalone_question_extraction(self):
        text = "¿A qué corresponde el código F-21-551-720?"
        entities = self.extractor.extract_entities(text)
        for entity in entities:
            print(entity)

    def test_extract_all_entities_from_pdf(self):
        pdf_path = "data/testset/decretos_part_8.pdf"  # Ruta al archivo PDF
        pdf_document = fitz.open(pdf_path)

        all_entities = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            entities = self.extractor.extract_entities(text)
            all_entities.extend(entities)

        for entity in all_entities:
            print(entity)


if __name__ == "__main__":
    unittest.main(exit=False)  # Cambia exit=True a exit=False para evitar SystemExit
