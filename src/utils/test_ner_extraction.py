import unittest
import json
import os
from ner_extraction import (
    EntityExtractor,
)  # Asegúrate de que la ruta y el nombre del archivo sean correctos


class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        # Crear un archivo JSON temporal con ubicaciones de ejemplo
        self.locations_file = "temp_locations.json"
        locations_data = {
            "municipios": ["Madrid", "Barcelona", "Almansa", "Alpera"],
            "comarcas": ["La Manchuela"],
            "pedanias": ["Casas de Lázaro", "El Ballestero"],
        }
        with open(self.locations_file, "w", encoding="utf-8") as f:
            json.dump(locations_data, f)

        self.extractor = EntityExtractor(self.locations_file)

    def tearDown(self):
        # Eliminar el archivo JSON temporal
        os.remove(self.locations_file)

    def test_extract_person(self):
        text = "Esta es una muestra de texto que menciona a Juan Pérez."
        entities = self.extractor.extract_entities(text)
        self.assertIn(("PER", "Juan Pérez"), entities)

    def test_extract_loc(self):
        text = "Esta es una muestra de texto que menciona Madrid y Barcelona."
        entities = self.extractor.extract_entities(text)
        self.assertIn(("LOC", "Madrid"), entities)
        self.assertIn(("LOC", "Barcelona"), entities)

    def test_extract_law(self):
        text = "P.D. Decreto Nº 1977/2019, Real Decreto 781/86, ley 20/21."
        entities = self.extractor.extract_entities(text)
        self.assertIn(("LAW", "1977/2019"), entities)
        self.assertIn(("LAW", "781/86"), entities)
        self.assertIn(("LAW", "20/21"), entities)

    def test_extract_id(self):
        text = "DNI: 52759644J, CIF: A28141935. Otro DNI: 12345678Z y otro CIF B12345678. CIF/NIF: C12345678. NIF/CIF D12345678."
        entities = self.extractor.extract_entities(text)
        self.assertIn(("ID", "52759644J"), entities)
        self.assertIn(("ID", "A28141935"), entities)
        self.assertIn(("ID", "12345678Z"), entities)
        self.assertIn(("ID", "B12345678"), entities)
        self.assertIn(("ID", "C12345678"), entities)
        self.assertIn(("ID", "D12345678"), entities)

    def test_extract_all_entities(self):
        text = """
        Juan Pérez, con DNI 52759644J, trabaja en la empresa XYZ, cuyo CIF es A28141935. 
        María García, que reside en Madrid, tiene el NIF Z1234567L y su amigo Carlos Díaz también vive en Barcelona, 
        posee el DNI 12345678Z. En un viaje a La Manchuela, Carlos mencionó el Real Decreto 781/86 y el Decreto Nº 1234/2020.

        Además, en la reunión anual de la empresa ABC (CIF: B87654321), celebrada en Casas de Lázaro, 
        se discutieron las implicaciones del P.D. Decreto Nº 1977/2019. La empresa DEF, con NIF Q9876543P, 
        también presentó sus informes en la misma reunión. 

        En la conferencia, que tuvo lugar en El Ballestero, se mencionaron varias leyes importantes, 
        como la ley 20/21 y el decreto presidencial nº 12345. El Dr. Ricardo López, con DNI 87654321H, 
        destacó la relevancia de la normativa europea, citando el Decreto Nº 5678 y el Real Decreto 1122/2018.

        Más tarde, en una cena en La Herrera, con la asistencia de varias personas, incluyendo a Ana Martínez, 
        se volvió a hablar del Decreto Nº 2345/2021. Javier Sánchez, quien tiene el NIF W1234567A, 
        y su colega Laura Gómez (CIF: H76543210), participaron activamente en las discusiones.

        Finalmente, en una visita a La Manchuela, donde también se encontraban representantes de Almansa y Alpera, 
        se revisaron documentos adicionales, incluyendo la ley 45/2020 y el Real Decreto 654/2019.
        También se habló del Decreto Presidencial nº 6789 y se mencionaron ubicaciones adicionales 
        como la pedanía de Casas de Lázaro y la entidad local menor de El Ballestero.
        """
        entities = self.extractor.extract_entities(text)
        expected_entities = [
            ("PER", "Juan Pérez"),
            ("PER", "María García"),
            ("PER", "Carlos Díaz"),
            ("PER", "Ricardo López"),
            ("PER", "Ana Martínez"),
            ("PER", "Javier Sánchez"),
            ("PER", "Laura Gómez"),
            ("LOC", "Madrid"),
            ("LOC", "Barcelona"),
            ("LOC", "La Manchuela"),
            ("LOC", "Casas de Lázaro"),
            ("LOC", "El Ballestero"),
            ("LOC", "La Herrera"),
            ("LOC", "Almansa"),
            ("LOC", "Alpera"),
            ("LAW", "781/86"),
            ("LAW", "1234/2020"),
            ("LAW", "1977/2019"),
            ("LAW", "20/21"),
            ("LAW", "12345"),
            ("LAW", "5678"),
            ("LAW", "1122/2018"),
            ("LAW", "2345/2021"),
            ("LAW", "45/2020"),
            ("LAW", "654/2019"),
            ("LAW", "6789"),
            ("ID", "52759644J"),
            ("ID", "A28141935"),
            ("ID", "Z1234567L"),
            ("ID", "12345678Z"),
            ("ID", "B87654321"),
            ("ID", "Q9876543P"),
            ("ID", "87654321H"),
            ("ID", "W1234567A"),
            ("ID", "H76543210"),
        ]
        for entity in expected_entities:
            self.assertIn(entity, entities)


if __name__ == "__main__":
    unittest.main(exit=False)  # Cambia exit=True a exit=False para evitar SystemExit
