import camelot
import json
import re


def extract_locations_from_pdf(pdf_path):
    municipios = []
    comarcas = []
    pedanias = []

    tables = camelot.read_pdf(pdf_path, pages="all")

    for table in tables:
        for row in table.df.itertuples(index=False):
            if len(row) >= 7:
                municipio = str(row[0]).strip()
                comarca = str(row[6]).strip()
                pedania_data = str(row[4]).strip()

                if municipio:
                    municipios.append(municipio)
                if comarca:
                    comarcas.append(comarca)
                if pedania_data:
                    if pedania_data.lower() != "no tiene":
                        pedania_data = re.sub(
                            r"Pedanías:\s*", "", pedania_data, flags=re.IGNORECASE
                        )
                        pedania_data = re.sub(
                            r"Entidad local menor:.*",
                            "",
                            pedania_data,
                            flags=re.IGNORECASE,
                        )
                        pedania_list = re.split(r"[,;.]\s*", pedania_data)
                        for pedania in pedania_list:
                            alternative_names = re.split(r"\s+o\s+", pedania)
                            pedanias.extend(alternative_names)

    # Remove duplicates while preserving order
    municipios = list(dict.fromkeys(municipios))
    comarcas = list(dict.fromkeys(comarcas))
    pedanias = list(dict.fromkeys(pedanias))

    # Create a dictionary with the grouped location data
    location_data = {
        "municipios": municipios,
        "comarcas": comarcas,
        "pedanias": pedanias,
    }

    return location_data


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# Specify the path to your PDF file
pdf_path = "data/Anexo_Municipios_de_la_provincia_de_Albacete.pdf"

# Specify the output path for the JSON file
output_path = "data/locations.json"

# Extract the location data from the PDF
location_data = extract_locations_from_pdf(pdf_path)

# Save the location data to a JSON file
save_to_json(location_data, output_path)

print(f"Location data has been saved to {output_path}")
