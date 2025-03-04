import csv
from datetime import datetime

# --------------------- CONFIGURATION ---------------------
INPUT_CSV = "DATA/RAECMBD_454_20241226-163036.csv"
OUTPUT_CSV = "DATA/RAECMBD_454_20241226-163036_pruned.csv"

# We will prune entries strictly AFTER end of 15 Mar 2020
CUTOFF_DATE = datetime(2020, 3, 15, 23, 59)

# A dictionary mapping original Spanish headers -> English safe headers
HEADER_MAP = {
    "Año": "Year",
    "Comunidad Autónoma": "AutonomousCommunity",
    "Provincia": "Province",
    "Fecha de nacimiento": "DateOfBirth",
    "Sexo": "Sex",
    "CCAA Residencia": "AutonomousCommunityResidence",
    "Provincia Residencia": "ProvinceResidence",
    "Fecha de Ingreso": "AdmissionDate",
    "Fecha de Fin Contacto": "ContactEndDate",
    "Fecha de Intervención": "InterventionDate",
    "Tipo Alta": "DischargeType",
    "Régimen Financiación": "FinancingRegime",
    "Circunstancia de Contacto": "ContactCircumstance",
    "Estancia Días": "DaysOfStay",
    "Diagnóstico Principal": "PrincipalDiagnosis",
    "Capitulo Diagnóstico Principal": "PrincipalDiagnosisChapter",
    "Sección Diagnóstico Principal": "PrincipalDiagnosisSection",
    "Categoría Diagnóstico Principal": "PrincipalDiagnosisCategory",
    "Subcategoría Diagnóstico Principal": "PrincipalDiagnosisSubcategory",
    "Diagnóstico 2": "Diagnosis2",
    "Diagnóstico 3": "Diagnosis3",
    "Diagnóstico 4": "Diagnosis4",
    "Diagnóstico 5": "Diagnosis5",
    "Diagnóstico 6": "Diagnosis6",
    "Diagnóstico 7": "Diagnosis7",
    "Diagnóstico 8": "Diagnosis8",
    "Diagnóstico 9": "Diagnosis9",
    "Diagnóstico 10": "Diagnosis10",
    "Diagnóstico 11": "Diagnosis11",
    "Diagnóstico 12": "Diagnosis12",
    "Diagnóstico 13": "Diagnosis13",
    "Diagnóstico 14": "Diagnosis14",
    "Diagnóstico 15": "Diagnosis15",
    "Diagnóstico 16": "Diagnosis16",
    "Diagnóstico 17": "Diagnosis17",
    "Diagnóstico 18": "Diagnosis18",
    "Diagnóstico 19": "Diagnosis19",
    "Diagnóstico 20": "Diagnosis20",
    "Procedimiento 1": "Procedure1",
    "Procedimiento 2": "Procedure2",
    "Procedimiento 3": "Procedure3",
    "Procedimiento 4": "Procedure4",
    "Procedimiento 5": "Procedure5",
    "Procedimiento 6": "Procedure6",
    "Procedimiento 7": "Procedure7",
    "Procedimiento 8": "Procedure8",
    "Procedimiento 9": "Procedure9",
    "Procedimiento 10": "Procedure10",
    "Procedimiento 11": "Procedure11",
    "Procedimiento 12": "Procedure12",
    "Procedimiento 13": "Procedure13",
    "Procedimiento 14": "Procedure14",
    "Procedimiento 15": "Procedure15",
    "Procedimiento 16": "Procedure16",
    "Procedimiento 17": "Procedure17",
    "Procedimiento 18": "Procedure18",
    "Procedimiento 19": "Procedure19",
    "Procedimiento 20": "Procedure20",
    "Procedimiento Externo 1": "ExternalProcedure1",
    "Procedimiento Externo 2": "ExternalProcedure2",
    "Procedimiento Externo 3": "ExternalProcedure3",
    "Procedimiento Externo 4": "ExternalProcedure4",
    "Procedimiento Externo 5": "ExternalProcedure5",
    "Procedimiento Externo 6": "ExternalProcedure6",
    "Morfología 1": "Morphology1",
    "Morfología 2": "Morphology2",
    "Morfología 3": "Morphology3",
    "Morfología 4": "Morphology4",
    "Morfología 5": "Morphology5",
    "Morfología 6": "Morphology6",
    "GRD APR": "APRDRG",
    "Tipo GRD APR": "APRDRGType",
    "Nivel Severidad APR": "APRSeverityLevel",
    "Riesgo Mortalidad APR": "APRMortalityRisk",
    "Peso Español APR": "APRSpanishWeight",
    "Coste APR": "APRCost"
}

# --------------------- MAIN SCRIPT -----------------------
def main():
    with open(INPUT_CSV, mode="r", encoding="utf-8", newline="") as infile, \
         open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as outfile:

        # Use comma delimiter (since the file has commas)
        reader = csv.reader(infile, delimiter=",")
        writer = csv.writer(outfile, delimiter=",")

        # Read the raw header
        raw_header_row = next(reader)
        print("DEBUG: Raw header row as read by Python:")
        print(raw_header_row)

        # Remove BOM characters and extra spaces from each column name
        original_headers = [col.replace('\ufeff', '').strip() for col in raw_header_row]
        print("DEBUG: Cleaned headers:")
        print(original_headers)

        # Check for "Fecha de Ingreso" (which we've mapped to "AdmissionDate")
        if "Fecha de Ingreso" not in original_headers:
            raise ValueError(f"'Fecha de Ingreso' not in {original_headers}")

        # Build the safe headers in English
        new_headers = []
        for col_name in original_headers:
            if col_name in HEADER_MAP:
                new_headers.append(HEADER_MAP[col_name])
            else:
                # fallback: remove non-alphanumeric to create a safe name
                safe_name = "".join(ch for ch in col_name if ch.isalnum())
                if not safe_name:
                    safe_name = "UnnamedColumn"
                new_headers.append(safe_name)

        writer.writerow(new_headers)

        # Get the index of "Fecha de Ingreso" in the original (Spanish) headers
        fecha_ingreso_idx = original_headers.index("Fecha de Ingreso")
        # And figure out which position "AdmissionDate" occupies in new_headers
        admission_date_idx = new_headers.index("AdmissionDate")

        # Iterate over rows, parse the date, and prune if after 15-Mar-2020
        for row in reader:
            if len(row) <= fecha_ingreso_idx:
                continue

            ingreso_str = row[fecha_ingreso_idx].strip()

            # "DDMMYYYY HHMM" format
            try:
                ingreso_date = datetime.strptime(ingreso_str, "%d%m%Y %H%M")
            except ValueError:
                # If date parsing fails, skip or handle differently
                continue

            if ingreso_date <= CUTOFF_DATE:
                # Keep the row as is (columns are still aligned, just the headers changed)
                writer.writerow(row)

    print(f"Done! Filtered rows saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
