import csv
from datetime import datetime

# --------------------- CONFIGURATION ---------------------
INPUT_CSV = "DATA/RAECMBD_454_20241226-163036.csv"
OUTPUT_CSV = "DATA/RAECMBD_454_20241226-163036_pruned.csv"
DISCARDED_CSV = "DATA/RAECMBD_454_20241226-163036_discarded.csv"

# Only keep rows with Fecha de Ingreso <= 15 Mar 2020
CUTOFF_DATE = datetime(2020, 3, 15, 23, 59)

# Spanish -> English header mapping
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

def main():
    # Stats counters
    total_rows = 0
    parse_fail_count = 0
    date_pruned_count = 0
    diag_pruned_count = 0
    sex_pruned_count = 0
    kept_count = 0

    with open(INPUT_CSV, mode="r", encoding="utf-8", newline="") as infile, \
         open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as outfile, \
         open(DISCARDED_CSV, mode="w", encoding="utf-8", newline="") as discardfile:
        
        reader = csv.reader(infile, delimiter=",")
        writer = csv.writer(outfile, delimiter=",")
        discard_writer = csv.writer(discardfile, delimiter=",")

        # Read the raw header
        raw_header_row = next(reader)
        original_headers = [col.replace('\ufeff', '').strip() for col in raw_header_row]

        # Must have "Fecha de Ingreso"
        if "Fecha de Ingreso" not in original_headers:
            raise ValueError(f"'Fecha de Ingreso' not found in {original_headers}")

        # Convert Spanish -> English
        new_headers = []
        for col_name in original_headers:
            if col_name in HEADER_MAP:
                new_headers.append(HEADER_MAP[col_name])
            else:
                # fallback: remove non-alphanumeric
                safe_name = "".join(ch for ch in col_name if ch.isalnum())
                if not safe_name:
                    safe_name = "UnnamedColumn"
                new_headers.append(safe_name)

        # Write headers to both main and discard CSV
        writer.writerow(new_headers)
        discard_writer.writerow(new_headers)

        # Indices
        fecha_ingreso_idx = original_headers.index("Fecha de Ingreso")
        
        # We'll also locate the "Sexo" -> "Sex" index, for checking 1 vs 2
        try:
            sex_idx = new_headers.index("Sex")
        except ValueError:
            sex_idx = None

        # We'll find PrincipalDiagnosis if it exists
        try:
            idx_principal_diag = new_headers.index("PrincipalDiagnosis")
        except ValueError:
            idx_principal_diag = None

        # Iterate over rows
        for row in reader:
            total_rows += 1

            if len(row) < len(original_headers):
                # Incomplete row - discard
                discard_writer.writerow(row)
                continue

            # Parse the admission date
            date_str = row[fecha_ingreso_idx].strip()
            try:
                ingreso_date = datetime.strptime(date_str, "%d%m%Y %H%M")
            except ValueError:
                # parse fail => discard
                parse_fail_count += 1
                discard_writer.writerow(row)
                continue

            # date > CUTOFF => discard
            if ingreso_date > CUTOFF_DATE:
                date_pruned_count += 1
                discard_writer.writerow(row)
                continue

            # Check sex if we have that column
            if sex_idx is not None and sex_idx < len(row):
                sex_val = row[sex_idx].strip()
                if sex_val not in ["1", "2"]:
                    sex_pruned_count += 1
                    discard_writer.writerow(row)
                    continue

            # Check PrincipalDiagnosis if available
            if idx_principal_diag is not None and idx_principal_diag < len(row):
                principal_code = row[idx_principal_diag].strip()
                if not principal_code.startswith("F"):
                    diag_pruned_count += 1
                    discard_writer.writerow(row)
                    continue
            else:
                # If no principal diag column or it's empty, discard
                diag_pruned_count += 1
                discard_writer.writerow(row)
                continue

            # If we reach here, row is kept
            kept_count += 1
            writer.writerow(row)

    # Print summary stats
    discarded_count = parse_fail_count + date_pruned_count + diag_pruned_count + sex_pruned_count
    print("===== PRUNING REPORT =====")
    print(f"Total rows read:        {total_rows}")
    print(f"Rows kept (final):     {kept_count}")
    print(f"Rows discarded total:  {discarded_count}")
    print("Breakdown of discarded reasons:")
    print(f"  - parse_fail_count  : {parse_fail_count}")
    print(f"  - date_pruned_count : {date_pruned_count}")
    print(f"  - diag_pruned_count : {diag_pruned_count}")
    print(f"  - sex_pruned_count  : {sex_pruned_count}")

    print(f"\nDone! Filtered rows saved to: {OUTPUT_CSV}")
    print(f"Discarded rows saved to: {DISCARDED_CSV}")

if __name__ == "__main__":
    main()
