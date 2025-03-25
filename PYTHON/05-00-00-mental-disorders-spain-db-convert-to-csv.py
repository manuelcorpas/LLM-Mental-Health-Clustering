#!/usr/bin/env python3

import pyreadstat
import pandas as pd

def main():
    # 1. Define file paths
    input_sav = "DATA/Base trast mentales CIE 9-10 fusionados_1.sav"
    output_csv = "DATA/mental_disorders_admissions_spain.csv"

    # 2. Read the SPSS .sav file into a pandas DataFrame
    df, meta = pyreadstat.read_sav(input_sav)

    # 3. Rename mapping: Update to match actual Spanish/original column names in your file
    rename_mapping = {
        "Año": "Year",
        "ComunidadAutónoma": "AutonomousCommunity",
        "Provincia": "Province",
        "Fechadenacimiento": "DateOfBirth",
        "Sexo": "Sex",
        "CCAAResidencia": "AutonomousCommunityResidence",
        "ProvinciaResidencia": "ProvinceResidence",
        "FechadeIngreso": "AdmissionDate",
        "CircunstanciadeContacto": "ContactCircumstance",
        "FechadeFinContacto": "ContactEndDate",
        "FechadeIntervención": "InterventionDate",
        "TipoAlta": "DischargeType",
        "RégimenFinanciación": "FinancingRegime",
        "EstanciaDías": "DaysOfStay",
        "DX1": "PrincipalDiagnosis",
        "CapituloDiagnósticoPrincipal": "PrincipalDiagnosisChapter",
        "SecciónDiagnósticoPrincipal": "PrincipalDiagnosisSection",
        "CategoríaDiagnósticoPrincipal": "PrincipalDiagnosisCategory",
        "SubcategoríaDiagnósticoPrincipal": "PrincipalDiagnosisSubcategory",
        "DX2": "Diagnosis2",
        "DX3": "Diagnosis3",
        "DX4": "Diagnosis4",
        # Add more as necessary...
    }

    # 4. Rename columns in the DataFrame
    df_renamed = df.rename(columns=rename_mapping)

    # 5. Print broad summary info
    print(f"DataFrame loaded from: {input_sav}\n")
    print("--- Column Names After Renaming ---")
    print(df_renamed.columns.tolist())

    print("\n--- DataFrame Info ---")
    df_renamed.info()

    print("\n--- Overall Summary Statistics ---")
    print(df_renamed.describe(include="all"))

    # 6. Per-column distribution & missing values
    print("\n=== Per-Column Distribution and Missing Values ===")
    for col in df_renamed.columns:
        print(f"\n--- Column: '{col}' ---")
        # How many missing values?
        num_missing = df_renamed[col].isna().sum()
        print(f"Missing values: {num_missing} out of {len(df_renamed)}")

        # Check numeric vs. non-numeric type
        if pd.api.types.is_numeric_dtype(df_renamed[col]):
            # Numeric distribution (describe())
            print("Numeric summary:")
            print(df_renamed[col].describe())
        else:
            # Categorical / Object => Value counts
            print("Value counts (top categories):")
            print(df_renamed[col].value_counts(dropna=False).head(20))  # Show top 20 for brevity

    # 7. Save the final DataFrame as CSV
    df_renamed.to_csv(output_csv, index=False)
    print(f"\nSaved CSV to {output_csv}")

if __name__ == "__main__":
    main()

