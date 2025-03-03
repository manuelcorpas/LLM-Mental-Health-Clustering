#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_subset(subdf, sex_val):
    """
    Produces descriptive statistics and plots for a given subset of rows (subdf),
    labeling each figure/output with the sex value (sex_val) for clarity.
    """

    if subdf.empty:
        print(f"\n[INFO] No records found for Sexo={sex_val}. Skipping.")
        return

    print(f"\n=== Analyzing subset with Sexo={sex_val} (n={len(subdf)}) ===")

    # A) Año (categorical)
    if 'Año' in subdf.columns:
        print(f"\nDescriptive Stats for 'Año' (Sexo={sex_val}):")
        year_counts = subdf['Año'].value_counts(dropna=False)
        print("Value Counts:\n", year_counts)
        print("Proportions (%):\n", (year_counts / len(subdf) * 100).round(2))

        plt.figure()
        year_counts.sort_index().plot(kind='bar')
        plt.title(f"Distribution of 'Año' (Sexo={sex_val})")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # B) Sexo (categorical)
    #   Here, we already know it's either 1 or 2, but let's show the counts anyway.
    if 'Sexo' in subdf.columns:
        print(f"\nDescriptive Stats for 'Sexo' (subset = {sex_val} only):")
        sexo_counts = subdf['Sexo'].value_counts(dropna=False)
        print("Value Counts:\n", sexo_counts)
        print("Proportions (%):\n", (sexo_counts / len(subdf) * 100).round(2))

        plt.figure()
        sexo_counts.plot(kind='bar')
        plt.title(f"Distribution of 'Sexo' (subset={sex_val})")
        plt.xlabel("Sexo")
        plt.ylabel("Count")
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()
        plt.show()

    # C1) Diagnóstico Principal (codes only)
    if 'Diagnóstico Principal' in subdf.columns:
        print(f"\nDescriptive Stats for 'Diagnóstico Principal' (Sexo={sex_val}):")
        diag_counts = subdf['Diagnóstico Principal'].value_counts(dropna=False)
        print("Top 10 Value Counts:\n", diag_counts.head(10))
        print("Total unique codes:", diag_counts.size)

        plt.figure(figsize=(10, 5))
        diag_counts.head(10).plot(kind='bar')
        plt.title(f"Top 10 Diagnóstico Principal (Codes) - Sexo={sex_val}")
        plt.xlabel("Code")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # C2) code_plus_desc top 20 (horizontal bars)
    if 'code_plus_desc' in subdf.columns:
        top20 = subdf['code_plus_desc'].value_counts(dropna=False).head(20)
        plt.figure(figsize=(10, 8))
        top20.sort_values().plot(kind='barh')
        plt.title(f"Top 20 Diagnóstico Principal (Code + Desc) - Sexo={sex_val}")
        plt.xlabel("Count")
        plt.ylabel("ICD10 Code + Short Description")
        plt.tight_layout()
        plt.show()

    # D) Estancia Días (numeric)
    if 'Estancia Días' in subdf.columns and pd.api.types.is_numeric_dtype(subdf['Estancia Días']):
        estancia_data = subdf['Estancia Días'].dropna()

        print(f"\nDescriptive Stats for 'Estancia Días' (Sexo={sex_val}):")
        print(estancia_data.describe())

        # Histogram
        plt.figure()
        estancia_data.plot(kind='hist', bins=20)
        plt.title(f"Histogram of 'Estancia Días' (Sexo={sex_val})")
        plt.xlabel("Days of Stay")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Box plot (original data)
        plt.figure()
        estancia_data.plot(kind='box')
        plt.title(f"Box Plot of 'Estancia Días' (Original) - Sexo={sex_val}")
        plt.ylabel("Days of Stay")
        plt.tight_layout()
        plt.show()

        # Box plot with outlier capping at 99th percentile
        p99 = estancia_data.quantile(0.99)
        capped = np.where(estancia_data > p99, p99, estancia_data)

        plt.figure()
        pd.Series(capped).plot(kind='box')
        plt.title(f"Box Plot of 'Estancia Días' (99th % cap) - Sexo={sex_val}")
        plt.ylabel("Days of Stay")
        plt.tight_layout()
        plt.show()

    # E) Nivel Severidad APR (categorical)
    if 'Nivel Severidad APR' in subdf.columns:
        print(f"\nDescriptive Stats for 'Nivel Severidad APR' (Sexo={sex_val}):")
        sev_counts = subdf['Nivel Severidad APR'].value_counts(dropna=False)
        print("Value Counts:\n", sev_counts)
        print("Proportions (%):\n", (sev_counts / len(subdf) * 100).round(2))

        plt.figure()
        sev_counts.sort_index().plot(kind='bar')
        plt.title(f"Distribution of 'Nivel Severidad APR' (Sexo={sex_val})")
        plt.xlabel("Severity Level")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # F) Coste APR (numeric)
    if 'Coste APR' in subdf.columns and pd.api.types.is_numeric_dtype(subdf['Coste APR']):
        coste_data = subdf['Coste APR'].dropna()

        print(f"\nDescriptive Stats for 'Coste APR' (Sexo={sex_val}):")
        print(coste_data.describe())

        # Histogram
        plt.figure()
        coste_data.plot(kind='hist', bins=20)
        plt.title(f"Histogram of 'Coste APR' (Sexo={sex_val})")
        plt.xlabel("Cost")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Box plot
        plt.figure()
        coste_data.plot(kind='box')
        plt.title(f"Box Plot of 'Coste APR' (Sexo={sex_val})")
        plt.ylabel("Cost")
        plt.tight_layout()
        plt.show()

    print(f"\n=== Finished analysis for Sexo={sex_val}. ===")


def analyze_data_by_sex(csv_file_path, icd_file_path):
    """
    Reads the main CSV, merges with an ICD reference file, then splits
    into two sub-dataframes by sex (Sexo=1, Sexo=2) and runs descriptive
    statistics/plots for each subset.
    """

    # 1) Read the main CSV
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8', delimiter=',')
    except Exception as e:
        print(f"ERROR reading main CSV at '{csv_file_path}':\n{e}")
        return

    print("[DEBUG] Loaded main CSV. Shape:", df.shape)
    print("[DEBUG] CSV columns:", df.columns.tolist())

    # 2) Read the ICD reference file and parse
    # -------------------------------------------------------------------------
    icd_data = []
    try:
        with open(icd_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    code, desc = parts
                elif len(parts) == 1:
                    code = parts[0]
                    desc = ""
                else:
                    code = ""
                    desc = ""
                icd_data.append((code, desc))
    except Exception as e:
        print(f"ERROR reading ICD file at '{icd_file_path}':\n{e}")
        return

    icd_ref = pd.DataFrame(icd_data, columns=['icd_code', 'icd_description'])

    # 3) Clean & uppercase codes in main CSV
    # -------------------------------------------------------------------------
    df['Diagnóstico Principal'] = df['Diagnóstico Principal'].astype(str)

    def clean_csv_code(code):
        code = code.strip().upper()
        code = code.replace('.', '')
        code = code.replace('-', '')
        code = ''.join(code.split())
        return code

    df['clean_code'] = df['Diagnóstico Principal'].apply(clean_csv_code)

    def clean_icd_code(code):
        code = code.strip().upper()
        code = ''.join(code.split())
        return code

    icd_ref['icd_code'] = icd_ref['icd_code'].astype(str)
    icd_ref['clean_code'] = icd_ref['icd_code'].apply(clean_icd_code)

    # 4) Merge
    # -------------------------------------------------------------------------
    merged_df = df.merge(icd_ref, how='left', left_on='clean_code', right_on='clean_code')
    merged_df.rename(columns={'icd_description': 'Diagnóstico Principal (Desc)'}, inplace=True)
    merged_df.drop(columns=['icd_code', 'clean_code'], inplace=True, errors='ignore')

    # Optionally combine code + description
    if 'Diagnóstico Principal (Desc)' in merged_df.columns:
        merged_df['code_plus_desc'] = (
            merged_df['Diagnóstico Principal'].fillna('') + ' - '
            + merged_df['Diagnóstico Principal (Desc)'].fillna('NO MATCH')
        )

    # 5) Now create subsets by sex and run the analysis for each
    # -------------------------------------------------------------------------
    for sex_val in [1, 2]:
        subdf = merged_df[merged_df['Sexo'] == sex_val].copy()
        analyze_subset(subdf, sex_val)

    print("\n=== DONE. Analyses by sex have been generated. ===")


if __name__ == "__main__":
    # Example usage:
    #   python3.11 PYTHON/00-00-01-analyze-data-by-sex.py 

    csv_file_path = "DATA/RAECMBD_454_20241226-163036.csv"
    icd_file_path = "DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt"

    analyze_data_by_sex(csv_file_path, icd_file_path)

