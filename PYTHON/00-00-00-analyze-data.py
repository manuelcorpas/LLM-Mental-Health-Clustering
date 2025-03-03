#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_data(csv_file_path, icd_file_path):
    """
    Reads the main CSV (with Diagnóstico Principal codes) and an ICD reference file,
    merges them so we can translate the codes into short descriptions, and creates plots.

    Key changes:
    - We visualize the top 20 'Diagnóstico Principal (Code + Description)' with a
      horizontal bar chart (barh).
    - For 'Estancia Días', we produce two box plots: one with original data,
      and one capped at the 99th percentile to mitigate extreme outliers.
    - We provide descriptive stats for all variables we analyze:
      * Año (categorical)
      * Sexo (categorical)
      * Estancia Días (numeric)
      * Nivel Severidad APR (categorical)
      * Coste APR (numeric)
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
    print("[DEBUG] First 5 rows:\n", df.head())

    # 2) Manually parse the ICD file, splitting each line once on whitespace
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
    print("\n[DEBUG] ICD file read & split. Shape:", icd_ref.shape)
    print("[DEBUG] ICD columns:", icd_ref.columns.tolist())
    print("[DEBUG] First 5 rows of ICD ref:\n", icd_ref.head())

    # 3) Clean & uppercase codes
    # -------------------------------------------------------------------------
    df['Diagnóstico Principal'] = df['Diagnóstico Principal'].astype(str)

    def clean_csv_code(code):
        code = code.strip().upper()
        code = code.replace('.', '')
        code = code.replace('-', '')
        code = ''.join(code.split())
        return code

    df['clean_code'] = df['Diagnóstico Principal'].apply(clean_csv_code)

    icd_ref['icd_code'] = icd_ref['icd_code'].astype(str)

    def clean_icd_code(code):
        code = code.strip().upper()
        code = ''.join(code.split())
        return code

    icd_ref['clean_code'] = icd_ref['icd_code'].apply(clean_icd_code)

    print("\n[DEBUG] Cleaned codes from CSV (top 5):")
    print(df[['Diagnóstico Principal', 'clean_code']].head(5))
    print("[DEBUG] Cleaned codes from ICD ref (top 5):")
    print(icd_ref[['icd_code', 'clean_code']].head(5))

    # 4) Merge
    # -------------------------------------------------------------------------
    merged_df = df.merge(icd_ref, how='left', left_on='clean_code', right_on='clean_code')
    merged_df.rename(columns={'icd_description': 'Diagnóstico Principal (Desc)'}, inplace=True)
    merged_df.drop(columns=['icd_code', 'clean_code'], inplace=True, errors='ignore')

    print("\n[DEBUG] Merge done. Shape:", merged_df.shape)
    print("[DEBUG] Columns now:", merged_df.columns.tolist())
    print("[DEBUG] Sample merged rows:\n",
          merged_df[['Diagnóstico Principal', 'Diagnóstico Principal (Desc)']].head())

    # Overwrite df for final analysis
    df = merged_df

    # 5) Summaries & Plots
    # -------------------------------------------------------------------------
    # Optionally combine code + description
    if 'Diagnóstico Principal (Desc)' in df.columns:
        df['code_plus_desc'] = (
            df['Diagnóstico Principal'] + ' - '
            + df['Diagnóstico Principal (Desc)'].fillna('NO MATCH')
        )

    # A) Analyzing 'Año' (categorical)
    if 'Año' in df.columns:
        # Print descriptive stats for a categorical variable:
        print("\n=== Descriptive Stats for 'Año' (categorical) ===")
        year_counts = df['Año'].value_counts(dropna=False)
        print("Value Counts:\n", year_counts)
        print("Proportions (%):\n", (year_counts / len(df) * 100).round(2))

        # Plot distribution
        plt.figure()
        year_counts.sort_index().plot(kind='bar')
        plt.title("Distribution of 'Año'")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # B) Analyzing 'Sexo' (categorical)
    if 'Sexo' in df.columns:
        print("\n=== Descriptive Stats for 'Sexo' (categorical) ===")
        sexo_counts = df['Sexo'].value_counts(dropna=False)
        print("Value Counts:\n", sexo_counts)
        print("Proportions (%):\n", (sexo_counts / len(df) * 100).round(2))

        plt.figure()
        sexo_counts.plot(kind='bar')
        plt.title("Distribution of 'Sexo'")
        plt.xlabel("Sexo")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # C1) Diagnóstico Principal (codes)
    if 'Diagnóstico Principal' in df.columns:
        print("\n=== Descriptive Stats for 'Diagnóstico Principal' (categorical) ===")
        diag_counts = df['Diagnóstico Principal'].value_counts(dropna=False)
        print("Top 10 Value Counts:\n", diag_counts.head(10))
        print("Total unique codes:", diag_counts.size)

        plt.figure(figsize=(10, 5))
        diag_counts.head(10).plot(kind='bar')
        plt.title("Top 10 Diagnóstico Principal (Codes)")
        plt.xlabel("Code")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # C2) code_plus_desc top 20 (horizontal bars)
    if 'code_plus_desc' in df.columns:
        top20 = df['code_plus_desc'].value_counts(dropna=False).head(20)
        plt.figure(figsize=(10, 8))
        top20.sort_values().plot(kind='barh')
        plt.title("Top 20 Diagnóstico Principal (Code + Description)")
        plt.xlabel("Count")
        plt.ylabel("ICD10 Code + Short Description")
        plt.tight_layout()
        plt.show()

    # D) Estancia Días (numeric)
    if 'Estancia Días' in df.columns and pd.api.types.is_numeric_dtype(df['Estancia Días']):
        estancia_data = df['Estancia Días'].dropna()

        print("\n=== Descriptive Stats for 'Estancia Días' (numeric) ===")
        print(estancia_data.describe())

        # Histogram
        plt.figure()
        estancia_data.plot(kind='hist', bins=20)
        plt.title("Histogram of 'Estancia Días'")
        plt.xlabel("Days of Stay")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Box plot (original data)
        plt.figure()
        estancia_data.plot(kind='box')
        plt.title("Box Plot of 'Estancia Días' (Original)")
        plt.ylabel("Days of Stay")
        plt.tight_layout()
        plt.show()

        # Box plot with outlier capping at 99th percentile
        p99 = estancia_data.quantile(0.99)
        capped = np.where(estancia_data > p99, p99, estancia_data)

        plt.figure()
        pd.Series(capped).plot(kind='box')
        plt.title("Box Plot of 'Estancia Días' (Capped at 99th percentile)")
        plt.ylabel("Days of Stay")
        plt.tight_layout()
        plt.show()

    # E) Nivel Severidad APR (categorical)
    if 'Nivel Severidad APR' in df.columns:
        print("\n=== Descriptive Stats for 'Nivel Severidad APR' (categorical) ===")
        sev_counts = df['Nivel Severidad APR'].value_counts(dropna=False)
        print("Value Counts:\n", sev_counts)
        print("Proportions (%):\n", (sev_counts / len(df) * 100).round(2))

        plt.figure()
        sev_counts.sort_index().plot(kind='bar')
        plt.title("Distribution of 'Nivel Severidad APR'")
        plt.xlabel("Severity Level")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # F) Coste APR (numeric)
    if 'Coste APR' in df.columns and pd.api.types.is_numeric_dtype(df['Coste APR']):
        coste_data = df['Coste APR'].dropna()

        print("\n=== Descriptive Stats for 'Coste APR' (numeric) ===")
        print(coste_data.describe())

        # Histogram
        plt.figure()
        coste_data.plot(kind='hist', bins=20)
        plt.title("Histogram of 'Coste APR'")
        plt.xlabel("Cost")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Box plot
        plt.figure()
        coste_data.plot(kind='box')
        plt.title("Box Plot of 'Coste APR'")
        plt.ylabel("Cost")
        plt.tight_layout()
        plt.show()

    print("\n=== DONE. Descriptive stats for all analyzed columns are printed, outliers in 'Estancia Días' are capped for the second box plot. ===")


if __name__ == "__main__":
    # Example usage:
    #   python my_analysis.py /path/to/datos.csv /path/to/icd10cm-codes-April-2025.txt
    #
    csv_file_path = "DATA/RAECMBD_454_20241226-163036.csv"
    icd_file_path = "DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt"

    analyze_data(csv_file_path, icd_file_path)
