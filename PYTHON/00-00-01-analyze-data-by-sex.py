#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency

def coerce_to_numeric(series):
    """
    Converts a mixed-type series to string, replaces commas with dots, and
    then tries to convert to float. Non-convertible values become NaN.
    """
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

def limit_diagnosis_to_top_n(df, n=20, diag_col="Diagnóstico Principal"):
    """
    Keeps only the top n most frequent diagnoses in diag_col, and lumps
    all others as 'OTHER'.
    """
    diag_counts = df[diag_col].value_counts()
    top_n_codes = set(diag_counts.head(n).index)  # top n frequent codes

    # Create a new column, e.g. 'Diagnóstico Principal (Limited)'
    limited_col = diag_col + " (Limited)"
    df[limited_col] = df[diag_col].apply(
        lambda x: x if x in top_n_codes else "OTHER"
    )
    return df, limited_col

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

    # C1) Diagnóstico Principal (codes)
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

def compare_categorical(df, column_name):
    """
    Runs a Chi-Square test comparing distribution of 'column_name' across males vs. females.
    """
    df_m = df[df["Sexo"] == 1]
    df_f = df[df["Sexo"] == 2]

    cats_m = set(df_m[column_name].dropna().unique())
    cats_f = set(df_f[column_name].dropna().unique())
    all_cats = sorted(cats_m.union(cats_f))

    table = []
    for cat in all_cats:
        male_count = (df_m[column_name] == cat).sum()
        female_count = (df_f[column_name] == cat).sum()
        table.append([male_count, female_count])
    table = np.array(table, dtype=int)

    if table.shape[0] < 2:
        print(f"\n[SKIP] For '{column_name}', only one category => no chi-square.")
        return

    chi2, pval, dof, expected = chi2_contingency(table)
    print(f"\n--- Chi-Square for '{column_name}' (Male vs. Female) ---")
    print(f"  #categories = {len(all_cats)}")
    print(f"  chi2={chi2:.3f}, p-value={pval:.3g}, dof={dof}")

def compare_male_female(df_full):
    """
    Performs simple statistical tests:
      1) Mann-Whitney U for numeric columns
      2) Chi-square for multiple categorical columns
         (including the top 20 Diagnóstico Principal).
    """

    from scipy.stats import mannwhitneyu, chi2_contingency

    print("\n=== Statistical Comparison: Males (Sexo=1) vs. Females (Sexo=2) ===")

    # 1) Numeric columns => Mann-Whitney
    numeric_cols = ["Estancia Días", "Coste APR"]
    df_m = df_full[df_full["Sexo"] == 1].copy()
    df_f = df_full[df_full["Sexo"] == 2].copy()

    for col in numeric_cols:
        if col in df_full.columns and pd.api.types.is_numeric_dtype(df_full[col]):
            male_vals = df_m[col].dropna()
            female_vals = df_f[col].dropna()
            if len(male_vals) > 1 and len(female_vals) > 1:
                stat, pval = mannwhitneyu(male_vals, female_vals, alternative="two-sided")
                print(f"\n--- Mann-Whitney U test for '{col}' ---")
                print(f"  Males n={len(male_vals)}, median={male_vals.median():.2f}")
                print(f"  Females n={len(female_vals)}, median={female_vals.median():.2f}")
                print(f"  U-statistic={stat:.1f}, p-value={pval:.3g}")
            else:
                print(f"\n[SKIP] Not enough data in column '{col}' for MW test.")
        else:
            print(f"\n[SKIP] Column '{col}' is not numeric or not found.")

    # 2) Chi-square on:
    #    - Año
    #    - Nivel Severidad APR
    #    - Diagnóstico Principal (Limited) => top 20
    cat_cols = ["Año", "Nivel Severidad APR"]
    for cat_col in cat_cols:
        if cat_col in df_full.columns:
            compare_categorical(df_full, cat_col)
        else:
            print(f"\n[SKIP] Column '{cat_col}' not found in DataFrame.")

    # For Diagnóstico Principal, limit to top 20 + "OTHER"
    if "Diagnóstico Principal" in df_full.columns:
        # Make a new column with the top 20 diagnoses + "OTHER"
        df_limited, limited_col = limit_diagnosis_to_top_n(df_full, n=20, diag_col="Diagnóstico Principal")
        compare_categorical(df_limited, limited_col)
    else:
        print("\n[SKIP] 'Diagnóstico Principal' column not found => no top-20 test.")

    print("\n=== End of M/F Comparisons ===")

def analyze_data_by_sex(csv_file_path, icd_file_path):
    """
    Reads the main CSV, merges with an ICD reference file, splits into sub-dataframes by sex
    for descriptive plots, and does Mann-Whitney and Chi-Square tests. For Diagnóstico Principal,
    it lumps all but the top 20 codes as "OTHER" before Chi-Square.
    """

    # 1) Read the main CSV
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8', delimiter=',', low_memory=False)
    except Exception as e:
        print(f"ERROR reading main CSV at '{csv_file_path}':\n{e}")
        return

    # 2) Read the ICD reference file
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

    # 3) Convert numeric columns
    if 'Estancia Días' in df.columns:
        df['Estancia Días'] = coerce_to_numeric(df['Estancia Días'])
    if 'Coste APR' in df.columns:
        df['Coste APR'] = coerce_to_numeric(df['Coste APR'])

    # 4) Clean & uppercase codes in main CSV
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

    # 5) Merge with ICD descriptors
    merged_df = df.merge(icd_ref, how='left', left_on='clean_code', right_on='clean_code')
    merged_df.rename(columns={'icd_description': 'Diagnóstico Principal (Desc)'}, inplace=True)
    merged_df.drop(columns=['icd_code', 'clean_code'], inplace=True, errors='ignore')

    if 'Diagnóstico Principal (Desc)' in merged_df.columns:
        merged_df['code_plus_desc'] = (
            merged_df['Diagnóstico Principal'].fillna('') + ' - '
            + merged_df['Diagnóstico Principal (Desc)'].fillna('NO MATCH')
        )

    # 6) Descriptive analysis/plots by sex
    for sex_val in [1, 2]:
        subdf = merged_df[merged_df['Sexo'] == sex_val].copy()
        analyze_subset(subdf, sex_val)

    # 7) Statistical comparisons
    compare_male_female(merged_df)

    print("\n=== DONE. Analyses by sex have been generated, plus statistical comparisons. ===")


if __name__ == "__main__":
    # Example usage:
    #   python3.11 PYTHON/00-00-01-analyze-data-by-sex.py

    csv_file_path = "DATA/RAECMBD_454_20241226-163036.csv"
    icd_file_path = "DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt"

    analyze_data_by_sex(csv_file_path, icd_file_path)
