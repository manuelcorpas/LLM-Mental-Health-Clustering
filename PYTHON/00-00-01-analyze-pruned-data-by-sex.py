#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency

def coerce_to_numeric(series):
    """
    Converts a mixed-type series to string, replaces commas with dots,
    then tries to convert to float. Non-convertible values become NaN.
    """
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

def limit_diagnosis_to_top_n(df, n=20, diag_col="PrincipalDiagnosis"):
    """
    Keeps only the top n most frequent diagnoses in diag_col, and lumps
    all others as 'OTHER'.
    """
    diag_counts = df[diag_col].value_counts()
    top_n_codes = set(diag_counts.head(n).index)
    limited_col = diag_col + " (Limited)"
    df[limited_col] = df[diag_col].apply(
        lambda x: x if x in top_n_codes else "OTHER"
    )
    return df, limited_col

def compare_categorical(df, column_name):
    """
    Runs a Chi-Square test comparing distribution of 'column_name' across
    males vs. females (Sex=1 vs. Sex=2).
    """
    df_m = df[df["Sex"] == 1]
    df_f = df[df["Sex"] == 2]

    cats_m = set(df_m[column_name].dropna().unique())
    cats_f = set(df_f[column_name].dropna().unique())
    all_cats = sorted(cats_m.union(cats_f))

    # If there's only 1 category total, skip
    if len(all_cats) < 2:
        print(f"[SKIP] For '{column_name}', only one category => no chi-square.")
        return

    # Build a contingency table
    table = []
    for cat in all_cats:
        male_count = (df_m[column_name] == cat).sum()
        female_count = (df_f[column_name] == cat).sum()
        table.append([male_count, female_count])
    table = np.array(table, dtype=int)

    chi2, pval, dof, expected = chi2_contingency(table)
    print(f"\n--- Chi-Square for '{column_name}' (Male vs. Female) ---")
    print(f"  #categories = {len(all_cats)}")
    print(f"  chi2={chi2:.3f}, p-value={pval:.3g}, dof={dof}")

def compare_male_female(df_full):
    """
    Performs simple statistical tests:
      1) Mann-Whitney U for numeric columns
      2) Chi-square for multiple categorical columns
         (including top 20 PrincipalDiagnosis).
    """
    print("\n=== Statistical Comparison: Males (Sex=1) vs. Females (Sex=2) ===")

    # 1) Numeric columns => Mann-Whitney
    numeric_cols = ["DaysOfStay", "APRCost"]
    df_m = df_full[df_full["Sex"] == 1].copy()
    df_f = df_full[df_full["Sex"] == 2].copy()

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
                print(f"[SKIP] Not enough data in column '{col}' for MW test.")
        else:
            print(f"[SKIP] Column '{col}' is not numeric or not found.")

    # 2) Chi-square on these categorical columns
    cat_cols = ["Year", "APRSeverityLevel"]
    for cat_col in cat_cols:
        if cat_col in df_full.columns:
            compare_categorical(df_full, cat_col)
        else:
            print(f"[SKIP] Column '{cat_col}' not found.")

    # Also do top-20 analysis of PrincipalDiagnosis
    if "PrincipalDiagnosis" in df_full.columns:
        df_limited, limited_col = limit_diagnosis_to_top_n(df_full, n=20, diag_col="PrincipalDiagnosis")
        compare_categorical(df_limited, limited_col)
    else:
        print("[SKIP] 'PrincipalDiagnosis' column not found => no top-20 test.")

    print("\n=== End of M/F Comparisons ===")

###############################################################################
# NEW: Descriptive analysis for each sex, as in your old examples
###############################################################################
def analyze_subset(subdf, sex_val):
    """
    Produces descriptive statistics for a given subset (subdf),
    labeling outputs with the sex value (sex_val). Format matches
    the older code's style (counts, proportions, top codes, describe).
    """

    if subdf.empty:
        print(f"\n[INFO] No records found for Sex={sex_val}. Skipping.")
        return

    print(f"\n=== Analyzing subset with Sex={sex_val} (n={len(subdf)}) ===")

    # A) Year (categorical)
    if 'Year' in subdf.columns:
        print(f"\nDescriptive Stats for 'Year' (Sex={sex_val}):")
        year_counts = subdf['Year'].value_counts(dropna=False)
        print("Value Counts:\n", year_counts)
        print("Proportions (%):\n", ((year_counts / len(subdf)) * 100).round(2))

    # B) Sex (categorical)
    if 'Sex' in subdf.columns:
        print(f"\nDescriptive Stats for 'Sex' (subset = {sex_val} only):")
        sex_counts = subdf['Sex'].value_counts(dropna=False)
        print("Value Counts:\n", sex_counts)
        print("Proportions (%):\n", ((sex_counts / len(subdf)) * 100).round(2))

    # C) PrincipalDiagnosis
    if 'PrincipalDiagnosis' in subdf.columns:
        print(f"\nDescriptive Stats for 'PrincipalDiagnosis' (Sex={sex_val}):")
        diag_counts = subdf['PrincipalDiagnosis'].value_counts(dropna=False)
        print("Top 10 Value Counts:\n", diag_counts.head(10))
        print("Total unique codes:", diag_counts.size)

    # D) DaysOfStay (numeric)
    if 'DaysOfStay' in subdf.columns and pd.api.types.is_numeric_dtype(subdf['DaysOfStay']):
        estancia_data = subdf['DaysOfStay'].dropna()
        if not estancia_data.empty:
            print(f"\nDescriptive Stats for 'DaysOfStay' (Sex={sex_val}):")
            print(estancia_data.describe())

    # E) APRSeverityLevel (categorical)
    if 'APRSeverityLevel' in subdf.columns:
        sev_counts = subdf['APRSeverityLevel'].value_counts(dropna=False)
        print(f"\nDescriptive Stats for 'APRSeverityLevel' (Sex={sex_val}):")
        print("Value Counts:\n", sev_counts)
        print("Proportions (%):\n", ((sev_counts / len(subdf)) * 100).round(2))

    # F) APRCost (numeric)
    if 'APRCost' in subdf.columns and pd.api.types.is_numeric_dtype(subdf['APRCost']):
        coste_data = subdf['APRCost'].dropna()
        if not coste_data.empty:
            print(f"\nDescriptive Stats for 'APRCost' (Sex={sex_val}):")
            print(coste_data.describe())

    print(f"\n=== Finished analysis for Sex={sex_val}. ===")

###############################################################################

def analyze_combined(df):
    """
    Single function to produce combined male/female figures
    but rename '1' -> 'Male' and '2' -> 'Female' in legends.
    Also caps DaysOfStay at the 99th percentile
    to reduce outlier skew.
    """

    print("\n=== Analyzing data combined (male/female in one figure) ===")
    if df.empty:
        print("[WARN] DataFrame is empty. Nothing to plot.")
        return

    # ------------------------------------------------------
    # Figure 1: Year distribution by Sex (side-by-side bars)
    # ------------------------------------------------------
    if "Year" in df.columns and "Sex" in df.columns:
        grouped = df.groupby(["Year", "Sex"]).size().unstack(fill_value=0)
        # rename columns from [1,2] => ['Male','Female']
        grouped.rename(columns={1: "Male", 2: "Female"}, inplace=True)

        print("\nYear distribution (counts) by Sex:\n", grouped)

        fig1, ax1 = plt.subplots()
        grouped.plot(kind="bar", ax=ax1)
        ax1.set_title("Figure 1: Distribution of 'Year' by Sex (Combined)")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------
    # Figure 2: Top 10 PrincipalDiagnosis
    # ------------------------------------------------------
    if "PrincipalDiagnosis" in df.columns and "Sex" in df.columns:
        top10codes = df["PrincipalDiagnosis"].value_counts().head(10).index
        sub = df[df["PrincipalDiagnosis"].isin(top10codes)]
        diag_counts = sub.groupby(["PrincipalDiagnosis", "Sex"]).size().unstack(fill_value=0)
        diag_counts.rename(columns={1: "Male", 2: "Female"}, inplace=True)

        print("\nTop 10 PrincipalDiagnosis by Sex:\n", diag_counts)

        fig2, ax2 = plt.subplots(figsize=(10,4))
        diag_counts.plot(kind="bar", ax=ax2)
        ax2.set_title("Figure 2: Top 10 PrincipalDiagnosis by Sex (Combined)")
        ax2.set_xlabel("Diagnosis Code")
        ax2.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------
    # DaysOfStay: Boxplot & Hist (with outlier capping)
    # ------------------------------------------------------
    if "DaysOfStay" in df.columns and pd.api.types.is_numeric_dtype(df["DaysOfStay"]):
        box_df = df.dropna(subset=["DaysOfStay", "Sex"]).copy()
        if not box_df.empty:
            # Cap at 99th percentile to limit outliers
            p99 = box_df["DaysOfStay"].quantile(0.99)
            box_df["DaysOfStayCapped"] = np.minimum(box_df["DaysOfStay"], p99)

            # We'll do a boxplot by Sex
            # But we want '1' => 'Male' and '2' => 'Female' on x-axis
            box_df["SexLabel"] = box_df["Sex"].map({1: "Male", 2: "Female"})

            fig3, ax3 = plt.subplots()
            box_df.boxplot(column="DaysOfStayCapped", by="SexLabel", ax=ax3)
            ax3.set_title("Figure 3: DaysOfStay (99% cap) by Sex (Combined)")
            plt.suptitle("")  # remove default Pandas subtitle
            ax3.set_xlabel("Sex")
            ax3.set_ylabel("Days of Stay (capped)")
            plt.tight_layout()
            plt.show()

            # Overlaid hist (capped)
            df_m = box_df[box_df["Sex"] == 1]["DaysOfStayCapped"]
            df_f = box_df[box_df["Sex"] == 2]["DaysOfStayCapped"]

            fig4, ax4 = plt.subplots()
            ax4.hist(df_m, bins=20, alpha=0.5, label="Male")
            ax4.hist(df_f, bins=20, alpha=0.5, label="Female")
            ax4.set_title("Figure 4: Histogram of DaysOfStay (99% cap) by Sex (Combined)")
            ax4.set_xlabel("Days of Stay (capped)")
            ax4.set_ylabel("Frequency")
            ax4.legend()
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------
    # APRSeverityLevel by Sex
    # ------------------------------------------------------
    if "APRSeverityLevel" in df.columns and "Sex" in df.columns:
        sev_grouped = df.groupby(["APRSeverityLevel", "Sex"]).size().unstack(fill_value=0)
        sev_grouped.rename(columns={1: "Male", 2: "Female"}, inplace=True)

        print("\nAPRSeverityLevel distribution by Sex:\n", sev_grouped)

        fig5, ax5 = plt.subplots()
        sev_grouped.plot(kind="bar", ax=ax5)
        ax5.set_title("Figure 5: Distribution of APRSeverityLevel by Sex (Combined)")
        ax5.set_xlabel("Severity Level")
        ax5.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------
    # APRCost boxplot by Sex
    # ------------------------------------------------------
    if "APRCost" in df.columns and pd.api.types.is_numeric_dtype(df["APRCost"]):
        cost_df = df.dropna(subset=["APRCost", "Sex"]).copy()
        if not cost_df.empty:
            # Optionally cap cost outliers
            p99_cost = cost_df["APRCost"].quantile(0.99)
            cost_df["APRCostCapped"] = np.minimum(cost_df["APRCost"], p99_cost)
            cost_df["SexLabel"] = cost_df["Sex"].map({1: "Male", 2: "Female"})

            fig6, ax6 = plt.subplots()
            cost_df.boxplot(column="APRCostCapped", by="SexLabel", ax=ax6)
            ax6.set_title("Figure 6: APRCost (99% cap) by Sex (Combined)")
            plt.suptitle("")
            ax6.set_xlabel("Sex")
            ax6.set_ylabel("Cost (capped)")
            plt.tight_layout()
            plt.show()

    print("\n=== Finished combined analysis. ===")

def analyze_data_by_sex(csv_file_path, icd_file_path):
    """
    1) Reads the pruned CSV with English headers
    2) Merges with an ICD reference file
    3) For each sex (1,2), prints a descriptive analysis (like old code).
    4) Produces combined M/F plots (with outlier capping for DaysOfStay, APRCost)
       using 'Male' and 'Female' in the legend
    5) Runs Mann-Whitney + Chi-Square tests
    """
    # 1) Read the pruned CSV
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8', delimiter=',', low_memory=False)
    except Exception as e:
        print(f"ERROR reading CSV '{csv_file_path}': {e}")
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
        print(f"ERROR reading ICD file '{icd_file_path}': {e}")
        return

    icd_ref = pd.DataFrame(icd_data, columns=['icd_code', 'icd_description'])

    # 3) Convert numeric columns
    if "DaysOfStay" in df.columns:
        df["DaysOfStay"] = coerce_to_numeric(df["DaysOfStay"])
    if "APRCost" in df.columns:
        df["APRCost"] = coerce_to_numeric(df["APRCost"])

    # 4) Clean & uppercase principal diagnosis codes
    if "PrincipalDiagnosis" not in df.columns:
        print("[ERROR] 'PrincipalDiagnosis' not found. Cannot continue.")
        return

    df["PrincipalDiagnosis"] = df["PrincipalDiagnosis"].astype(str)

    def clean_csv_code(code):
        code = code.strip().upper()
        code = code.replace('.', '').replace('-', '')
        return ''.join(code.split())

    df["clean_code"] = df["PrincipalDiagnosis"].apply(clean_csv_code)

    def clean_icd_code(code):
        code = code.strip().upper()
        return ''.join(code.split())

    icd_ref["icd_code"] = icd_ref["icd_code"].astype(str)
    icd_ref["clean_code"] = icd_ref["icd_code"].apply(clean_icd_code)

    # 5) Merge with ICD descriptors
    merged_df = df.merge(icd_ref, how="left", on="clean_code")
    merged_df.rename(columns={"icd_description": "PrincipalDiagnosisDesc"}, inplace=True)
    merged_df.drop(columns=["icd_code","clean_code"], errors="ignore", inplace=True)
    generate_supplementary_table_1(merged_df)


    # Create code+desc column if desired
    if "PrincipalDiagnosisDesc" in merged_df.columns:
        merged_df["code_plus_desc"] = (
            merged_df["PrincipalDiagnosis"].fillna("") + " - "
            + merged_df["PrincipalDiagnosisDesc"].fillna("NO MATCH")
        )

    # 6) Per-sex descriptive analysis (like old code)
    for sex_val in [1, 2]:
        subdf = merged_df[merged_df["Sex"] == sex_val].copy()
        analyze_subset(subdf, sex_val)

    # 7) Combined M/F Plots
    analyze_combined(merged_df)

    # 8) Mann-Whitney + Chi-Square
    compare_male_female(merged_df)

    print("\n=== DONE. Combined M/F figures + stats generated. ===")

def generate_supplementary_table_1(df):
    """
    Generate Supplementary Table 1 for ICD-10 F11-F19 SUD codes,
    stratified by sex with chi-square p-values, based on presence
    in any diagnostic field (PrincipalDiagnosis through Diagnosis20).
    Includes % of total males and % of total females affected per row.
    """
    from scipy.stats import chi2_contingency
    from tabulate import tabulate
    import os

    # Define SUD root codes and labels
    sud_groups = {
        "F11": "Opioids",
        "F12": "Cannabis",
        "F13": "Sedatives, hypnotics, anxiolytics (benzodiazepines)",
        "F14": "Cocaine",
        "F15": "Amphetamines, psychostimulants",
        "F16": "Hallucinogens",
        "F18": "Other drugs",
        "F19": "Combos opioids + non-opioid + others"
    }

    # Capture all diagnosis fields (PrincipalDiagnosis through Diagnosis20)
    diag_cols = [col for col in df.columns if col.startswith("PrincipalDiagnosis") or col.startswith("Diagnosis")]
    records = []

    total_male = len(df[df["Sex"] == 1])
    total_female = len(df[df["Sex"] == 2])

    for idx, (code_prefix, label) in enumerate(sud_groups.items(), start=1):
        matches = df[diag_cols].apply(
            lambda row: any(str(val).startswith(code_prefix) for val in row if pd.notna(val)), axis=1
        )
        subset = df[matches]

        total = len(subset)
        male = len(subset[subset["Sex"] == 1])
        female = len(subset[subset["Sex"] == 2])

        contingency = [[male, female], [total_male - male, total_female - female]]
        chi2, pval, _, _ = chi2_contingency(contingency)

        pct_male = round((male / total_male) * 100, 1) if total_male > 0 else 0.0
        pct_female = round((female / total_female) * 100, 1) if total_female > 0 else 0.0

        records.append({
            "N": idx,
            "SUD condition": label,
            "ICD-10": code_prefix,
            "Total diagnoses": total,
            "Rate (%)": round((total / len(df)) * 100, 1),
            "Male": male,
            "% of Males": pct_male,
            "Female": female,
            "% of Females": pct_female,
            "P value": f"{pval:.4f}"
        })

    sud_df = pd.DataFrame(records)
    sud_df = sud_df.sort_values(by="Rate (%)", ascending=False)

    print("\n=== Supplementary Table 1: Diagnoses in adolescents with SUD in Spain ===")
    print(tabulate(sud_df, headers="keys", tablefmt="github", showindex=False))

    os.makedirs("RESULTS", exist_ok=True)
    sud_df.to_csv("RESULTS/supplementary_table_1_sud_by_sex.csv", index=False)
    print("✅ Supplementary Table 1 saved to RESULTS/supplementary_table_1_sud_by_sex.csv")

    return sud_df





# We haven't removed any code from your script, only added "analyze_subset"
# and calls to it for the older-style descriptive stats.

if __name__ == "__main__":
    # Example usage
    csv_file_path = "DATA/RAECMBD_454_20241226-163036_pruned.csv"
    icd_file_path = "DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt"

    analyze_data_by_sex(csv_file_path, icd_file_path)
