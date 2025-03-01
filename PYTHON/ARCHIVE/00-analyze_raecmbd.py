#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_raecmbd(csv_filepath):
    """
    Reads a RAE-CMBD CSV file, cleans/transforms the data,
    and performs a series of analyses:
      1. Basic info and summary stats
      2. Grouped counts (by year, diagnosis, etc.)
      3. Example pivot table
      4. Example visualizations
    Adjust column names, date parsing, or numeric conversions as needed.
    """

    print(f"\n--- Loading data from: {csv_filepath} ---")
    # Load the data
    try:
        df = pd.read_csv(csv_filepath, sep=',', encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)

    # Show a quick peek of the data
    print("\n--- DataFrame Info ---")
    print(df.info())
    print("\n--- First 5 rows ---")
    print(df.head())

    # Example: Convert date columns to datetime if present
    date_cols = ['Fecha de nacimiento', 'Fecha de Ingreso', 'Fecha de Fin Contacto']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Example: Convert numeric columns if present
    numeric_cols = ['Año', 'Edad', 'Estancia Días', 'Coste APR']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for missing values
    print("\n--- Missing Values per Column ---")
    print(df.isnull().sum())

    # Basic descriptive stats
    print("\n--- Descriptive Statistics ---")
    # 'include="all"' provides stats for numeric and object columns
    print(df.describe(include='all'))

    # Example 1: Count rows per 'Año' (year)
    if 'Año' in df.columns:
        year_counts = df.groupby('Año').size().reset_index(name='count')
        print("\n--- Records by Año (Year) ---")
        print(year_counts)

        # Visualization: bar chart by year
        plt.figure(figsize=(8, 5))
        sns.barplot(data=year_counts, x='Año', y='count')
        plt.title('Count of Records by Year')
        plt.tight_layout()
        plt.savefig('plot_records_by_year.png')
        plt.close()
        print("-> Saved bar chart: plot_records_by_year.png")

    # Example 2: Distribution of Diagnóstico Principal
    if 'Diagnóstico Principal' in df.columns:
        diag_counts = (
            df
            .groupby('Diagnóstico Principal')
            .size()
            .reset_index(name='count')
            .sort_values(by='count', ascending=False)
        )
        print("\n--- Top 20 Diagnóstico Principal Counts ---")
        print(diag_counts.head(20))

    # Example 3: Summaries by Region and Year
    if 'Comunidad Autónoma' in df.columns and 'Año' in df.columns:
        region_year_counts = (
            df
            .groupby(['Comunidad Autónoma', 'Año'])
            .size()
            .reset_index(name='count')
            .sort_values(by=['Comunidad Autónoma', 'Año'])
        )
        print("\n--- Records by Comunidad Autónoma and Año (Year) ---")
        print(region_year_counts.head(30))  # Show only first 30 for brevity

    # Example 4: Length of Stay Distribution
    if 'Estancia Días' in df.columns:
        print("\n--- Distribution of Estancia Días ---")
        print(df['Estancia Días'].describe())

        # Visualization: histogram of Estancia Días
        plt.figure(figsize=(8, 5))
        sns.histplot(df['Estancia Días'].dropna(), bins=30, kde=False)
        plt.title('Distribution of Estancia Días')
        plt.xlabel('Days of Stay')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('hist_estancia_dias.png')
        plt.close()
        print("-> Saved histogram: hist_estancia_dias.png")

    # Example 5: Pivot Table
    # Summarize Estancia Días by Año and Diagnóstico Principal
    if 'Año' in df.columns and 'Diagnóstico Principal' in df.columns and 'Estancia Días' in df.columns:
        pivot = pd.pivot_table(
            df,
            values='Estancia Días',
            index='Diagnóstico Principal',
            columns='Año',
            aggfunc='mean'
        )
        print("\n--- Pivot Table: Mean Estancia Días by Diagnóstico Principal and Año ---")
        print(pivot.head(10))

    print("\nAnalysis completed!\n")


def main():
    csv_filepath = "DATA/RAECMBD_454_20241226-163036.csv"
    analyze_raecmbd(csv_filepath)    

if __name__ == "__main__":
    main()

