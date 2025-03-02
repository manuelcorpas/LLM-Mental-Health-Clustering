# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
analyze_clustered_patients.py

Script to analyze the CSV 'RESULTS/clustered_patients.csv' output
from your prior clustering pipeline.

Key functionalities:
- Loads the final cluster assignments.
- Prints basic stats (number of patients, distribution across clusters).
- Shows the top 5 frequent codes (and optional descriptors) in each cluster.
- Computes c-DF–IPF for each cluster to highlight overrepresented codes.
- Reports average length of stay (Estancia Días) per cluster.

Usage example:
  python analyze_clustered_patients.py --csv_file RESULTS/clustered_patients.csv \
    --desc_file Data/icd10cm-codes-April-2025.txt
"""

import argparse
import math
import pandas as pd

def load_code_descriptors(desc_file):
    """
    Reads a descriptor file with lines like:
      F11.2    Opioid dependence
      F14.9    Cocaine use, unspecified
    Returns a dict: code -> short description
    """
    code2desc = {}
    with open(desc_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            code = parts[0].strip()
            desc = parts[1].strip()
            code2desc[code] = desc
    return code2desc

def compute_cdfipf(df, diag_col="Diagnóstico Principal", cluster_col="cluster"):
    """
    c-DF–IPF procedure:
    1) Parse DIAGNOSTICO_PRINCIPAL as comma-separated codes.
    2) For each cluster, compute a weight that measures how overrepresented a code is
       in that cluster vs. the entire dataset.
       w_{c, code} = (sum of IPF for that code in cluster c) / cluster_size
       IPF(code) = ln( total_patients / number_of_patients_who_have_code )
    """
    all_codes = set()
    pid2codes = {}

    # Gather all codes for each patient
    for idx, row in df.iterrows():
        pid = row["ID"]
        diag_str = row.get(diag_col, "")
        codes = set(x.strip() for x in diag_str.split(",") if x.strip())
        if not codes:
            codes = {"NOCODE"}
        pid2codes[pid] = codes
        for c in codes:
            all_codes.add(c)

    # cluster -> list of pids
    cluster2pids = {}
    for idx, row in df.iterrows():
        c = row[cluster_col]
        pid = row["ID"]
        cluster2pids.setdefault(c, []).append(pid)

    n_patients = df["ID"].nunique()

    # code -> how many patients have it
    code2count = {c: 0 for c in all_codes}
    for pid, codeset in pid2codes.items():
        for cd in codeset:
            code2count[cd] += 1

    # code -> IPF
    # IPF = ln(n_patients / code2count[cd])  but handle zero
    code2ipf = {}
    for c in all_codes:
        nd = code2count[c]
        if nd == 0:
            code2ipf[c] = 0
        else:
            code2ipf[c] = math.log(n_patients / nd)

    # Now compute cluster-level c-DF–IPF
    cluster2dfipf = {}
    for c, pids in cluster2pids.items():
        sums = {cd: 0.0 for cd in all_codes}
        for pid in pids:
            for cd in pid2codes[pid]:
                sums[cd] += code2ipf[cd]
        csize = len(pids)
        for cd in sums:
            sums[cd] = sums[cd]/csize if csize>0 else 0
        cluster2dfipf[c] = sums

    return cluster2dfipf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True,
                        help="Path to the CSV file, e.g. 'RESULTS/clustered_patients.csv'.")
    parser.add_argument("--desc_file", default=None,
                        help="Optional ICD-10 code descriptor file for printing short text.")
    parser.add_argument("--diag_col", default="Diagnóstico Principal",
                        help="Column name for the comma-separated diagnosis codes.")
    parser.add_argument("--stay_col", default="Estancia Días",
                        help="Numeric column for length of stay.")
    parser.add_argument("--cluster_col", default="cluster",
                        help="Column with assigned cluster labels.")
    args = parser.parse_args()

    # 1) Load data
    df = pd.read_csv(args.csv_file, low_memory=False)
    if "ID" not in df.columns:
        # If the data doesn't have an ID col, create one from index
        df["ID"] = df.index + 1

    # Convert length-of-stay to numeric
    df[args.stay_col] = pd.to_numeric(df[args.stay_col], errors="coerce")

    print(f"Loaded {len(df)} rows from '{args.csv_file}'.")
    if args.cluster_col not in df.columns:
        raise ValueError(f"Column '{args.cluster_col}' not found. You must have 'cluster' col in your CSV.")

    num_patients = len(df)
    unique_clusters = sorted(df[args.cluster_col].unique())
    num_clusters = len(unique_clusters)
    print(f"\nTotal patients: {num_patients}")
    print(f"Total clusters: {num_clusters}")

    cluster_counts = df[args.cluster_col].value_counts().sort_index()
    print("\nPatients per cluster:")
    print(cluster_counts)

    # 2) (Optional) Load descriptors
    code2desc = {}
    if args.desc_file:
        code2desc = load_code_descriptors(args.desc_file)
        print(f"\nLoaded {len(code2desc)} code descriptors from '{args.desc_file}'.")

    # 3) Show top 5 most frequent codes per cluster
    print("\nTop 5 most frequent codes per cluster:")
    for c in unique_clusters:
        cluster_df = df[df[args.cluster_col] == c]
        diag_list = []
        for idx, row in cluster_df.iterrows():
            diag_str = row.get(args.diag_col, "")
            codes = [x.strip() for x in diag_str.split(",") if x.strip()]
            diag_list.extend(codes)
        diag_series = pd.Series(diag_list)
        top5 = diag_series.value_counts().head(5)
        print(f"\nCluster {c} [N={len(cluster_df)}]:")
        for code, freq in top5.items():
            if code2desc and code in code2desc:
                desc = code2desc[code]
                print(f"  {code} -> {desc} (count={freq})")
            else:
                print(f"  {code} (count={freq})")

    # 4) Average length of stay per cluster
    avg_stay = df.groupby(args.cluster_col)[args.stay_col].mean().round(2)
    print("\nAverage length of stay (days) per cluster:")
    for c in unique_clusters:
        val = avg_stay.loc[c]
        print(f"  Cluster {c}: {val} days")

    # 5) c-DF–IPF analysis
    print("\n=== c-DF–IPF analysis ===")
    cluster2dfipf = compute_cdfipf(df, diag_col=args.diag_col, cluster_col=args.cluster_col)

    for c in unique_clusters:
        code_map = cluster2dfipf[c]
        # sort by descending weight
        sorted_codes = sorted(code_map.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_codes[:5]
        print(f"\nCluster {c} top 5 codes by c-DF–IPF:")
        for cd, val in top5:
            if code2desc and cd in code2desc:
                desc = code2desc[cd]
                print(f"  {cd} -> {desc}: {val:.4f}")
            else:
                print(f"  {cd}: {val:.4f}")

    print("\nDone. You now have cluster-level stats, top codes, length-of-stay data, and c-DF–IPF results.")


if __name__ == "__main__":
    main()

