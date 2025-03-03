#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze clusters from a CSV (e.g. 'clustered_patients-co-occ-sex.csv') that contains:
  - Diagnoses (e.g., 'Diagnostico', 'Diagnostico 2', ...)
  - Sex column (e.g., 'Sexo')
  - A 'cluster' column
  - Possibly other demographic fields

This script will:
1) Print the top N diagnoses per cluster by raw frequency.
2) Show the sex breakdown per cluster (and plot a bar chart of sex distribution).
3) Build a simple model (decision tree or logistic regression) for each cluster vs. the rest
   to see which diagnoses (and 'Sexo') best distinguish that cluster.
4) Plot bar charts of the top distinguishing features.

Example usage:
  python3 PYTHON/03-01-cluster-analysis.py \
    --csv_file RESULTS/clustered_patients-co-occ-sex.csv \
    --desc_file DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt \
    --cluster_col cluster \
    --diag_cols Diagnostico "Diagnostico 2" "Diagnostico 3" \
    --top_n 5 \
    --method tree \
    --sex_col Sexo

Dependencies:
  - python -m pip install pandas numpy scikit-learn matplotlib
"""

import pandas as pd
import numpy as np
import argparse
from collections import Counter
import matplotlib.pyplot as plt

# scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


###############################################################################
# 1) Load ICD-10 Descriptors
###############################################################################
def load_code_descriptors(desc_file):
    """
    Reads lines of the form:
       CODE   DESCRIPTION
    e.g.:
       F17210  Nicotine dependence
       F1210   Cannabis use, unspecified
    Returns {code -> short description}.
    """
    code2desc = {}
    if not desc_file:
        return code2desc

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


###############################################################################
# 2) Gather Diagnoses & Build Feature Matrix (Diag + Sexo)
###############################################################################
def gather_diagnoses_per_row(row, diag_columns):
    """
    Extract non-empty diagnoses from the row's diag_columns.
    Remove any '.' to align with code file if needed.
    """
    codes = []
    for col in diag_columns:
        val = row.get(col, "")
        if pd.notnull(val) and str(val).strip():
            # Remove the dot for consistent matching in descriptor files (if needed)
            code_no_dot = str(val).replace('.', '').strip()
            codes.append(code_no_dot)
    return codes


def build_feature_matrix(df, diag_columns, sex_col=None):
    """
    1) Gather all unique diagnosis codes from diag_columns.
    2) Build a one-hot encoding for those diagnoses across all rows.
    3) Optionally add the 'Sexo' column (numeric) if available.

    Returns:
      X -> DataFrame [n_patients x (n_unique_codes + possibly 1)]
      all_codes -> list of code columns
    """
    # Collect all diagnoses
    unique_codes = set()
    for _, row in df.iterrows():
        diag_list = gather_diagnoses_per_row(row, diag_columns)
        unique_codes.update(diag_list)
    all_codes = sorted(unique_codes)

    # Build one-hot matrix
    data = []
    for idx, row in df.iterrows():
        diag_list = set(gather_diagnoses_per_row(row, diag_columns))
        row_vec = [1 if c in diag_list else 0 for c in all_codes]
        data.append(row_vec)

    X = pd.DataFrame(data, columns=all_codes, index=df.index)

    # Add sex col if requested
    if sex_col and sex_col in df.columns:
        X["Sexo"] = df[sex_col]
    else:
        if sex_col:
            print(f"Warning: sex_col='{sex_col}' not found in DataFrame. Skipping 'Sexo' feature.")

    return X, all_codes


###############################################################################
# 3) Plot Cluster Sex Distribution
###############################################################################
def plot_cluster_sex_distribution(cluster_val, subdf, sex_col):
    """
    Creates a bar chart showing #patients in each sex category for this cluster.
    """
    sex_counts = subdf[sex_col].value_counts(dropna=False)
    labels = [f"Sexo={val}" for val in sex_counts.index]
    values = sex_counts.values

    plt.figure()
    plt.bar(labels, values)
    plt.title(f"Cluster {cluster_val}: Sex Distribution")
    plt.xlabel("Sex Category")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


###############################################################################
# 4) Show Top Diagnoses by Cluster & Print/Plot Sex Breakdown
###############################################################################
def analyze_top_codes_by_cluster(df, diag_columns, cluster_col="cluster",
                                 top_n=5, code2desc=None, sex_col=None):
    """
    For each cluster, print:
      - The top N diagnoses by frequency
      - The sex breakdown (if sex_col given), plus a bar chart
    """
    grouped = df.groupby(cluster_col)
    for cluster_val, subdf in grouped:
        # Summarize top diagnoses
        all_codes = []
        for idx, row in subdf.iterrows():
            diag_list = gather_diagnoses_per_row(row, diag_columns)
            all_codes.extend(diag_list)
        freq = Counter(all_codes).most_common(top_n)

        print(f"\n=== Cluster {cluster_val} ===")
        print(f"Top {top_n} Diagnoses:")
        for code, count in freq:
            desc = code2desc.get(code, "") if code2desc else ""
            if desc:
                print(f"  {code} ({desc}): {count} occurrences")
            else:
                print(f"  {code}: {count} occurrences")

        # Sex breakdown
        if sex_col and sex_col in subdf.columns:
            sex_counts = subdf[sex_col].value_counts(dropna=False)
            total_patients = len(subdf)
            print(f"\nSex distribution (N={total_patients}):")
            for sex_value, ccount in sex_counts.items():
                pct = 100.0 * ccount / total_patients
                print(f"  Sexo={sex_value}: {ccount} ({pct:.1f}%)")

            # Plot
            plot_cluster_sex_distribution(cluster_val, subdf, sex_col)
        else:
            print("(No sex_col provided or column not found—skipping sex breakdown.)")


###############################################################################
# 5) Plotting Helper for Distinguishing Diagnoses
###############################################################################
def plot_cluster_importances(cluster_id, top_features, code2desc,
                             title=None, max_desc_len=40):
    """
    Bar chart for cluster's top features (diagnoses or 'Sexo').
    If 'Sexo', label it clearly.
    """
    if not top_features:
        return

    labels, importances = [], []
    for code, imp in top_features:
        if code == "Sexo":
            desc = "Sexo"
        else:
            desc = code2desc.get(code, "")
            if not desc:
                desc = "Unknown code"

        # Truncate descriptor for neat display
        if len(desc) > max_desc_len:
            desc = desc[:max_desc_len] + "..."

        labels.append(desc)
        importances.append(imp)

    plt.figure()
    plt.bar(labels, importances)
    plt.title(title if title else f"Cluster {cluster_id}: Top Features")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


###############################################################################
# 6) Classification: Which Diagnoses Distinguish a Cluster?
###############################################################################
def analyze_cluster_feature_importance(X, df, cluster_col="cluster",
                                       method="tree", max_features=10,
                                       code2desc=None):
    """
    For each cluster c, build a binary classification model (c vs. rest).
    Then show top features with highest importance (decision tree or logistic).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression

    unique_clusters = sorted(df[cluster_col].unique())

    for c in unique_clusters:
        # Construct binary target: 1 if cluster==c, else 0
        y = (df[cluster_col] == c).astype(int)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if method == "logreg":
            model = LogisticRegression(max_iter=500, solver="liblinear")
        else:
            model = DecisionTreeClassifier(random_state=42, max_depth=6)

        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        # Feature importances
        if isinstance(model, DecisionTreeClassifier):
            importances = model.feature_importances_
        else:  # logistic
            importances = np.abs(model.coef_[0])

        # Sort descending
        idxs_desc = np.argsort(importances)[::-1]
        top_idx = idxs_desc[:max_features]

        top_feats = []
        for i in top_idx:
            # skip features with 0 importance
            if importances[i] > 0:
                feat_name = X.columns[i]
                feat_imp = importances[i]
                top_feats.append((feat_name, feat_imp))

        print(f"\nCluster {c}: {method} classifier (vs. all others)")
        print(f"  Test accuracy: {acc:.2f}")
        if not top_feats:
            print("  No significant features.")
            continue

        print(f"  Top {len(top_feats)} features:")
        for feat, imp in top_feats:
            if feat == "Sexo":
                desc_str = "Sexo"
            else:
                desc_str = code2desc.get(feat, "")
            if desc_str:
                print(f"    {feat} ({desc_str}): importance={imp:.4f}")
            else:
                print(f"    {feat}: importance={imp:.4f}")

        # Plot them
        plot_cluster_importances(
            cluster_id=c,
            top_features=top_feats,
            code2desc=code2desc,
            title=f"Cluster {c}: {method} (Acc={acc:.2f})"
        )


###############################################################################
# 7) MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True,
                        help="CSV with patient data & cluster labels (e.g., 'clustered_patients-co-occ-sex.csv').")
    parser.add_argument("--desc_file", default=None,
                        help="Optional ICD-10 descriptor file for code -> short text mapping.")
    parser.add_argument("--cluster_col", default="cluster",
                        help="Column name with cluster assignments (default='cluster').")
    parser.add_argument("--diag_cols", nargs="+", default=["Diagnostico"],
                        help="Diagnosis columns to analyze (e.g. 'Diagnostico', 'Diagnostico 2', ...).")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of top diagnoses to show per cluster (by frequency).")
    parser.add_argument("--method", choices=["tree", "logreg"], default="tree",
                        help="Model type for cluster vs. rest classification ('tree' or 'logreg').")
    parser.add_argument("--sex_col", default="Sexo",
                        help="Column name for sex (default='Sexo').")
    args = parser.parse_args()

    # 1) Load CSV
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} rows from '{args.csv_file}'. Clusters in '{args.cluster_col}'.")

    # 2) Load code descriptors
    code2desc = load_code_descriptors(args.desc_file)
    if code2desc:
        print(f"Loaded {len(code2desc)} ICD-10 descriptors from '{args.desc_file}'.")

    # 3) Analyze top frequent diagnoses by cluster, plus sex distribution
    print("\n======= Frequent Diagnoses by Cluster =======")
    analyze_top_codes_by_cluster(df, args.diag_cols, args.cluster_col,
                                 top_n=args.top_n, code2desc=code2desc, sex_col=args.sex_col)

    # 4) Build feature matrix (diagnoses + Sexo)
    print("\n======= Building Feature Matrix for Classification =======")
    X, all_codes = build_feature_matrix(df, args.diag_cols, sex_col=args.sex_col)
    print(f"Feature matrix shape: {X.shape[0]} rows x {X.shape[1]} columns.")

    # 5) Classification: which diagnoses (and 'Sexo') best distinguish each cluster?
    print("\n======= Feature Importance by Cluster =======")
    analyze_cluster_feature_importance(X, df,
                                       cluster_col=args.cluster_col,
                                       method=args.method,
                                       max_features=10,
                                       code2desc=code2desc)

    print("\nAnalysis complete. Check the console output and plots.")
    print("Done.")


if __name__ == "__main__":
    main()

