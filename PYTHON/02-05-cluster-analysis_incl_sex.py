#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze clusters by:
1) Showing top diagnoses per cluster by frequency (with ICD-10 descriptors).
   Also prints the sex breakdown for each cluster (Sexo=1 or Sexo=2) and
   plots a bar chart of the sex distribution.
2) Building a simple classification model (decision tree or logistic regression)
   for each cluster vs. the rest, to find which diagnoses (and possibly 'Sexo')
   best distinguish that cluster.
3) Plotting a bar chart of the top distinguishing diagnoses using only the
   ICD-10 descriptors (or "Sexo") on the x-axis. If a descriptor is missing,
   show "Unknown code".

Example usage:
  python3 cluster_analysis_with_sex.py \
    --csv_file RESULTS/clustered_patients-co-occ.csv \
    --desc_file DATA/Code-descriptions/icd10cm-codes.txt \
    --cluster_col cluster \
    --diag_cols Diagnostico Diagnostico2 Diagnostico3 \
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

    Returns a dict {code -> short description}.
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
# 2) Gathering Diagnoses & Building Feature Matrix (Diag + Sexo)
###############################################################################
def gather_diagnoses_per_row(row, diag_columns):
    """
    Extract all non-empty diagnoses from the row's diag_columns,
    removing any dot '.' for consistent matching to descriptor file.
    """
    codes = []
    for col in diag_columns:
        val = row.get(col, "")
        if pd.notnull(val) and str(val).strip():
            # Remove the dot for matching with descriptor file
            code_no_dot = str(val).replace('.', '').strip()
            codes.append(code_no_dot)
    return codes

def build_feature_matrix(df, diag_columns, sex_col=None):
    """
    1) Gather all unique diagnosis codes (dot removed).
    2) Build a one-hot encoding for those diagnoses.
    3) Optionally add a 'Sexo' column as an extra feature if sex_col is given.

    Returns:
      X -> DataFrame [n_patients x (n_unique_codes + possibly 1)]
      all_codes -> list of code columns
    """
    unique_codes = set()
    for _, row in df.iterrows():
        diag_list = gather_diagnoses_per_row(row, diag_columns)
        unique_codes.update(diag_list)

    all_codes = sorted(unique_codes)

    # Build one-hot encoding
    data = []
    for idx, row in df.iterrows():
        diag_list = set(gather_diagnoses_per_row(row, diag_columns))
        row_vec = [1 if c in diag_list else 0 for c in all_codes]
        data.append(row_vec)

    X = pd.DataFrame(data, columns=all_codes, index=df.index)

    # If there's a known sex_col, add it
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
    Creates a bar chart showing how many patients are in each Sexo category
    for the given cluster (e.g. Sexo=1, Sexo=2).
    """
    sex_counts = subdf[sex_col].value_counts(dropna=False)
    labels = [f"Sexo={val}" for val in sex_counts.index]
    values = sex_counts.values

    plt.figure()
    plt.bar(labels, values)
    plt.title(f"Cluster {cluster_val}: Sex Distribution")
    plt.xlabel("Sex")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

###############################################################################
# 4) Show Top Diagnoses by Cluster & Print/Plot Sex Breakdown
###############################################################################
def analyze_top_codes_by_cluster(df, diag_columns, cluster_col="cluster",
                                 top_n=10, code2desc=None, sex_col=None):
    """
    For each cluster, print:
      - The top N diagnoses by frequency
      - The breakdown of sex (if sex_col is provided) (1 or 2)
      - A bar chart of how many patients are Sexo=1, Sexo=2, etc.
    """
    grouped = df.groupby(cluster_col)
    for cluster_val, subdf in grouped:
        # Print top diagnoses
        all_codes = []
        for idx, row in subdf.iterrows():
            diag_list = gather_diagnoses_per_row(row, diag_columns)
            all_codes.extend(diag_list)
        freq = Counter(all_codes).most_common(top_n)

        print(f"\n=== Cluster {cluster_val}: Top {top_n} Diagnoses ===")
        for code, count in freq:
            desc = code2desc.get(code, "") if code2desc else ""
            if desc:
                print(f"  {code} ({desc}): {count} occurrences")
            else:
                print(f"  {code}: {count} occurrences")

        # Print sex breakdown & plot
        if sex_col and sex_col in subdf.columns:
            sex_counts = subdf[sex_col].value_counts(dropna=False)
            total_patients = len(subdf)
            print(f"\nSex breakdown in cluster {cluster_val} (N={total_patients}):")
            for sex_value, count in sex_counts.items():
                percent = 100.0 * count / total_patients
                print(f"  Sexo={sex_value}: {count} patients ({percent:.1f}%)")

            plot_cluster_sex_distribution(cluster_val, subdf, sex_col)
        else:
            print("\n(No sex_col provided or column not found—cannot plot sex breakdown.)")

###############################################################################
# 5) Plotting Helper for Distinguishing Diagnoses
###############################################################################
def plot_cluster_importances(cluster_id, top_features, code2desc,
                             title=None, max_desc_len=40):
    """
    Creates a bar chart for a cluster's top features (diagnoses or 'Sexo') vs. importance.
    If feature == 'Sexo', label it as "Sexo" (no ICD-10 descriptor).
    """
    if not top_features:
        return

    labels = []
    importances = []

    for code, imp in top_features:
        if code == "Sexo":
            desc = "Sexo"
        else:
            desc = code2desc.get(code, "")
            if not desc:
                desc = "Unknown code"

        # Limit descriptor length
        if len(desc) > max_desc_len:
            desc = desc[:max_desc_len] + "..."

        labels.append(desc)
        importances.append(imp)

    plt.figure()
    plt.bar(labels, importances)
    if title:
        plt.title(title)
    else:
        plt.title(f"Cluster {cluster_id}: Top Diagnoses by Importance")
    plt.xlabel("Feature (Diagnosis or Sex)")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

###############################################################################
# 6) Classification to Find Distinguishing Diagnoses (and Sex)
###############################################################################
def analyze_cluster_feature_importance(X, df, cluster_col="cluster",
                                       method="tree", max_features=10, code2desc=None):
    """
    For each cluster c, build a classification model (c vs rest) to see which features
    best distinguish that cluster. Features can be diagnoses + 'Sexo' if present.

    Prints results and then plots them.
    """
    unique_clusters = sorted(df[cluster_col].unique())

    for c in unique_clusters:
        # binary target: cluster c vs. all others
        y = (df[cluster_col] == c).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if method == "logreg":
            model = LogisticRegression(max_iter=500, solver="liblinear")
        else:
            model = DecisionTreeClassifier(random_state=42, max_depth=6)

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        if isinstance(model, DecisionTreeClassifier):
            importances = model.feature_importances_
        else:  # logistic
            importances = np.abs(model.coef_[0])

        feat_indices = np.argsort(importances)[::-1]
        top_idx = feat_indices[:max_features]

        # gather top features (skip if exactly 0 importance)
        top_features = [(X.columns[i], importances[i]) for i in top_idx
                        if importances[i] > 0]

        print(f"\nCluster {c}: {method} classification (vs. all others)")
        print(f"  Test accuracy: {score:.2f}")

        if not top_features:
            print("  No significant features found.")
            continue

        print(f"  Top ~{max_features} distinguishing features:")
        for feat, imp in top_features:
            if feat == "Sexo":
                desc_str = "Sexo"
            else:
                desc_str = code2desc.get(feat, "") if code2desc else ""
            if desc_str:
                print(f"    {feat} ({desc_str}) => importance={imp:.4f}")
            else:
                print(f"    {feat} => importance={imp:.4f}")

        # Plot them
        plot_cluster_importances(
            cluster_id=c,
            top_features=top_features,
            code2desc=code2desc,
            title=f"Cluster {c}: {method} (Accuracy={score:.2f})"
        )

###############################################################################
# 7) MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True, help="CSV with patient data & cluster labels.")
    parser.add_argument("--desc_file", default=None, help="File with ICD-10 code descriptors.")
    parser.add_argument("--cluster_col", default="cluster", help="Column containing cluster labels.")
    parser.add_argument("--diag_cols", nargs="+", default=["Diagnostico"],
                        help="List of diagnosis columns to gather.")
    parser.add_argument("--top_n", type=int, default=5,
                        help="How many top diagnoses to show per cluster.")
    parser.add_argument("--method", default="tree", choices=["tree", "logreg"],
                        help="Model type for feature importance.")
    parser.add_argument("--sex_col", default="Sexo",
                        help="Column name for sex/gender (default='Sexo').")
    args = parser.parse_args()

    # 1) Load data
    df = pd.read_csv(args.csv_file, low_memory=True)
    print(f"Loaded {len(df)} rows from '{args.csv_file}'. Using cluster col '{args.cluster_col}'.")

    # 2) Load descriptors
    code2desc = load_code_descriptors(args.desc_file)
    if code2desc:
        print(f"Loaded {len(code2desc)} descriptors from '{args.desc_file}'.")

    # 3) Show top frequent diagnoses + sex breakdown
    print("\n========== Frequent Diagnoses by Cluster ==========")
    analyze_top_codes_by_cluster(
        df=df,
        diag_columns=args.diag_cols,
        cluster_col=args.cluster_col,
        top_n=args.top_n,
        code2desc=code2desc,
        sex_col=args.sex_col
    )

    # 4) Build feature matrix (diagnoses + sex)
    print("\n========== Building Feature Matrix ==========")
    X, all_codes = build_feature_matrix(df, diag_columns=args.diag_cols, sex_col=args.sex_col)
    print(f"Feature matrix shape: {X.shape}")
    if args.sex_col and args.sex_col in df.columns:
        print(f"Including '{args.sex_col}' as a feature. #diagnosis columns = {len(all_codes)}")

    # 5) Classification for each cluster
    print("\n========== Feature Importance & Plots ==========")
    analyze_cluster_feature_importance(
        X=X,
        df=df,
        cluster_col=args.cluster_col,
        method=args.method,
        max_features=10,
        code2desc=code2desc
    )

    print("\nDone! You've seen frequency-based top diagnoses, plus sex breakdown & plots,")
    print("plus classification-based feature importance (including sex if relevant).")

if __name__ == "__main__":
    main()
