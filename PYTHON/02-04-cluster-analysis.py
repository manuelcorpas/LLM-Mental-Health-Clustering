#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze clusters by:
1) Showing top diagnoses per cluster by frequency (with ICD-10 descriptors).
2) Building a simple classification model (decision tree or logistic regression) for each cluster vs. the rest,
   to find which diagnoses best distinguish that cluster.
3) Plotting a bar chart of the top distinguishing diagnoses using **only** the ICD-10 descriptors on the x-axis.
   If a descriptor is missing, show "Unknown code".

Usage Example:
  python3.11 PYTHON/02-04-cluster-analysis.py \
    --csv_file RESULTS/clustered_patients-co-occ.csv \
    --desc_file DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt \
    --cluster_col cluster \
    --diag_cols Diagnostico Diagnostico2 Diagnostico3 \
    --top_n 5 \
    --method tree

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
# 2) Gathering Diagnoses & Building One-Hot
###############################################################################
def gather_diagnoses_per_row(row, diag_columns):
    """Extract all non-empty diagnoses from the row's diag_columns, removing any dot."""
    codes = []
    for col in diag_columns:
        val = row.get(col, "")
        if pd.notnull(val) and str(val).strip():
            # Remove the dot for matching with the descriptor file
            code_no_dot = str(val).replace('.', '').strip()
            codes.append(code_no_dot)
    return codes


def build_one_hot_matrix(df, diag_columns):
    """
    Creates one-hot matrix for all diagnoses in diag_columns.
    Returns (X, all_codes):
      X -> DataFrame of shape [n_patients x n_unique_codes]
      all_codes -> list of all unique codes (column ordering)
    """
    unique_codes = set()
    for idx, row in df.iterrows():
        diag_list = gather_diagnoses_per_row(row, diag_columns)
        unique_codes.update(diag_list)

    all_codes = sorted(unique_codes)
    data = []
    for idx, row in df.iterrows():
        diag_list = set(gather_diagnoses_per_row(row, diag_columns))
        row_vec = [1 if c in diag_list else 0 for c in all_codes]
        data.append(row_vec)

    X = pd.DataFrame(data, columns=all_codes, index=df.index)
    return X, all_codes

###############################################################################
# 3) Show Top Diagnoses by Frequency for Each Cluster
###############################################################################
def analyze_top_codes_by_cluster(df, diag_columns, cluster_col="cluster", top_n=10, code2desc=None):
    """
    For each cluster, print the top N diagnoses by frequency.
    If code2desc is provided, also print the short descriptor.
    """
    grouped = df.groupby(cluster_col)
    for cluster_val, subdf in grouped:
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

###############################################################################
# 4) Plotting Helper for Distinguishing Diagnoses
###############################################################################
def plot_cluster_importances(cluster_id, top_features, code2desc, title=None, max_desc_len=40):
    """
    Creates a bar chart for a cluster's top features (diagnoses) vs. importance.
    **Shows only the descriptor** on the x-axis. If missing, uses "Unknown code".
    By default, we plot only the first 20 characters of the descriptor.
    """
    if not top_features:
        return

    labels = []
    importances = []

    for code, imp in top_features:
        desc = code2desc.get(code, "")
        if not desc:
            desc = "Unknown code"
        # Limit descriptor to 20 chars
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
    plt.xlabel("Diagnosis Descriptor")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

###############################################################################
# 5) Classification to Find Distinguishing Diagnoses
###############################################################################
def analyze_cluster_feature_importance(
    X, df, cluster_col="cluster", method="tree", max_features=20, code2desc=None
):
    """
    For each cluster c, build a classification model (c vs rest) to see which diagnoses
    best distinguish that cluster. Prints results and then plots them.

    method: 'tree' or 'logreg'
    """
    unique_clusters = sorted(df[cluster_col].unique())

    for c in unique_clusters:
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

        # fetch feature importance
        if isinstance(model, DecisionTreeClassifier):
            importances = model.feature_importances_
        else:  # logistic
            importances = np.abs(model.coef_[0])

        feat_indices = np.argsort(importances)[::-1]
        top_idx = feat_indices[:max_features]
        # store (code, importance) pairs
        top_features = [(X.columns[i], importances[i]) for i in top_idx if importances[i] > 0]

        print(f"\nCluster {c}: {method} classification (vs. all others)")
        print(f"  Test accuracy: {score:.2f}")
        if not top_features:
            print("  No significant features found.")
            continue
        print(f"  Top ~{max_features} distinguishing diagnoses:")
        for feat, imp in top_features:
            d = code2desc.get(feat, "") if code2desc else ""
            if d:
                print(f"    {feat} ({d}) => importance={imp:.4f}")
            else:
                print(f"    {feat} => importance={imp:.4f}")

        # plot them with descriptors only
        plot_cluster_importances(
            cluster_id=c,
            top_features=top_features,
            code2desc=code2desc,
            title=f"Cluster {c}: {method} (Accuracy={score:.2f})"
        )

###############################################################################
# 6) MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True, help="CSV with patient data & cluster labels.")
    parser.add_argument("--desc_file", default=None, help="File with ICD-10 code descriptors.")
    parser.add_argument("--cluster_col", default="cluster", help="Column containing cluster labels.")
    parser.add_argument("--diag_cols", nargs="+", default=["Diagnostico"],
                        help="List of diagnosis columns to gather.")
    parser.add_argument("--top_n", type=int, default=5, help="How many top diagnoses to show per cluster.")
    parser.add_argument("--method", default="tree", choices=["tree", "logreg"],
                        help="Model type for feature importance.")
    args = parser.parse_args()

    # 1) Load data
    df = pd.read_csv(args.csv_file, low_memory=True)
    print(f"Loaded {len(df)} rows from '{args.csv_file}'. Using cluster col '{args.cluster_col}'.")

    # 2) Load descriptors
    code2desc = load_code_descriptors(args.desc_file)
    if code2desc:
        print(f"Loaded {len(code2desc)} descriptors from '{args.desc_file}'.")

    # 3) Show top frequent diagnoses in each cluster
    print("\n========== Frequent Diagnoses by Cluster ==========")
    analyze_top_codes_by_cluster(
        df,
        diag_columns=args.diag_cols,
        cluster_col=args.cluster_col,
        top_n=args.top_n,
        code2desc=code2desc
    )

    # 4) Build one-hot matrix
    print("\n========== Building One-Hot Matrix ==========")
    X, all_codes = build_one_hot_matrix(df, args.diag_cols)
    print(f"One-hot shape: {X.shape}. #unique codes = {len(all_codes)}")

    # 5) For each cluster, do classification (cluster vs. rest)
    print("\n========== Feature Importance & Plots ==========")
    analyze_cluster_feature_importance(
        X, df,
        cluster_col=args.cluster_col,
        method=args.method,
        max_features=10,
        code2desc=code2desc
    )

    print("\nDone! You've seen frequency-based top diagnoses, plus feature-importance with descriptor-only plots.")

if __name__ == "__main__":
    main()
