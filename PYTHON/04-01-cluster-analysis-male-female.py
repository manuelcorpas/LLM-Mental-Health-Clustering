#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render male & female clusters in ONE figure (cluster-level embeddings),
using short labels ('M3', 'F9', etc.) with NO lines/arrows and NO automatic nudging.

Steps:
 1) Read male_csv/female_csv. For each row, rename the cluster to 'male_cluster_X' or
    'female_cluster_X'.
 2) Combine them, build row-level TF-IDF across all rows.
 3) For each cluster, average row embeddings => single cluster embedding.
 4) Dim-reduce (PCA or UMAP) and plot them in one figure. We label each cluster
    with a short code, e.g. "M3", placed slightly to the right, with no lines/arrows.
 5) Print top ICD codes & top TF-IDF terms for each cluster.


Usage Example:
  python3.11 PYTHON/04-01-cluster-analysis-male-female.py \
    --male_csv RESULTS/clustered_patients_male.csv \
    --female_csv RESULTS/clustered_patients_female.csv \
    --desc_file DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt \
    --cluster_col cluster \
    --diag_cols Diagnostico "Diagnostico 2" \
    --top_n 5 \
    --csv_encoding utf-8 \
    --method pca

Requires:
  pip install pandas numpy scikit-learn matplotlib
  (If you want method=umap, also pip install umap-learn)
"""

import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

###############################################################################
# 1) Shorten cluster labels: e.g. "male_cluster_3" => "M3"
###############################################################################
def short_label(full_label):
    if full_label.startswith("male_cluster_"):
        return "M" + full_label.split("_")[-1]
    elif full_label.startswith("female_cluster_"):
        return "F" + full_label.split("_")[-1]
    return full_label


###############################################################################
# 2) Load optional ICD code->descriptor
###############################################################################
def load_code_descriptors(path):
    code2desc = {}
    if not path:
        return code2desc
    with open(path, "r", encoding="utf-8") as f:
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
# 3) Convert row to doc string (ICD codes -> descriptors)
###############################################################################
def row_to_doc(row, diag_cols, code2desc):
    desc_list = []
    for c in diag_cols:
        val = row.get(c, "")
        if pd.notnull(val) and str(val).strip():
            code_clean = str(val).replace('.', '').strip()
            if code2desc and code_clean in code2desc:
                descriptor = re.sub(r"\s+", "_", code2desc[code_clean].strip())
                desc_list.append(descriptor)
            else:
                desc_list.append(code_clean)
    return " ".join(desc_list)


###############################################################################
# 4) Build docs & store raw codes
###############################################################################
def build_docs_codes(df, diag_cols, code2desc):
    doc_list = []
    row_codes = []
    for _, row in df.iterrows():
        doc = row_to_doc(row, diag_cols, code2desc)
        doc_list.append(doc)

        codes_here = []
        for c in diag_cols:
            val = row.get(c, "")
            if pd.notnull(val) and str(val).strip():
                codes_here.append(str(val).replace('.', '').strip())
        row_codes.append(codes_here)
    return doc_list, row_codes


###############################################################################
# 5) Build row-level TF-IDF for entire dataset
###############################################################################
def build_tfidf_embeddings(doc_list):
    vectorizer = TfidfVectorizer(
        token_pattern=r'[^\s]+',
        stop_words='english'
    )
    X = vectorizer.fit_transform(doc_list)
    vocab = vectorizer.get_feature_names_out()
    return X, vocab, vectorizer


###############################################################################
# 6) For each cluster, average row embeddings => cluster-level embedding
###############################################################################
def compute_cluster_embeddings(X, cluster_labels):
    cluster2idxs = defaultdict(list)
    for i, cval in enumerate(cluster_labels):
        cluster2idxs[cval].append(i)

    cluster2vec = {}
    for cval, idxs in cluster2idxs.items():
        subX = X[idxs]
        mean_vec = subX.mean(axis=0).A1
        cluster2vec[cval] = mean_vec
    return cluster2vec


###############################################################################
# 7) Dim reduce cluster embeddings (pca or umap)
###############################################################################
def reduce_cluster_embeddings(cluster2vec, method='pca'):
    if not cluster2vec:
        return {}
    labels = sorted(cluster2vec.keys())
    arr = np.array([cluster2vec[l] for l in labels])
    if arr.shape[0] < 2:
        print("(Warning) <2 clusters => skipping plot.")
        return {}
    if method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            print("UMAP not installed => fallback to PCA.")
            method = 'pca'
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(arr)
            coords_2d = {lab: coords[i] for i, lab in enumerate(labels)}
            return coords_2d

    # Default: PCA
    if arr.shape[1] < 2:
        print("(Warning) <2 features => can't do PCA.")
        return {}
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(arr)
    coords_2d = {lab: coords[i] for i, lab in enumerate(labels)}
    return coords_2d


###############################################################################
# 8) Plot cluster-level embeddings in a single figure with short labels, no arrows, no nudging
###############################################################################
def plot_cluster_embeddings(coords_2d):
    """
    coords_2d: { cluster_label -> (x, y) }
    We'll do 1 point per cluster, color-coded, short label next to each point (slight offset).
    No automatic nudging or lines.
    """
    if not coords_2d:
        print("(No cluster-level coords to plot).")
        return
    labels = sorted(coords_2d.keys())
    xs = [coords_2d[l][0] for l in labels]
    ys = [coords_2d[l][1] for l in labels]

    plt.figure(figsize=(9,7))
    plt.scatter(xs, ys, c=range(len(labels)), cmap='tab10', s=120)

    # Place short label with a small horizontal offset
    offset_x = 0.005  # you can tweak this if you want them closer/farther
    for i, lab in enumerate(labels):
        short = short_label(lab)
        plt.text(xs[i] + offset_x, ys[i], short, fontsize=9)

    plt.title("Cluster-level Embeddings (Short-Labelled, no lines, no nudging)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()


###############################################################################
# 9) Summaries: top ICD codes + top TF-IDF terms
###############################################################################
def print_cluster_summaries(cluster_labels, row_codes, X, vocab, top_n=5):
    cluster2rows = defaultdict(list)
    for i, cval in enumerate(cluster_labels):
        cluster2rows[cval].append(i)

    for cval in sorted(cluster2rows.keys()):
        idxs = cluster2rows[cval]
        print(f"\n=== {cval} ===")
        print(f"  #rows: {len(idxs)}")

        # top ICD codes
        code_list = []
        for irow in idxs:
            code_list.extend(row_codes[irow])
        freq = Counter(code_list).most_common(top_n)
        print(f"  Top {top_n} ICD codes by freq:")
        for cd, cnt in freq:
            print(f"    {cd} => {cnt}")

        # top TF-IDF terms from average
        subX = X[idxs]
        mean_vec = subX.mean(axis=0).A1
        top_idx = np.argsort(-mean_vec)[:top_n]
        top_tokens = [vocab[i_t] for i_t in top_idx]
        joined = " + ".join(top_tokens)
        print(f"  Dominant semantic terms (top {top_n}):\n  {joined}")


###############################################################################
# 10) MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--male_csv", default=None)
    parser.add_argument("--female_csv", default=None)
    parser.add_argument("--desc_file", default=None)
    parser.add_argument("--cluster_col", default="cluster")
    parser.add_argument("--diag_cols", nargs="+", default=["Diagnostico"])
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--csv_encoding", default="utf-8")
    parser.add_argument("--method", default="pca", choices=["pca","umap"])
    args = parser.parse_args()

    # 1) Load male/female
    dfs = []
    if args.male_csv:
        df_m = pd.read_csv(args.male_csv, encoding=args.csv_encoding)
        if args.cluster_col not in df_m.columns:
            print(f"(Error) cluster_col='{args.cluster_col}' not in male CSV => skip.")
        else:
            old_clus = df_m[args.cluster_col].tolist()
            new_clus = [f"male_cluster_{oc}" for oc in old_clus]
            df_m["MergedCluster"] = new_clus
            dfs.append(df_m)
            print(f"Loaded {len(df_m)} male rows from {args.male_csv}.")

    if args.female_csv:
        df_f = pd.read_csv(args.female_csv, encoding=args.csv_encoding)
        if args.cluster_col not in df_f.columns:
            print(f"(Error) cluster_col='{args.cluster_col}' not in female CSV => skip.")
        else:
            old_clus = df_f[args.cluster_col].tolist()
            new_clus = [f"female_cluster_{oc}" for oc in old_clus]
            df_f["MergedCluster"] = new_clus
            dfs.append(df_f)
            print(f"Loaded {len(df_f)} female rows from {args.female_csv}.")

    if not dfs:
        print("No valid data => exiting.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(df_all)} total rows from male + female subsets.\n")

    code2desc = load_code_descriptors(args.desc_file)
    if code2desc:
        print(f"Loaded {len(code2desc)} code descriptors from '{args.desc_file}'.")

    doc_list, row_codes = build_docs_codes(df_all, args.diag_cols, code2desc)
    X, vocab, vectorizer = build_tfidf_embeddings(doc_list)

    cluster_labels = df_all["MergedCluster"].tolist()
    print("=== Summaries for each cluster ===")
    print_cluster_summaries(cluster_labels, row_codes, X, vocab, top_n=args.top_n)

    # cluster-level
    cluster2vec = compute_cluster_embeddings(X, cluster_labels)
    coords_2d = reduce_cluster_embeddings(cluster2vec, method=args.method)
    plot_cluster_embeddings(coords_2d)

    print("\nAll done.")


if __name__ == "__main__":
    main()
