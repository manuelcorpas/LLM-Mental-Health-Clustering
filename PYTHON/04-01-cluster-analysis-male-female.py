#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render male & female clusters with PCA in THREE plots:
 1) Combined (male & female together) with male clusters in one color,
    female clusters in another.
 2) Only male clusters.
 3) Only female clusters.

Usage Example:
  python3.11 04-01-cluster-analysis-male-female.py \
    --male_csv RESULTS/clustered_patients_male.csv \
    --female_csv RESULTS/clustered_patients_female.csv \
    --desc_file DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt \
    --cluster_col cluster \
    --diag_cols PrincipalDiagnosis Diagnosis2 Diagnosis3 \
    --top_n 5 \
    --csv_encoding utf-8 \
    --method pca

Requires:
  pip install pandas numpy scikit-learn matplotlib
  (If you want UMAP, also pip install umap-learn)
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
            if code_clean in code2desc:
                # use descriptor
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
                code_clean = str(val).replace('.', '').strip()
                codes_here.append(code_clean)
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
    """
    Return {cluster_label -> 1D np.array(mean TF-IDF features)}.
    """
    cluster2idxs = defaultdict(list)
    for i, cval in enumerate(cluster_labels):
        cluster2idxs[cval].append(i)

    cluster2vec = {}
    for cval, idxs in cluster2idxs.items():
        subX = X[idxs]
        mean_vec = subX.mean(axis=0).A1  # .A1 => 1D array
        cluster2vec[cval] = mean_vec
    return cluster2vec

###############################################################################
# 7) Dim reduce cluster embeddings (pca or umap)
###############################################################################
def reduce_cluster_embeddings(cluster2vec, method='pca'):
    """
    Return {cluster_label -> (x, y)} 2D coords after dimension reduction.
    """
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

    # fallback = pca
    if arr.shape[1] < 2:
        print("(Warning) <2 features => can't do PCA.")
        return {}
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(arr)
    coords_2d = {lab: coords[i] for i, lab in enumerate(labels)}
    return coords_2d

###############################################################################
# 8) Plotting
###############################################################################
def plot_clusters_combined(coords_2d):
    """
    Single figure with male clusters in one color, female clusters in another.
    """
    if not coords_2d:
        print("(No coords to plot).")
        return
    labels = sorted(coords_2d.keys())
    xs = [coords_2d[l][0] for l in labels]
    ys = [coords_2d[l][1] for l in labels]

    # separate male vs female
    male_idx = []
    female_idx = []
    for i, lab in enumerate(labels):
        if lab.startswith("male_cluster_"):
            male_idx.append(i)
        else:
            female_idx.append(i)

    plt.figure(figsize=(8,6))
    # Plot male clusters as one color, female as another
    for i in male_idx:
        plt.scatter(xs[i], ys[i], color='blue', s=120)
        plt.text(xs[i]+0.005, ys[i], short_label(labels[i]), fontsize=9)

    for i in female_idx:
        plt.scatter(xs[i], ys[i], color='red', s=120)
        plt.text(xs[i]+0.005, ys[i], short_label(labels[i]), fontsize=9)

    plt.title("Combined Clusters (Males in Blue, Females in Red)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()


def plot_clusters_subset(coords_2d, subset="male"):
    """
    If subset=="male", plot only male_cluster_* in one color (blue).
    If subset=="female", plot only female_cluster_* in one color (red).
    Each short label is placed near the point.
    We run PCA or UMAP on that subset's cluster embeddings only (the caller
    should have done reduce_cluster_embeddings on the subset).
    """
    if not coords_2d:
        print("(No coords for subset).")
        return

    labels = sorted(coords_2d.keys())
    if subset=="male":
        color = "blue"
        plot_title = "Male Clusters Only"
    else:
        color = "red"
        plot_title = "Female Clusters Only"

    xs = []
    ys = []
    label_list = []
    for lab in labels:
        if subset=="male" and lab.startswith("male_cluster_"):
            xs.append(coords_2d[lab][0])
            ys.append(coords_2d[lab][1])
            label_list.append(lab)
        elif subset=="female" and lab.startswith("female_cluster_"):
            xs.append(coords_2d[lab][0])
            ys.append(coords_2d[lab][1])
            label_list.append(lab)

    if not xs:
        print(f"(No {subset} clusters to plot).")
        return

    plt.figure(figsize=(8,6))
    for i, lab in enumerate(label_list):
        plt.scatter(xs[i], ys[i], color=color, s=120)
        plt.text(xs[i]+0.005, ys[i], short_label(lab), fontsize=9)

    plt.title(plot_title)
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
    parser.add_argument("--cluster_col", default="cluster",
                        help="Column in each CSV that stores the cluster label.")
    parser.add_argument("--diag_cols", nargs="+", default=["PrincipalDiagnosis"],
                        help="Which columns contain ICD codes or diagnoses.")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--csv_encoding", default="utf-8")
    parser.add_argument("--method", default="pca", choices=["pca","umap"])
    args = parser.parse_args()

    # 1) Load male/female CSVs
    df_list = []
    if args.male_csv:
        df_m = pd.read_csv(args.male_csv, encoding=args.csv_encoding)
        if args.cluster_col not in df_m.columns:
            print(f"(Error) cluster_col='{args.cluster_col}' not in male CSV => skip.")
        else:
            old_clus = df_m[args.cluster_col].tolist()
            new_clus = [f"male_cluster_{oc}" for oc in old_clus]
            df_m["MergedCluster"] = new_clus
            df_list.append(df_m)
            print(f"Loaded {len(df_m)} rows from male CSV: {args.male_csv}")

    if args.female_csv:
        df_f = pd.read_csv(args.female_csv, encoding=args.csv_encoding)
        if args.cluster_col not in df_f.columns:
            print(f"(Error) cluster_col='{args.cluster_col}' not in female CSV => skip.")
        else:
            old_clus = df_f[args.cluster_col].tolist()
            new_clus = [f"female_cluster_{oc}" for oc in old_clus]
            df_f["MergedCluster"] = new_clus
            df_list.append(df_f)
            print(f"Loaded {len(df_f)} rows from female CSV: {args.female_csv}")

    if not df_list:
        print("No valid data => exiting.")
        return

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"\nCombined: {len(df_all)} total rows from male + female subsets.\n")

    # 2) Load code descriptors
    code2desc = load_code_descriptors(args.desc_file)
    if code2desc:
        print(f"Loaded {len(code2desc)} code descriptors from '{args.desc_file}'.")

    # 3) Build doc strings & row_codes
    doc_list, row_codes = build_docs_codes(df_all, args.diag_cols, code2desc)

    # 4) Build TF-IDF
    X, vocab, _ = build_tfidf_embeddings(doc_list)

    # 5) Summaries
    cluster_labels = df_all["MergedCluster"].tolist()
    print("=== Summaries for each cluster ===")
    print_cluster_summaries(cluster_labels, row_codes, X, vocab, top_n=args.top_n)

    # 6) cluster-level embeddings for the COMBINED set
    cluster2vec_combined = compute_cluster_embeddings(X, cluster_labels)
    coords_2d_combined = reduce_cluster_embeddings(cluster2vec_combined, method=args.method)

    # Plot A: combined with 2 colors
    plot_clusters_combined(coords_2d_combined)

    # Plot B: only male clusters => we must compute subset embeddings alone
    male_clusters = [k for k in cluster2vec_combined.keys() if k.startswith("male_cluster_")]
    if len(male_clusters) >= 2:
        sub_male = {mc: cluster2vec_combined[mc] for mc in male_clusters}
        coords_male = reduce_cluster_embeddings(sub_male, method=args.method)
        plot_clusters_subset(coords_male, subset="male")
    else:
        print("(Not enough male clusters for a separate male-only PCA plot)")

    # Plot C: only female clusters => we must compute subset embeddings alone
    female_clusters = [k for k in cluster2vec_combined.keys() if k.startswith("female_cluster_")]
    if len(female_clusters) >= 2:
        sub_female = {fc: cluster2vec_combined[fc] for fc in female_clusters}
        coords_female = reduce_cluster_embeddings(sub_female, method=args.method)
        plot_clusters_subset(coords_female, subset="female")
    else:
        print("(Not enough female clusters for a separate female-only PCA plot)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
