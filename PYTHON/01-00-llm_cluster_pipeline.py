#!/usr/bin/env python3
"""
Optimized LLM Clustering Pipeline (DeBERTa-v3)

Steps:
1. Load CSV dataset (patient rows with diagnosis codes).
2. Map ICD-10-CM codes to descriptive medical terms.
3. Train DeBERTa-v3 transformer via Masked Language Modeling (MLM).
4. Fine-tune embeddings using Contrastive Learning (DiffCSE-inspired).
5. Generate patient embeddings.
6. Determine optimal cluster count via silhouette bootstrapping.
7. Cluster embeddings using K-Means.
8. Identify cluster-defining ICD codes via c-DF-IPF.

Usage Example:
  python llm_cluster_pipeline.py --train_csv data.csv \
                                 --desc_file codes.txt \
                                 --mlm_epochs 5 \
                                 --fine_tune_epochs 5 \
                                 --batch_size 16 \
                                 --max_clusters 10

Dependencies:
  pip install transformers datasets torch scikit-learn pandas numpy tqdm
"""

import argparse
from pipeline import (
    SingleRowEHRDataset, MLMDataset, mlm_pretrain,
    ContrastiveDataset, contrastive_fine_tune,
    generate_embeddings, bootstrap_find_k,
    run_kmeans, compute_cdfipf
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Input CSV file path.")
    parser.add_argument("--desc_file", required=True, help="ICD code-description file path.")
    parser.add_argument("--mlm_epochs", default=5, type=int, help="MLM pre-training epochs.")
    parser.add_argument("--fine_tune_epochs", default=5, type=int, help="Contrastive fine-tuning epochs.")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_clusters", default=10, type=int)
    args = parser.parse_args()

    raw_dataset = SingleRowEHRDataset(args.train_csv, args.desc_file)
    mlm_dataset = MLMDataset(raw_dataset)

    print("[1] MLM Pre-training")
    mlm_pretrain(mlm_dataset, epochs=args.mlm_epochs, batch_size=args.batch_size)

    print("[2] Contrastive Fine-tuning")
    contrastive_data = ContrastiveDataset(raw_dataset)
    contrastive_fine_tune("mlm_pretrain", contrastive_data,
                          epochs=args.fine_tune_epochs, batch_size=args.batch_size)

    print("[3] Generating Embeddings")
    embeddings = generate_embeddings("contrastive_model", contrastive_data,
                                     batch_size=args.batch_size)

    print("[4] Finding Optimal Cluster Number")
    best_k = bootstrap_find_k(embeddings, max_k=args.max_clusters, 
                              n_bootstrap=10, sample_frac=0.05)

    print(f"Optimal number of clusters determined: {best_k}")

    print("[5] K-Means Clustering")
    labels = run_kmeans(embeddings, best_k)

    df = raw_dataset.df.copy()
    df["cluster"] = labels

    print("[6] Computing Cluster-specific c-DF-IPF Scores")
    cluster2dfipf = compute_cdfipf(df)
    for cluster, codes in cluster2dfipf.items():
        top5 = sorted(codes.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nCluster {cluster} top codes:")
        for code, score in top5:
            print(f"  {code}: {score:.4f}")

    output_file = "clustered_patients.csv"
    df.to_csv(output_file, index=False)
    print(f"\nClustering complete. Results saved in '{output_file}'.")

if __name__ == "__main__":
    main()

