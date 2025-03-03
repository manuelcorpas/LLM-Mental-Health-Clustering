#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-Contained LLM Clustering Pipeline (DeBERTa-v3) for Co-Occurrence Diagnoses,
analyzing males (Sexo=1) and females (Sexo=2) separately.

Usage Example:
  python3 PYTHON/04-00-llm_cluster_co-occ_sep_sex.py \
    --train_csv DATA/RAECMBD_454_20241226-163036.csv \
    --desc_file DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt \
    --mlm_epochs 5 \
    --fine_tune_epochs 5 \
    --batch_size 16 \
    --max_clusters 10

Dependencies:
  pip install transformers datasets torch scikit-learn pandas numpy tqdm
"""

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Hugging Face Transformers
from transformers import (
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# For Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

#############################################################################
# Dictionary Loader
#############################################################################

def load_code_descriptors(desc_file):
    """
    Reads something like 'icd10cm-codes-April-2025.txt' with lines:
      F11.2    Opioid dependence
      F14.9    Cocaine use, unspecified
      ...
    Returns { 'F11.2': 'Opioid dependence', 'F14.9': 'Cocaine use, unspecified', ... }
    """
    code2desc = {}
    with open(desc_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # split on first whitespace
            if len(parts) < 2:
                continue
            code = parts[0].strip()
            desc = parts[1].strip()
            code2desc[code] = desc
    return code2desc

#############################################################################
# 1) SingleRowEHRDataset
#############################################################################

class SingleRowEHRDataset(Dataset):
    """
    Reads a CSV with co-occurring diagnoses across columns Diagnóstico 1..20.
    Maps codes to their descriptors, building a single text string for each patient.
    """
    def __init__(self, df, desc_file, max_length=256):
        # df is already filtered to males or females as needed
        df = df.reset_index(drop=True)

        if "ID" not in df.columns:
            df["ID"] = df.index + 1

        self.df = df
        self.code2desc = load_code_descriptors(desc_file)

        # Build text lines from Diagnóstico 1..20
        self.texts = []
        for _, row in df.iterrows():
            diag_codes = []
            for i in range(1, 21):
                col_name = f"Diagnóstico {i}"
                if col_name in row and pd.notna(row[col_name]):
                    diag_codes.append(str(row[col_name]).strip())

            # Convert each ICD code to a textual descriptor
            desc_list = [self.code2desc.get(code, "NOCODE") for code in diag_codes]
            if not desc_list:
                desc_list = ["NOCODE"]

            # Join all descriptors into a single string
            text_line = " ".join(desc_list)
            self.texts.append(text_line)

        self.ids = df["ID"].tolist()

        # Initialize DeBERTa tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.texts[idx]

#############################################################################
# 2) MLMDataset for masked language modeling
#############################################################################

class MLMDataset(Dataset):
    """
    Tokenizes the text lines for MLM. Each item is a dict with input_ids & attention_mask.
    """
    def __init__(self, raw_dataset, max_length=256):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = raw_dataset.tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        text = self.raw_dataset[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

#############################################################################
# 3) MLM Pre-training
#############################################################################

def mlm_pretrain(dataset, output_dir="mlm_pretrain", epochs=1, batch_size=8, lr=5e-5):
    config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-base")
    model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v3-base", config=config)

    tokenizer = dataset.tokenizer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10000,
        save_total_limit=1,
        logging_steps=100,
        learning_rate=lr
    )

    n = len(dataset)
    if n < 2:
        # If extremely small, skip training or do minimal
        print(f"Warning: dataset too small (n={n}). Skipping MLM pretraining.")
        return None

    n_train = int(n * 0.9)
    n_eval = n - n_train
    train_ds, eval_ds = random_split(dataset, [n_train, n_eval])

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds
    )
    trainer.train()

    # Save model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer

#############################################################################
# 4) Contrastive Fine-tuning
#############################################################################

class ContrastiveDataset(Dataset):
    """
    We do a naive data augmentation approach: each text is turned into two random
    'perturbations' for the contrastive objective (shuffle or remove tokens).
    """
    def __init__(self, raw_dataset, max_length=256):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = raw_dataset.tokenizer
        self.max_length = max_length

    def random_perturb(self, text):
        tokens = text.split()
        # remove up to 2 tokens if length permits
        if len(tokens) > 3:
            n_drop = min(2, len(tokens)//3)
            for _ in range(n_drop):
                idx = random.randint(0, len(tokens)-1)
                tokens.pop(idx)
        random.shuffle(tokens)
        return " ".join(tokens)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        base_text = self.raw_dataset[idx]
        text1 = self.random_perturb(base_text)
        text2 = self.random_perturb(base_text)

        enc1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        enc2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids1": enc1["input_ids"].squeeze(0),
            "attention_mask1": enc1["attention_mask"].squeeze(0),
            "input_ids2": enc2["input_ids"].squeeze(0),
            "attention_mask2": enc2["attention_mask"].squeeze(0),
        }

class ContrastiveModel(nn.Module):
    def __init__(self, pretrained_dir):
        super().__init__()
        # Load the MLM-pretrained DeBERTa
        self.deberta = DebertaV2ForMaskedLM.from_pretrained(pretrained_dir)
        hidden_size = self.deberta.config.hidden_size
        # A projection layer to get the final embedding
        self.proj = nn.Linear(hidden_size, hidden_size)

    def encode(self, input_ids, attention_mask):
        # Pass through DeBERTa but only use the base network (self.deberta.deberta)
        outputs = self.deberta.deberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # shape [batch, seq_len, hidden]

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = last_hidden * mask
        sum_hidden = torch.sum(masked_hidden, dim=1)
        lengths = torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9).unsqueeze(-1)
        mean_hidden = sum_hidden / lengths
        emb = self.proj(mean_hidden)
        return emb

    def forward(self, batch):
        emb1 = self.encode(batch["input_ids1"], batch["attention_mask1"])
        emb2 = self.encode(batch["input_ids2"], batch["attention_mask2"])
        return emb1, emb2

def nt_xent_loss(emb1, emb2, temperature=0.05):
    """
    Normalized temperature-scaled cross entropy loss for contrastive learning.
    """
    batch_size = emb1.size(0)
    emb1 = nn.functional.normalize(emb1, dim=-1)
    emb2 = nn.functional.normalize(emb2, dim=-1)
    logits = torch.matmul(emb1, emb2.t()) / temperature
    labels = torch.arange(batch_size, device=emb1.device)
    loss1 = nn.functional.cross_entropy(logits, labels)
    loss2 = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5*(loss1 + loss2)

def contrastive_fine_tune(pretrained_dir, dataset, output_dir="contrastive_model",
                          epochs=1, batch_size=8, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Contrastive fine-tune device:", device)

    if not os.path.exists(pretrained_dir) or not os.path.isdir(pretrained_dir):
        print(f"Warning: pretrained_dir '{pretrained_dir}' not found. Skipping contrastive.")
        return None

    n = len(dataset)
    if n < 2:
        print(f"Warning: dataset too small (n={n}). Skipping contrastive fine-tuning.")
        return None

    model = ContrastiveModel(pretrained_dir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        total_loss = 0.0
        model.train()
        for b in tqdm(loader, desc=f"Contrastive epoch {ep}"):
            for k in b:
                b[k] = b[k].to(device)
            emb1, emb2 = model(b)
            loss = nt_xent_loss(emb1, emb2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep}, contrastive loss={avg_loss:.4f}")

    # Save final model artifacts
    os.makedirs(output_dir, exist_ok=True)
    model.deberta.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "contrastive_model.bin"))
    return model

#############################################################################
# 5) Generate Embeddings
#############################################################################

def generate_embeddings(model_dir, dataset, batch_size=8):
    """
    Runs the ContrastiveModel in inference mode to get a single embedding per row/patient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_dir) or not os.path.isfile(os.path.join(model_dir, "contrastive_model.bin")):
        print(f"Warning: no contrastive model file in '{model_dir}'. Skipping embedding generation.")
        return None

    model = ContrastiveModel(model_dir)
    model.load_state_dict(torch.load(os.path.join(model_dir, "contrastive_model.bin")))
    model = model.to(device)
    model.eval()

    raw_data = dataset.raw_dataset
    if len(raw_data) == 0:
        return None

    results = []
    loader = DataLoader(range(len(raw_data)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for idxs in tqdm(loader, desc="Generating embeddings"):
            input_ids_list = []
            attention_mask_list = []
            for i in idxs:
                text = raw_data[i]
                enc = dataset.tokenizer(
                    text,
                    max_length=dataset.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids_list.append(enc["input_ids"])
                attention_mask_list.append(enc["attention_mask"])

            input_ids_batch = torch.cat(input_ids_list, dim=0).to(device)
            attention_mask_batch = torch.cat(attention_mask_list, dim=0).to(device)

            emb = model.encode(input_ids_batch, attention_mask_batch).cpu().numpy()
            for k, iidx in enumerate(idxs):
                results.append((iidx, emb[k]))

    # Sort by original index, then build a single array
    results.sort(key=lambda x: x[0])
    embeddings = np.array([r[1] for r in results])
    return embeddings

#############################################################################
# 6) Bootstrap for K
#############################################################################

def bootstrap_find_k(embeddings, max_k=10, n_bootstrap=5, sample_frac=0.01):
    """
    Uses a simple bootstrap-based silhouette score to pick an optimal K.
    """
    best_k = None
    best_score = -9999
    k2scores = {k: [] for k in range(2, max_k+1)}
    n_total = embeddings.shape[0]

    for _ in range(n_bootstrap):
        n_sample = max(2, int(n_total * sample_frac))
        idxs = np.random.choice(n_total, n_sample, replace=False)
        sub_emb = embeddings[idxs]
        for k in range(2, max_k+1):
            ac = AgglomerativeClustering(n_clusters=k)
            labels = ac.fit_predict(sub_emb)
            # Only compute silhouette if more than one cluster exists
            if len(np.unique(labels)) > 1:
                score = silhouette_score(sub_emb, labels)
                k2scores[k].append(score)

    for k in range(2, max_k+1):
        if k2scores[k]:
            mean_score = np.mean(k2scores[k])
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
    print(f"Chosen best_k={best_k} by silhouette (score={best_score:.4f})")
    return best_k

#############################################################################
# 7) K-Means
#############################################################################

def run_kmeans(embeddings, k):
    """
    K-Means clustering with multiple (10) random inits.
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels

#############################################################################
# 8) c-DF-IPF
#############################################################################

def compute_cdfipf(df, diag_col="Diagnóstico 1", cluster_col="cluster"):
    """
    For each cluster, compute cluster-level Disease Frequency – Inverse Patient Frequency.
    This is analogous to TF-IDF for diagnoses across clusters.
    """
    df = df.copy()
    all_codes = set()
    pid2codes = {}

    # Build a set of codes assigned to each patient
    for idx, row in df.iterrows():
        pid = row["ID"]  # Must exist in df
        diag_str = row.get(diag_col, "")
        codes = set(x.strip() for x in diag_str.split(",") if x.strip())
        if not codes:
            codes = {"NOCODE"}
        pid2codes[pid] = codes
        all_codes.update(codes)

    n_patients = df["ID"].nunique()
    # cluster -> list of pids
    cluster2pids = {}
    for idx, row in df.iterrows():
        c = row[cluster_col]
        pid = row["ID"]
        cluster2pids.setdefault(c, []).append(pid)

    # Count how many patients have each code
    code2count = {c: 0 for c in all_codes}
    for pid, cset in pid2codes.items():
        for cd in cset:
            code2count[cd] += 1

    # Inverse Patient Frequency: log(total patients / code frequency)
    code2ipf = {}
    for c in all_codes:
        nd = code2count[c]
        code2ipf[c] = math.log(n_patients / nd) if nd > 0 else 0

    # For each cluster, compute average IPF for codes present
    cluster2dfipf = {}
    for c, pids in cluster2pids.items():
        sums = {cd: 0.0 for cd in all_codes}
        for pid in pids:
            for cd in pid2codes[pid]:
                sums[cd] += code2ipf[cd]
        csize = len(pids)
        if csize == 0:
            cluster2dfipf[c] = sums
            continue
        for cd in sums:
            sums[cd] /= csize
        cluster2dfipf[c] = sums

    return cluster2dfipf

#############################################################################
# Pipeline to run for one sex
#############################################################################

def run_pipeline_for_subset(df_subset, desc_file, mlm_epochs, fine_tune_epochs,
                            batch_size, max_clusters, output_prefix):
    """
    Runs MLM pre-training, contrastive fine-tuning, embedding generation, clustering,
    and c-DF-IPF on a subset DataFrame (all males or all females).
    Saves final results to 'clustered_patients_<output_prefix>.csv'.
    """
    if len(df_subset) < 2:
        print(f"\n[SKIP] Subset '{output_prefix}' has < 2 rows. No clustering performed.")
        return

    # 1) Create dataset for this subset
    dataset = SingleRowEHRDataset(df_subset, desc_file=desc_file, max_length=256)

    # 2) MLM Pre-training
    print(f"\n=== MLM Pre-training for subset '{output_prefix}' ===")
    mlm_out_dir = f"mlm_pretrain_{output_prefix}"
    mlm_trainer = mlm_pretrain(
        MLMDataset(dataset),
        output_dir=mlm_out_dir,
        epochs=mlm_epochs,
        batch_size=batch_size,
        lr=5e-5
    )
    if not mlm_trainer:
        print(f"No MLM model trained for '{output_prefix}' (likely not enough samples).")
        return

    # 3) Contrastive Fine-tuning
    print(f"\n=== Contrastive Fine-tuning for subset '{output_prefix}' ===")
    contrastive_out_dir = f"contrastive_model_{output_prefix}"
    c_dataset = ContrastiveDataset(dataset)
    c_model = contrastive_fine_tune(
        mlm_out_dir,
        c_dataset,
        output_dir=contrastive_out_dir,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        lr=1e-5
    )
    if not c_model:
        print(f"No contrastive model trained for '{output_prefix}'.")
        return

    # 4) Generate embeddings
    print(f"\n=== Generating embeddings for subset '{output_prefix}' ===")
    emb = generate_embeddings(contrastive_out_dir, c_dataset, batch_size=batch_size)
    if emb is None:
        print(f"No embeddings generated for '{output_prefix}'.")
        return
    print(f"Embeddings shape for '{output_prefix}' = {emb.shape}")

    # 5) Bootstrap for K
    print(f"\n=== Bootstrapping K for subset '{output_prefix}' ===")
    best_k = bootstrap_find_k(emb, max_k=max_clusters, n_bootstrap=5, sample_frac=0.01)
    print(f"Chosen best_k = {best_k} for '{output_prefix}'")

    # 6) K-Means
    print(f"\n=== K-Means Clustering for subset '{output_prefix}' ===")
    labels = run_kmeans(emb, best_k)
    print(f"Assigned {len(labels)} patients to {best_k} clusters for '{output_prefix}'.")

    # Store cluster labels in df_subset
    df_subset = df_subset.copy()
    df_subset["cluster"] = labels

    # 7) c-DF-IPF
    print(f"\n=== c-DF-IPF for subset '{output_prefix}' ===")
    cluster2dfipf = compute_cdfipf(df_subset, diag_col="Diagnóstico 1", cluster_col="cluster")

    for c in sorted(cluster2dfipf.keys()):
        code_map = cluster2dfipf[c]
        sorted_items = sorted(code_map.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_items[:5]
        print(f"\nCluster {c} top 5 codes by c-DF-IPF (subset '{output_prefix}'):")
        for cd, val in top5:
            print(f"  {cd}: {val:.4f}")

    # 8) Save output
    out_file = f"clustered_patients_{output_prefix}.csv"
    df_subset.to_csv(out_file, index=False)
    print(f"\nClustering complete for '{output_prefix}'. Results saved in '{out_file}'.")


#############################################################################
# MAIN
#############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True,
                        help="CSV with Diagnóstico columns (Diagnóstico 1..20), including 'Sexo'.")
    parser.add_argument("--desc_file", required=True,
                        help="ICD code descriptors file (CODE [whitespace] DESC).")
    parser.add_argument("--mlm_epochs", type=int, default=5,
                        help="MLM pre-training epochs.")
    parser.add_argument("--fine_tune_epochs", type=int, default=5,
                        help="Contrastive fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--max_clusters", type=int, default=10,
                        help="Max number of clusters to try in bootstrap.")
    args = parser.parse_args()

    # Read the entire dataset
    df_full = pd.read_csv(args.train_csv, low_memory=True)
    
    # -----------------------------------------------------------------------
    # Minimal fix: ensure df_full has "ID" before we do anything else
    # -----------------------------------------------------------------------
    if "ID" not in df_full.columns:
        df_full["ID"] = df_full.index + 1
    
    print(f"Loaded {len(df_full)} rows from '{args.train_csv}'.")

    # Separate into males (Sexo=1) and females (Sexo=2)
    df_males = df_full[df_full["Sexo"] == 1].copy()
    df_females = df_full[df_full["Sexo"] == 2].copy()

    print(f"Males subset: {len(df_males)} rows.  Females subset: {len(df_females)} rows.")

    # Run pipeline for males
    run_pipeline_for_subset(
        df_subset=df_males,
        desc_file=args.desc_file,
        mlm_epochs=args.mlm_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        max_clusters=args.max_clusters,
        output_prefix="male"
    )

    # Run pipeline for females
    run_pipeline_for_subset(
        df_subset=df_females,
        desc_file=args.desc_file,
        mlm_epochs=args.mlm_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        max_clusters=args.max_clusters,
        output_prefix="female"
    )

    print("\nAll done. Two clustering outputs were produced, one for males and one for females.")


if __name__ == "__main__":
    main()
