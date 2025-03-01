#!/usr/bin/env python3
"""
00-llm_cluster_pipeline.py

A single script that:
1) Reads your CSV (one row per patient, with DIAGNOSTICO_PRINCIPAL).
2) Loads ICD-10-CM code-descriptors from a text file, e.g.:
      DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt
   which might have lines like:
      F11.2    Opioid dependence
      F14.9    Cocaine use, unspecified
   ...
3) For each row, we parse the codes in DIAGNOSTICO_PRINCIPAL,
   and map each to its descriptor, building a final text string
   that includes only the medical terms (no raw code).
   e.g.: "Opioid dependence Cocaine use, unspecified"
4) Trains a DeBERTa model on masked language modeling (MLM).
5) Does a simplified contrastive fine-tuning (DiffCSE-like).
6) Generates embeddings for each row/patient.
7) Bootstraps to find the best number of clusters.
8) Clusters with K-Means and runs c-DF-IPF for cluster-specific codes.

Usage example:
  python 00-llm_cluster_pipeline.py --train_csv PATH.csv --desc_file PATH.txt \
    --mlm_epochs 2 --fine_tune_epochs 2 --batch_size 8 --max_clusters 10

Requires:
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
from torch.utils.data import Dataset, DataLoader
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
            # Suppose there's a tab or multiple spaces between code and descriptor
            parts = line.split(None, 1)  # split on first whitespace
            if len(parts) < 2:
                continue
            code = parts[0].strip()
            desc = parts[1].strip()
            code2desc[code] = desc
    return code2desc

#############################################################################
# 1. Dataset for reading single-row-per-patient
#############################################################################

class SingleRowEHRDataset(Dataset):
    """
    Reads a CSV with at least 'DIAGNOSTICO_PRINCIPAL'.
    Optionally uses 'ID' if present. If not, we create a fake ID from the row index.

    We'll store each row's diagnoses as a single text string *only using the descriptor*,
    e.g. "Opioid dependence Cocaine use, unspecified" (with no codes).
    """
    def __init__(self, csv_file, desc_file, max_length=256):
        # 1) Read CSV
        df = pd.read_csv(csv_file, low_memory=False)
        df = df.reset_index(drop=True)

        # 2) If no 'ID' column, create one
        if "ID" not in df.columns:
            df["ID"] = df.index + 1

        self.df = df

        # 3) Load code->descriptor dictionary
        self.code2desc = load_code_descriptors(desc_file)

        # 4) For each row, parse DIAGNOSTICO_PRINCIPAL, map to descriptors only
        self.texts = []
        for idx, row in df.iterrows():
            diag_str = row.get("DIAGNOSTICO_PRINCIPAL", "")
            if pd.isna(diag_str):
                diag_str = ""
            # parse codes
            codes = [c.strip() for c in diag_str.split(",") if c.strip()]

            # build text of descriptors only
            desc_list = []
            for code in codes:
                # If the code is known, use the descriptor; else fallback
                if code in self.code2desc:
                    desc_list.append(self.code2desc[code])
                else:
                    desc_list.append("NOCODE")  # or skip it
            if not desc_list:
                desc_list = ["NOCODE"]
            text_line = " ".join(desc_list)
            self.texts.append(text_line)

        # 5) Keep track of IDs
        self.ids = df["ID"].tolist()

        # 6) DeBERTa V3 Base tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.texts[idx]

    def get_id(self, idx):
        return self.ids[idx]


#############################################################################
# 2. MLM Dataset wrapper
#############################################################################

class MLMDataset(Dataset):
    """
    Wraps SingleRowEHRDataset, performing tokenization for MLM.
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
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


#############################################################################
# 3. MLM Pre-training
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

    from torch.utils.data import random_split
    n = len(dataset)
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

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


#############################################################################
# 4. Simplified Contrastive Fine-tuning
#############################################################################

class ContrastiveDataset(Dataset):
    """
    We'll do a naive approach: for each text, produce 2 random augmentations.
    """
    def __init__(self, raw_dataset, max_length=256):
        self.raw_dataset = raw_dataset
        self.tokenizer = raw_dataset.tokenizer
        self.max_length = max_length

    def random_perturb(self, text):
        tokens = text.split()
        if len(tokens) > 3:
            # randomly remove 1 or 2 tokens
            n_drop = min(2, len(tokens)//3)
            for _ in range(n_drop):
                idx = random.randint(0, len(tokens)-1)
                tokens.pop(idx)
        # shuffle
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
        self.deberta = DebertaV2ForMaskedLM.from_pretrained(pretrained_dir)
        hidden_size = self.deberta.config.hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def encode(self, input_ids, attention_mask):
        outputs = self.deberta.deberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # shape [batch, seq_len, hidden]
        # mean pool
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
    batch_size = emb1.size(0)
    emb1 = nn.functional.normalize(emb1, dim=-1)
    emb2 = nn.functional.normalize(emb2, dim=-1)
    logits = torch.matmul(emb1, emb2.t()) / temperature
    labels = torch.arange(batch_size, device=emb1.device)
    loss1 = nn.functional.cross_entropy(logits, labels)
    loss2 = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5*(loss1+loss2)


def contrastive_fine_tune(pretrained_dir, dataset, output_dir="contrastive_model", epochs=1, batch_size=8, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Contrastive fine-tune device:", device)

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
            loss = nt_xent_loss(emb1, emb2, temperature=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep}, contrastive loss={avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.deberta.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "contrastive_model.bin"))
    return model


#############################################################################
# 5. Generate embeddings for each row
#############################################################################

def generate_embeddings(model_dir, dataset, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveModel(model_dir)
    model.load_state_dict(torch.load(os.path.join(model_dir, "contrastive_model.bin")))
    model = model.to(device)
    model.eval()

    texts = dataset.raw_dataset
    results = []
    loader = DataLoader(range(len(texts)), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for idxs in tqdm(loader, desc="Generating embeddings"):
            input_ids_list = []
            attention_mask_list = []
            for i in idxs:
                text = texts[i]
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

    results.sort(key=lambda x: x[0])
    embeddings = np.array([r[1] for r in results])
    return embeddings


#############################################################################
# 6. Bootstrap for K
#############################################################################

def bootstrap_find_k(embeddings, max_k=10, n_bootstrap=5, sample_frac=0.01):
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
            if len(np.unique(labels)) > 1:
                score = silhouette_score(sub_emb, labels)
                k2scores[k].append(score)

    for k in range(2, max_k+1):
        if len(k2scores[k]) > 0:
            mean_score = np.mean(k2scores[k])
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
    print(f"Chosen best_k={best_k} by silhouette (score={best_score:.4f})")
    return best_k


#############################################################################
# 7. K-Means
#############################################################################

def run_kmeans(embeddings, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels


#############################################################################
# 8. c-DF-IPF
#############################################################################

def compute_cdfipf(df, diag_col="DIAGNOSTICO_PRINCIPAL", cluster_col="cluster"):
    """
    We parse DIAGNOSTICO_PRINCIPAL as codes (raw codes, not descriptors) for c-DF-IPF.
    If you want the final cluster interpretation to mention codes, this is how.
    If you prefer the descriptors at cluster-level, adapt accordingly.
    """
    df = df.copy()
    all_codes = set()
    pid2codes = {}

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
        if c not in cluster2pids:
            cluster2pids[c] = []
        cluster2pids[c].append(pid)

    n_patients = df["ID"].nunique()
    code2count = {c: 0 for c in all_codes}
    for pid in pid2codes:
        for cd in pid2codes[pid]:
            code2count[cd] += 1

    code2ipf = {}
    for c in all_codes:
        nd = code2count[c]
        if nd == 0:
            code2ipf[c] = 0
        else:
            code2ipf[c] = math.log(n_patients / nd)

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


#############################################################################
# MAIN
#############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True,
                        help="CSV with 'DIAGNOSTICO_PRINCIPAL' column.")
    parser.add_argument("--desc_file", type=str, required=True,
                        help="File with lines: CODE [space/tab] DESCRIPTOR")
    parser.add_argument("--mlm_epochs", type=int, default=1,
                        help="Epochs for MLM pre-training.")
    parser.add_argument("--fine_tune_epochs", type=int, default=1,
                        help="Epochs for contrastive fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_clusters", type=int, default=10)
    args = parser.parse_args()

    # 1) Load single-row EHR with descriptor mapping
    raw_dataset = SingleRowEHRDataset(args.train_csv, args.desc_file, max_length=256)
    mlm_dataset = MLMDataset(raw_dataset, max_length=256)

    print("=== Step 1: MLM Pre-training ===")
    mlm_pretrain(
        mlm_dataset,
        output_dir="mlm_pretrain",
        epochs=args.mlm_epochs,
        batch_size=args.batch_size,
        lr=5e-5
    )
    print("MLM pre-training done.")

    print("=== Step 2: Contrastive Fine-tuning ===")
    contrastive_data = ContrastiveDataset(raw_dataset, max_length=256)
    contrastive_fine_tune(
        "mlm_pretrain",
        contrastive_data,
        output_dir="contrastive_model",
        epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        lr=1e-5
    )
    print("Contrastive fine-tuning done.")

    print("=== Step 3: Generate Embeddings ===")
    emb = generate_embeddings("contrastive_model", contrastive_data, batch_size=args.batch_size)
    print(f"Embeddings shape = {emb.shape}")

    print("=== Step 4: Bootstrap for K ===")
    best_k = bootstrap_find_k(emb, max_k=args.max_clusters, n_bootstrap=5, sample_frac=0.01)

    print("=== Step 5: Final K-Means ===")
    labels = run_kmeans(emb, best_k)
    print(f"Assigned {len(labels)} patients to {best_k} clusters.")

    print("=== Step 6: c-DF-IPF ===")
    # We'll add 'cluster' to the raw_dataset.df
    df = raw_dataset.df.copy()
    df["cluster"] = labels

    # compute c-DF-IPF on the raw codes (since DIAGNOSTICO_PRINCIPAL was codes)
    cluster2dfipf = compute_cdfipf(df, diag_col="DIAGNOSTICO_PRINCIPAL", cluster_col="cluster")
    # get top 5 for each cluster
    for c in sorted(cluster2dfipf.keys()):
        code_map = cluster2dfipf[c]
        sorted_items = sorted(code_map.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_items[:5]
        print(f"\nCluster {c} top 5 codes by c-DF-IPF:")
        for cd, val in top5:
            print(f"  {cd}: {val:.4f}")

    # Save final
    df.to_csv("clustered_patients.csv", index=False)
    print("\nDone! Wrote 'clustered_patients.csv' with cluster assignments.")
    print("You can run e.g.:")
    print("  python3.11 00-llm_cluster_pipeline.py --train_csv data.csv --desc_file codes.txt --mlm_epochs 2 --fine_tune_epochs 2 --batch_size 8 --max_clusters 10")


if __name__ == "__main__":
    main()
