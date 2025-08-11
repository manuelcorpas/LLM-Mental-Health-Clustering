# 🧠 LLM-Based Clustering of Adolescent Mental Health Data

## 📌 Overview
This project applies **Large Language Models (LLMs)** and **unsupervised clustering** to analyze adolescent psychiatric disorders in Spain using hospital discharge records from the Spanish National Registry of Hospital Discharges (RAE-CMBD).

### **Why This Matters**
- **40% of adolescent psychiatric admissions** in Spain involve substance use disorders (SUD).
- **Mental health conditions frequently co-occur**, forming distinct patient subgroups with different risk profiles and treatment needs.
- **Clustering can uncover hidden patterns**, leading to **better targeted interventions** and more efficient allocation of resources.

---

## 📂 Project Structure
```
📁 LLM-Mental-Health-Clustering
├── 📂 PYTHON/ → Python scripts for training, clustering, and embedding generation
├── 📂 SH/ → Shell scripts for automation
├── 📂 PDF/ → References and related research
├── 📂 model_checkpoints/ → Saved model embeddings
├── 📜 README.md → Documentation and reproducibility guide
└── 🛑 .gitignore → Excludes sensitive data (DATA/)
```

---

## 🚀 **How to Reproduce the Study**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/manuelcorpas/LLM-Mental-Health-Clustering.git
cd LLM-Mental-Health-Clustering
```

### **2️⃣ Install Dependencies**
On an EC2 GPU Instance or local GPU machine:
```bash
sudo apt update && sudo apt install -y python3-pip
pip install torch torchvision torchaudio transformers datasets scikit-learn pandas numpy tqdm accelerate
```

### **3️⃣ Run the Training Script**
```bash
python3 PYTHON/00-llm_cluster_pipeline.py   --train_csv DATA/RAECMBD_454_20241226-163036.csv   --desc_file DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt   --mlm_epochs 2   --fine_tune_epochs 2   --batch_size 8   --max_clusters 10
```

### **4️⃣ Monitor GPU Usage**
```bash
watch -n 1 nvidia-smi
```

---

## 📊 Research Methods

* **Data Source:** Spanish National Registry of Hospital Discharges (RAE-CMBD)
* **Language Model:** DeBERTa-v3 (microsoft/deberta-v3-base)
* **Training Strategy:**  
  - Masked Language Modelling (MLM) pre-training on sex-specific diagnostic text  
  - Contrastive fine-tuning for embedding stability  
* **Clustering Method:** K-Means with bootstrapped silhouette analysis
* **Evaluation Metrics:** Cluster coherence, disease-specific feature weighting (c-DF–IPF)

---

## 📜 Citation

If you use this work, please cite:

> Corpas M, Blasco H, Gallego L, Ramos-Rincón JM, Moreno-Torres V, Soriano V.  
> Sex-Specific Diagnostic Subtypes in Adolescents Hospitalized for Substance Use Disorders Revealed by Transformer-Based Clustering.  
> *medRxiv* 2025. doi:[10.1101/2025.08.06.25333108](https://doi.org/10.1101/2025.08.06.25333108)  
> Posted August 10, 2025. This preprint is made available under a CC-BY 4.0 International license.

---

## 📂 Data and Code Availability

All data processing scripts, fine-tuning procedures, and clustering pipelines are available in this repository.  

The dataset (anonymized and pre-processed) for males and females, along with predicted cluster assignments, can be found in the `DATA/` directory (not tracked in GitHub due to privacy rules).  
Access to the raw RAE-CMBD dataset must be requested directly from the Spanish Ministry of Health.

**Repository contents:**
- Python scripts for:
  - ICD-10 to descriptor mapping
  - Transformer-based embedding generation (DeBERTa-v3)
  - Contrastive fine-tuning
  - K-Means clustering and silhouette bootstrapping
  - PCA and UMAP visualizations
- Supplementary tables and cluster descriptors in the `PDF/` folder
- Model checkpoints in `model_checkpoints/`

---


---

## 📁 RESULTS Overview

| File | Purpose | Links to Preprint |
|------|---------|-------------------|
| `RESULTS/FIGS/` | Figures 1–9 for manuscript | Matches figure captions in preprint |
| `RESULTS/clustered_patients_female.csv` | Cluster assignments for female patients | Methods section & Results, Female clusters |
| `RESULTS/clustered_patients_male.csv` | Cluster assignments for male patients | Methods section & Results, Male clusters |
| `RESULTS/supplementary_table_1_sud_by_sex.csv` | Summary of SUD distribution by sex | Supplementary Table S1 |
| `RESULTS/supplementary_table_2_all_clusters.docx` | All clusters with ICD-10 and semantic terms | Supplementary Table S3 |
| `RESULTS/supplementary_table_2_all_clusters_paired.docx` | Cluster pairs for comparison | Supplementary data |
| `RESULTS/supplementary_table_2_paired_icd_terms.docx` | ICD term mapping for paired clusters | Supplementary data |
| `RESULTS/supplementary_table_2_word_format.docx` | Word-formatted version of cluster table | Supplementary data |
| `RESULTS/supplementary_table_word_format.docx` | Alternate Word format | Supplementary data |

---

## 📄 License

- **Code:** MIT License  
- **Manuscript & figures:** CC-BY 4.0 International License

---

## 📬 Contact

For questions or collaborations:  
📧 mc@manuelcorpas.com  
🔗 [GitHub Profile](https://github.com/manuelcorpas)
