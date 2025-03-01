# 🧠 LLM-Based Clustering of Adolescent Mental Health Data

## 📌 Overview
This project applies **Large Language Models (LLMs)** and **unsupervised clustering** to analyze adolescent psychiatric disorders in Spain using hospital discharge records.

### **Why This Matters?**
- **40% of adolescent psychiatric admissions** in Spain involve substance use disorders.
- **Mental health conditions co-occur**, forming distinct patient subgroups.
- **Clustering can uncover hidden patterns**, leading to **better intervention strategies**.

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
2️⃣ Install Dependencies
On EC2 GPU Instance:
```bash
sudo apt update && sudo apt install -y python3-pip
pip install torch torchvision torchaudio transformers datasets scikit-learn pandas numpy tqdm accelerate
```
3️⃣ Run the Training Script
```bash
python3 PYTHON/00-llm_cluster_pipeline.py \
  --train_csv DATA/RAECMBD_454_20241226-163036.csv \
  --desc_file DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt \
  --mlm_epochs 2 \
  --fine_tune_epochs 2 \
  --batch_size 8 \
  --max_clusters 10
```
4️⃣ Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```
📊 Research Methods
Data Source: Spanish National Registry of Hospital Discharges (RAE-CMBD)
Language Model: DeBERTa (microsoft/deberta-v3-base)
Clustering Method: K-Means with bootstrapped silhouette analysis
Evaluation Metrics: Cluster coherence, disease-specific feature weighting

📜 Citation
If you use this work, please cite:
Manuel Corpas, "LLM-Based Clustering of Adolescent Mental Health Data", 2025.
📬 Contact
For questions or collaborations: 📧 mc@manuelcorpas.com
🔗 GitHub Profile: https://github.com/manuelcorpas
