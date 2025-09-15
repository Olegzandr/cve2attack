# CVE → MITRE ATT&CK Mapping with BERT

Research project that maps CVE descriptions to MITRE ATT&CK techniques using a BERT-based multi-label classifier.

---

## Overview
- Task: multi-label text classification (one CVE can map to several techniques).
- Data: hybrid corpus of CVE descriptions with **manual** labels (high-trust) and **pseudo-labels** (noisy).
- Model: `bert-base-cased` fine-tuned with a weighted BCE loss (manual > pseudo) and light label smoothing.
- Evaluation: grouped **per-CVE** (aggregate multiple descriptions per CVE with max-pool), reporting Micro-/Macro-F1, Precision@k and mAP.

---

## Repository Structure
```CVE2ATTACK/
├─ data/ # lightweight CSVs for quick tests (full dataset on Google Drive)
│ ├─ cve_clean.csv
│ ├─ train.csv
│ ├─ val.csv
│ └─ test.csv
│
├─ results/ # evaluation outputs
│ ├─ metrics/ # JSON with grouped metrics per experiment
│ │ ├─ test_metrics_grouped_ep2_bs64.json
│ │ ├─ test_metrics_grouped_ep3_bs32.json
│ │ └─ test_metrics_grouped_ep4_bs64.json
│ ├─ plots/ # confusion maps and visualizations
│ │ ├─ ep2_bs64_confusion_th015.png
│ │ ├─ ep3_bs32_confusion_th015.png
│ │ └─ ep4_bs64_fast_confusion_th015.png
│ ├─ y_prob.npy # cached model probabilities (N × L)
│ └─ y_true.npy # ground-truth labels (N × L)
│
├─ scripts/ # main project scripts
│ ├─ split_dataset.py # prepare train/val/test splits
│ ├─ train.py # fine-tune BERT
│ ├─ infer_save.py # run inference and save predictions
│ ├─ eval.py # compute metrics (+ threshold sweep)
│ └─ confusion.py # generate confusion heatmaps
│
├─ .gitignore
├─ requirements.txt
└─ README.md
```
## External Resources

- 📂 **Full dataset (CVE corpus)** → [Google Drive link](https://drive.google.com/drive/folders/17yBEvfqLKrkmIus4hkptbWa9_paflK_x?usp=sharing)  
- 📂 **Fine-tuned model checkpoints** → [Google Drive link](https://drive.google.com/drive/folders/17yBEvfqLKrkmIus4hkptbWa9_paflK_x?usp=sharing) 
---

## Quick Start
```bash
# 1) install deps
pip install -r requirements.txt

# 2) (optional) create splits from the cleaned corpus
python scripts/split_dataset.py

# 3) train (example: 2 epochs, batch 64)
python scripts/train.py --data_dir data --save_dir models/ep2_bs64 --epochs 2 --batch_size 64

# 4) run inference and cache probabilities
python scripts/infer_save.py --model_dir models/ep2_bs64 --test_file data/test.csv --out_dir results

# 5) compute metrics (fixed threshold or sweep)
python scripts/eval.py --results_dir results --threshold 0.15 --sweep

# 6) visualize confusion map
python scripts/confusion.py --results_dir results --threshold 0.15
```
---

## Notes on Evaluation

- Metrics are computed **per CVE**: all descriptions for the same CVE are encoded and **max-pooled**.  
- Threshold (`--threshold`) affects F1: sweep with `--sweep` to find a reasonable operating point.  
- Confusion maps highlight frequent confusions (e.g., overlapping *Execution* / *Phishing* patterns).  

---

## What Works / Limitations

- ✅ Works out of the box for English CVE descriptions, including long sequences (truncated to 256 tokens by default).  
- ⚠️ Pseudo-labels introduce noise; weighting helps but does not eliminate it.  
- 📉 Results depend on thresholding; consider **Precision@k** for ranking-based use cases.  

---

## Future Work

- 🔹 Expand with mobile/ICS CVE corpora.  
- 🔹 Multi-task learning (predict tactic → technique).  
- 🔹 Knowledge distillation to a lighter model for Edge/EDR deployment.  
