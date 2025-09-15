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
CVE2ATTACK/

├─ data/ # datasets

│ ├─ cve_clean.csv # cleaned CVE descriptions
│ ├─ train.csv # training split
│ ├─ val.csv # validation split
│ └─ test.csv # test split
│
├─ models/ # saved fine-tuned model checkpoints
│ ├─ ep2_bs64/ # model after 2 epochs, batch size 64
│ ├─ ep3_bs32/ # model after 3 epochs, batch size 32
│ └─ ep4_bs64/ # model after 4 epochs, batch size 64
│
├─ results/ # evaluation outputs
│ ├─ metrics/ # metrics JSON/CSV for each run
│ ├─ plots/ # confusion maps and visualizations
│ ├─ y_prob.npy # cached model probabilities (N × L)
│ └─ y_true.npy # ground-truth labels (N × L)
│
├─ scripts/ # main project scripts
│ ├─ split_dataset.py # prepare train/val/test splits
│ ├─ train.py # fine-tune BERT model
│ ├─ infer_save.py # run inference and save predictions
│ ├─ eval.py # compute metrics (+ threshold sweep)
│ └─ confusion.py # generate confusion heatmaps
│
├─ requirements.txt # Python dependencies
└─ README_en.md

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
