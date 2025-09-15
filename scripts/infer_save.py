import argparse, json, numpy as np, pandas as pd, torch
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main(args):
    model_dir = Path(args.model_dir)
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    tid2idx = {t:i for i,t in enumerate(labels)}
    L = len(labels)

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=args.local_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=args.local_only).eval()
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model.to(device)

    test = pd.read_csv(args.test_file)
    group_texts  = defaultdict(list)
    group_labels = defaultdict(lambda: np.zeros(L, dtype=np.float32))
    for _, row in test.iterrows():
        cid = row["cve_id"]
        group_texts[cid].append(row["description"])
        if row["technique_id"] in tid2idx:
            group_labels[cid][tid2idx[row["technique_id"]]] = 1.0

    cve_ids = list(group_texts)

    def encode(batch):
        return tok(batch, truncation=True, padding="max_length",
                   max_length=args.max_len, return_tensors="pt")

    probs, trues = [], []
    with torch.no_grad():
        for cid in cve_ids:
            descs = group_texts[cid]
            chunk_logits = []
            for i in range(0, len(descs), args.batch):
                batch = encode(descs[i:i+args.batch]).to(device)
                out = model(**batch).logits.sigmoid().cpu().numpy()
                chunk_logits.append(out)
            probs.append(np.concatenate(chunk_logits, 0).max(0))  # max-pool
            trues.append(group_labels[cid])

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "y_prob.npy", np.stack(probs))
    np.save(outdir / "y_true.npy", np.stack(trues))
    (outdir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    (outdir / "cve_ids.json").write_text(json.dumps(cve_ids, ensure_ascii=False))
    print(f"✓ saved to {outdir} (y_prob.npy / y_true.npy)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/final_model")
    p.add_argument("--test_file", default="data/test.csv")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--local_only", action="store_true")
    main(p.parse_args())
