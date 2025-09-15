import json, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

def build_labels(train_df, val_df, test_df):
    all_tids = pd.concat([train_df, val_df, test_df])["technique_id"].unique()
    labels = sorted(all_tids)
    tid2idx = {t:i for i,t in enumerate(labels)}
    return labels, tid2idx

def to_multihot(df, tid2idx, L):
    def one(row):
        v = np.zeros(L, dtype=np.float32)
        v[tid2idx[row["technique_id"]]] = 1.0
        return v
    df = df.copy()
    df["labels"] = df.apply(one, axis=1)
    return df

class WeightedTrainer(Trainer):
    def __init__(self, manual_wt=0.8, label_smooth=0.05, **kw):
        super().__init__(**kw)
        self.manual_wt = manual_wt
        self.label_smooth = label_smooth

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        wflag  = inputs.pop("src_flag", None)

        if wflag is None:
            weight = torch.ones(labels.size(0), device=labels.device)
        else:
            weight = torch.tensor(wflag, dtype=torch.float, device=labels.device) * self.manual_wt + (1 - self.manual_wt)

        labels_smooth = labels * (1 - self.label_smooth) + 0.5 * self.label_smooth
        outputs = model(**inputs)
        loss = torch.nn.BCEWithLogitsLoss(weight=weight.unsqueeze(1))(outputs.logits, labels_smooth)
        return (loss, outputs) if return_outputs else loss

def main(args):
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    labels, tid2idx = build_labels(train_df, val_df, test_df)
    L = len(labels)
    train_df = to_multihot(train_df, tid2idx, L)
    val_df   = to_multihot(val_df,   tid2idx, L)
    test_df  = to_multihot(test_df,  tid2idx, L)

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        enc = tok(batch["description"], truncation=True, padding="max_length", max_length=args.max_len)
        enc["labels"] = batch["labels"]
        if "label_source" in batch:
            enc["src_flag"] = [1 if s == "manual" else 0 for s in batch["label_source"]]
        return enc

    train_ds = Dataset.from_pandas(train_df[["description","labels","label_source"]]) \
                      .map(tokenize, batched=True, remove_columns=["description","label_source"])
    # валидируем только на «золотых»
    val_man = val_df[val_df["label_source"] == "manual"] if "label_source" in val_df else val_df
    val_ds  = Dataset.from_pandas(val_man[["description","labels"]]) \
                     .map(tokenize, batched=True, remove_columns=["description"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=L,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )

    use_fp16 = torch.cuda.is_available() and args.fp16
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        dataloader_num_workers=args.num_workers,
        evaluation_strategy="no" if args.fast else "steps",
        eval_steps=args.eval_steps,
        save_strategy="no" if args.fast else "steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=not args.fast,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        report_to="none",
    )

    trainer = WeightedTrainer(
        manual_wt=args.manual_wt,
        label_smooth=args.label_smooth,
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok
    )
    trainer.train()

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out)
    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    print(f"✓ saved model to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_name", default="bert-base-cased")
    p.add_argument("--output_dir", default="out")
    p.add_argument("--save_dir", default="models/final_model")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--manual_wt", type=float, default=0.8)
    p.add_argument("--label_smooth", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=200)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--fast", action="store_true", help="без eval/save по ходу (быстрее)")
    main(p.parse_args())
