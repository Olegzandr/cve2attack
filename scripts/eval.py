import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, average_precision_score

def topk_mask(probs: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-probs, k-1, axis=1)[:, :k]
    mask = np.zeros_like(probs, dtype=int)
    mask[np.arange(probs.shape[0])[:, None], idx] = 1
    return mask

def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # доля ошибочных меток (FP+FN) среди всех меток
    return float((y_true ^ y_pred).sum() / (y_true.size + 1e-12))

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float, ks=(1,)):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "threshold" : thr,
        "Micro-F1"  : f1_score(y_true, y_pred, average="micro",  zero_division=0),
        "Macro-F1"  : f1_score(y_true, y_pred, average="macro",  zero_division=0),
        "Hamming"   : hamming_loss(y_true, y_pred),
        "mAP"       : average_precision_score(y_true, y_prob, average="macro"),
    }
    for k in ks:
        out[f"Precision@{k}"] = precision_score(y_true, topk_mask(y_prob, k), average="micro", zero_division=0)
    return out

def main(args):
    resdir = Path(args.results_dir)
    # можно указать прямые пути к npy, иначе возьмём стандартные имена
    y_prob_path = Path(args.y_prob) if args.y_prob else resdir / "y_prob.npy"
    y_true_path = Path(args.y_true) if args.y_true else resdir / "y_true.npy"

    y_prob = np.load(y_prob_path)
    y_true = np.load(y_true_path)

    # выравнивание на случай несовпадения L
    L = min(y_prob.shape[1], y_true.shape[1])
    y_prob, y_true = y_prob[:, :L], y_true[:, :L]

    ks = tuple(int(k) for k in args.k.split(","))  # например "1,3"

    # sweep по порогу (опционально)
    if args.sweep:
        thrs = np.linspace(args.t_min, args.t_max, args.t_steps)
        rows = [compute_metrics(y_true, y_prob, float(t), ks) for t in thrs]
        df = pd.DataFrame(rows)
        df.to_csv(resdir / f"{args.prefix}metrics_sweep.csv", index=False)
        best_micro = df.iloc[df["Micro-F1"].idxmax()].to_dict()
        best_macro = df.iloc[df["Macro-F1"].idxmax()].to_dict()
        print("Best Micro-F1:", {k: round(v, 6) if isinstance(v, float) else v for k,v in best_micro.items()})
        print("Best Macro-F1:", {k: round(v, 6) if isinstance(v, float) else v for k,v in best_macro.items()})

    # фиксированный порог
    m = compute_metrics(y_true, y_prob, args.threshold, ks)
    pretty = {k: (round(v, 6) if isinstance(v, float) else v) for k,v in m.items()}
    print(pretty)

    # сохраняем с префиксом модели (например ep2_bs64_)
    pd.DataFrame([m]).to_csv(resdir / f"{args.prefix}metrics.csv", index=False)
    (resdir / f"{args.prefix}metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"✓ saved metrics to {resdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")        # где лежат y_prob.npy / y_true.npy
    p.add_argument("--y_prob", default=None)                  # опционально — явные пути
    p.add_argument("--y_true", default=None)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--k", default="1")                       
    p.add_argument("--prefix", default="")                    
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.25)
    p.add_argument("--t_steps", type=int, default=21)
    main(p.parse_args())
