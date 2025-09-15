import argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from pathlib import Path
import json

def main(args):
    res = Path(args.results_dir)
    y_true = np.load(res / "y_true.npy")
    y_prob = np.load(res / "y_prob.npy")
    L = min(y_true.shape[1], y_prob.shape[1])
    y_true, y_prob = y_true[:, :L], y_prob[:, :L]
    y_pred = (y_prob >= args.threshold).astype(int)

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    err = np.zeros((L, L))
    for i in range(L):
        fn_mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
        if fn_mask.any():
            err[i] += y_pred[fn_mask].sum(0)
    np.fill_diagonal(err, 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(err, cmap="Reds", aspect="auto")
    plt.colorbar()
    plt.title("Confusion map (false negatives redistributed)")
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.tight_layout()
    out = res / "conf_map.png"
    plt.savefig(out, dpi=160)
    print(f"✓ saved {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--threshold", type=float, default=0.15)
    main(p.parse_args())
