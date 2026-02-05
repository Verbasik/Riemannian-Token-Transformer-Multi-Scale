import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import ensure_outdir, load_val_preds
except ImportError:
    from common import ensure_outdir, load_val_preds


def top_confusions(y_true, y_pred, k=10):
    pairs = {}
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        pairs[(t, p)] = pairs.get((t, p), 0) + 1
    return sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:k]


def main():
    parser = argparse.ArgumentParser(description="Error analysis: top confusions and confident mistakes")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory with val_preds.npz")
    parser.add_argument("--top-k", type=int, default=20, help="How many misclassified samples to save")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.exp_dir / "analysis" / "errors")
    preds = load_val_preds(args.exp_dir)
    y_true, y_pred, proba = preds["y_true"], preds["y_pred"], preds["proba"]
    subject_id = preds.get("subject_id")
    sample_id = preds.get("sample_id")

    # Top confusion pairs
    conf_pairs = top_confusions(y_true, y_pred, k=args.top_k)
    with (out_dir / "top_confusions.csv").open("w") as f:
        f.write("true,pred,count\n")
        for (t, p), c in conf_pairs:
            f.write(f"{t},{p},{c}\n")

    # Confident mistakes ranked by max prob
    wrong_mask = y_true != y_pred
    wrong_idx = np.where(wrong_mask)[0]
    if wrong_idx.size:
        max_conf = proba[wrong_idx].max(axis=1)
        entropy = (-proba[wrong_idx] * np.log(proba[wrong_idx] + 1e-8)).sum(axis=1)
        df = pd.DataFrame(
            {
                "idx": wrong_idx,
                "true": y_true[wrong_idx],
                "pred": y_pred[wrong_idx],
                "confidence": max_conf,
                "entropy": entropy,
            }
        )
        if subject_id is not None:
            df["subject_id"] = subject_id[wrong_idx]
        if sample_id is not None:
            df["sample_id"] = sample_id[wrong_idx]
        df = df.sort_values(by=["confidence", "entropy"], ascending=[False, True]).head(args.top_k)
        df.to_csv(out_dir / "top_confident_errors.csv", index=False)

    print(f"Saved error reports to {out_dir}")


if __name__ == "__main__":
    main()
