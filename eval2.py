import argparse
import json
import math
from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr


def load_scores(
    jsonl_path: str,
    pred_key: str = "pred_score_1to5",
    gt_key: str = "mos_align",
) -> Tuple[List[float], List[float]]:
    preds = []
    gts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] Skip line {line_idx}: invalid json ({e})")
                continue

            if pred_key not in item or gt_key not in item:
                print(f"[Warning] Skip line {line_idx}: missing key '{pred_key}' or '{gt_key}'")
                continue

            pred = item[pred_key]
            gt = item[gt_key]

            if pred is None or gt is None:
                print(f"[Warning] Skip line {line_idx}: pred/gt is None")
                continue

            try:
                pred = float(pred)
                gt = float(gt)
            except (TypeError, ValueError):
                print(f"[Warning] Skip line {line_idx}: pred/gt cannot be converted to float")
                continue

            if math.isnan(pred) or math.isnan(gt):
                print(f"[Warning] Skip line {line_idx}: pred/gt is NaN")
                continue

            preds.append(pred)
            gts.append(gt)

    return preds, gts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True, help="Path to jsonl result file")
    parser.add_argument(
        "--pred-key",
        type=str,
        default="pred_score_1to5",
        help="Prediction field name, e.g. pred_score_1to5 / p_yes / vqa_p_yes",
    )
    parser.add_argument("--gt-key", type=str, default="mos_align", help="Ground-truth field name")
    args = parser.parse_args()

    preds, gts = load_scores(args.result_path, args.pred_key, args.gt_key)

    if len(preds) < 2:
        raise ValueError(f"Not enough valid samples: {len(preds)}")

    preds = np.array(preds, dtype=np.float64)
    gts = np.array(gts, dtype=np.float64)

    srcc, srcc_p = spearmanr(preds, gts)
    plcc, plcc_p = pearsonr(preds, gts)

    print(f"Valid samples: {len(preds)}")
    print(f"Pred key: {args.pred_key}")
    print(f"GT key: {args.gt_key}")
    print(f"SRCC: {srcc:.6f}")
    print(f"PLCC: {plcc:.6f}")


if __name__ == "__main__":
    main()
