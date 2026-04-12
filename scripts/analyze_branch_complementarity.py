import argparse
import json
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _valid_float(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[Warn] skip invalid json line {i}")
    return rows


def collect_arrays(
    rows: List[Dict],
    gt_key: str,
    vqa_key: str,
    decomp_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gts, vqa, decomp = [], [], []
    for r in rows:
        gt = _valid_float(r.get(gt_key))
        v = _valid_float(r.get(vqa_key))
        d = _valid_float(r.get(decomp_key))
        if gt is None or v is None or d is None:
            continue
        gts.append(gt)
        vqa.append(v)
        decomp.append(d)
    return (
        np.asarray(gts, dtype=np.float64),
        np.asarray(vqa, dtype=np.float64),
        np.asarray(decomp, dtype=np.float64),
    )


def corr_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    srcc, _ = spearmanr(pred, gt)
    plcc, _ = pearsonr(pred, gt)
    return float(srcc), float(plcc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--gt-key", type=str, default="mos_align")
    parser.add_argument("--vqa-key", type=str, default="pred_score_1to5")
    parser.add_argument("--decomp-key", type=str, default="decomp_pred_score_1to5")
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_jsonl(args.result_path)
    gt, vqa, decomp = collect_arrays(rows, args.gt_key, args.vqa_key, args.decomp_key)
    n = len(gt)
    if n < 2:
        raise ValueError(f"Not enough valid rows: {n}")

    sr_vqa, pl_vqa = corr_metrics(vqa, gt)
    sr_dec, pl_dec = corr_metrics(decomp, gt)
    sr_cross, _ = spearmanr(vqa, decomp)
    pl_cross, _ = pearsonr(vqa, decomp)

    print(f"Valid samples: {n}")
    print(f"VQA ({args.vqa_key})   -> SRCC: {sr_vqa:.6f}, PLCC: {pl_vqa:.6f}")
    print(f"Decomp ({args.decomp_key}) -> SRCC: {sr_dec:.6f}, PLCC: {pl_dec:.6f}")
    print(f"Branch corr (VQA vs Decomp): Spearman={float(sr_cross):.6f}, Pearson={float(pl_cross):.6f}")

    alphas = np.arange(args.alpha_min, args.alpha_max + 1e-12, args.alpha_step)
    best_sr = (-1e9, None, None)
    best_pl = (-1e9, None, None)

    for a in alphas:
        fused = vqa + a * decomp
        sr, pl = corr_metrics(fused, gt)
        if sr > best_sr[0]:
            best_sr = (sr, a, pl)
        if pl > best_pl[0]:
            best_pl = (pl, a, sr)

    print(
        f"Best-by-SRCC: alpha={best_sr[1]:.3f}, SRCC={best_sr[0]:.6f}, PLCC={best_sr[2]:.6f}"
    )
    print(
        f"Best-by-PLCC: alpha={best_pl[1]:.3f}, PLCC={best_pl[0]:.6f}, SRCC={best_pl[2]:.6f}"
    )
    print("Note: fused score uses additive form: fused = vqa + alpha * decomp")


if __name__ == "__main__":
    main()
