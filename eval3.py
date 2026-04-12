import argparse
import json
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_jsonl_as_dict(path: str) -> Dict[str, dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] Skip invalid json at line {line_idx} in {path}: {e}")
                continue

            sample_id = item.get("id")
            if sample_id is None:
                print(f"[Warning] Skip line {line_idx} in {path}: missing 'id'")
                continue
            data[sample_id] = item
    return data


def is_valid_number(x) -> bool:
    if x is None:
        return False
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return not math.isnan(x)


def build_merged_records(
    full_path: str,
    decomp_path: str,
    full_key: str = "pred_score",
    decomp_key: str = "pred_score",
    gt_key: str = "mos_align",
) -> List[dict]:
    full_data = load_jsonl_as_dict(full_path)
    decomp_data = load_jsonl_as_dict(decomp_path)

    common_ids = sorted(set(full_data.keys()) & set(decomp_data.keys()))
    print(f"Full file samples   : {len(full_data)}")
    print(f"Decomp file samples : {len(decomp_data)}")
    print(f"Matched samples     : {len(common_ids)}")

    merged = []
    skipped = 0

    for sid in common_ids:
        full_item = full_data[sid]
        decomp_item = decomp_data[sid]

        s_full = full_item.get(full_key)
        s_decomp = decomp_item.get(decomp_key)
        gt = full_item.get(gt_key)

        if not (is_valid_number(s_full) and is_valid_number(s_decomp) and is_valid_number(gt)):
            skipped += 1
            continue

        merged.append(
            {
                "id": sid,
                "full_score": float(s_full),
                "decomp_score": float(s_decomp),
                "gt": float(gt),
            }
        )

    print(f"Valid merged samples: {len(merged)}")
    print(f"Skipped samples     : {skipped}")
    return merged


def evaluate_scores(preds: List[float], gts: List[float]) -> Tuple[float, float]:
    srcc = spearmanr(preds, gts)[0]
    plcc = pearsonr(preds, gts)[0]
    return srcc, plcc


def fusion_linear(records: List[dict], alpha: float) -> List[float]:
    return [
        alpha * x["full_score"] + (1.0 - alpha) * x["decomp_score"]
        for x in records
    ]


def fusion_uncertainty_gate(records: List[dict], lam: float, delta: float) -> List[float]:
    preds = []
    for x in records:
        s_full = x["full_score"]
        s_decomp = x["decomp_score"]

        # full 越接近 0.5，越不确定，w 越大
        w = max(0.0, 1.0 - abs(s_full - 0.5) / delta)
        w = min(1.0, w)

        s = (1.0 - lam * w) * s_full + (lam * w) * s_decomp
        s = max(0.0, min(1.0, s))
        preds.append(s)
    return preds


def fusion_penalty(records: List[dict], lam: float, tau: float) -> List[float]:
    preds = []
    for x in records:
        s_full = x["full_score"]
        s_decomp = x["decomp_score"]

        penalty = lam * max(0.0, tau - s_decomp)
        s = s_full - penalty
        s = max(0.0, min(1.0, s))
        preds.append(s)
    return preds


def fusion_hybrid(records: List[dict], lam: float, delta: float, tau: float) -> List[float]:
    preds = []
    for x in records:
        s_full = x["full_score"]
        s_decomp = x["decomp_score"]

        # 只在 full 不确定时启用惩罚
        w = max(0.0, 1.0 - abs(s_full - 0.5) / delta)
        w = min(1.0, w)

        penalty = lam * w * max(0.0, tau - s_decomp)
        s = s_full - penalty
        s = max(0.0, min(1.0, s))
        preds.append(s)
    return preds


def print_baselines(records: List[dict]):
    gts = [x["gt"] for x in records]
    full_preds = [x["full_score"] for x in records]
    decomp_preds = [x["decomp_score"] for x in records]

    full_srcc, full_plcc = evaluate_scores(full_preds, gts)
    decomp_srcc, decomp_plcc = evaluate_scores(decomp_preds, gts)

    print("\n=== Baselines ===")
    print(f"Full-only   | SRCC = {full_srcc:.6f} | PLCC = {full_plcc:.6f}")
    print(f"Decomp-only | SRCC = {decomp_srcc:.6f} | PLCC = {decomp_plcc:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-path", type=str, required=True)
    parser.add_argument("--decomp-path", type=str, required=True)
    parser.add_argument("--full-key", type=str, default="pred_score")
    parser.add_argument("--decomp-key", type=str, default="pred_score")
    parser.add_argument("--gt-key", type=str, default="mos_align")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["linear", "gate", "penalty", "hybrid", "all"],
    )
    args = parser.parse_args()

    records = build_merged_records(
        full_path=args.full_path,
        decomp_path=args.decomp_path,
        full_key=args.full_key,
        decomp_key=args.decomp_key,
        gt_key=args.gt_key,
    )

    if len(records) < 2:
        raise ValueError("Not enough valid matched samples to evaluate.")

    gts = [x["gt"] for x in records]
    print_baselines(records)

    if args.mode in ["linear", "all"]:
        print("\n=== Linear Fusion ===")
        best = None
        for alpha in np.linspace(0, 1, 21):
            preds = fusion_linear(records, float(alpha))
            srcc, plcc = evaluate_scores(preds, gts)
            print(f"alpha={alpha:.2f} | SRCC = {srcc:.6f} | PLCC = {plcc:.6f}")
            score = srcc + plcc
            if best is None or score > best[0]:
                best = (score, alpha, srcc, plcc)
        print(
            f"Best Linear | alpha={best[1]:.2f} | SRCC = {best[2]:.6f} | PLCC = {best[3]:.6f}"
        )

    if args.mode in ["gate", "all"]:
        print("\n=== Uncertainty-Gated Fusion ===")
        best = None
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for delta in [0.05, 0.10, 0.15, 0.20, 0.25]:
                preds = fusion_uncertainty_gate(records, lam=float(lam), delta=float(delta))
                srcc, plcc = evaluate_scores(preds, gts)
                print(
                    f"lam={lam:.2f}, delta={delta:.2f} | SRCC = {srcc:.6f} | PLCC = {plcc:.6f}"
                )
                score = srcc + plcc
                if best is None or score > best[0]:
                    best = (score, lam, delta, srcc, plcc)
        print(
            f"Best Gate | lam={best[1]:.2f}, delta={best[2]:.2f} | "
            f"SRCC = {best[3]:.6f} | PLCC = {best[4]:.6f}"
        )

    if args.mode in ["penalty", "all"]:
        print("\n=== Penalty Fusion ===")
        best = None
        for lam in [0.05, 0.10, 0.15, 0.20, 0.25]:
            for tau in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
                preds = fusion_penalty(records, lam=float(lam), tau=float(tau))
                srcc, plcc = evaluate_scores(preds, gts)
                print(
                    f"lam={lam:.2f}, tau={tau:.2f} | SRCC = {srcc:.6f} | PLCC = {plcc:.6f}"
                )
                score = srcc + plcc
                if best is None or score > best[0]:
                    best = (score, lam, tau, srcc, plcc)
        print(
            f"Best Penalty | lam={best[1]:.2f}, tau={best[2]:.2f} | "
            f"SRCC = {best[3]:.6f} | PLCC = {best[4]:.6f}"
        )

    if args.mode in ["hybrid", "all"]:
        print("\n=== Hybrid: Uncertain + Penalty ===")
        best = None
        for lam in [0.05, 0.10, 0.15, 0.20]:
            for delta in [0.05, 0.10, 0.15, 0.20]:
                for tau in [0.35, 0.40, 0.45, 0.50]:
                    preds = fusion_hybrid(
                        records,
                        lam=float(lam),
                        delta=float(delta),
                        tau=float(tau),
                    )
                    srcc, plcc = evaluate_scores(preds, gts)
                    print(
                        f"lam={lam:.2f}, delta={delta:.2f}, tau={tau:.2f} | "
                        f"SRCC = {srcc:.6f} | PLCC = {plcc:.6f}"
                    )
                    score = srcc + plcc
                    if best is None or score > best[0]:
                        best = (score, lam, delta, tau, srcc, plcc)
        print(
            f"Best Hybrid | lam={best[1]:.2f}, delta={best[2]:.2f}, tau={best[3]:.2f} | "
            f"SRCC = {best[4]:.6f} | PLCC = {best[5]:.6f}"
        )


if __name__ == "__main__":
    main()