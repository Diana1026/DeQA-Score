import json
import math
from scipy.stats import spearmanr, pearsonr


LEVEL_WEIGHTS = {
    "excellent": 5.0,
    "good": 4.0,
    "fair": 3.0,
    "poor": 2.0,
    "bad": 1.0,
}


def softmax_from_logits(logits_dict, level_order=None):
    if level_order is None:
        level_order = ["excellent", "good", "fair", "poor", "bad"]

    vals = [float(logits_dict[k]) for k in level_order]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    probs = [x / s for x in exps]

    return {k: p for k, p in zip(level_order, probs)}


def expected_score_from_logits(logits_dict):
    probs = softmax_from_logits(logits_dict)
    score = 0.0
    for k, w in LEVEL_WEIGHTS.items():
        score += probs[k] * w
    return score


def aggregate_fact_scores(fact_scores, mode="mean"):
    if len(fact_scores) == 0:
        return None
    if mode == "mean":
        return sum(fact_scores) / len(fact_scores)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def load_pred_and_gt(result_path, gt_key="mos_align"):
    y_true = []
    y_pred = []

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            gt = item.get(gt_key, None)
            fact_results = item.get("fact_results", [])

            if gt is None or len(fact_results) == 0:
                continue

            fact_scores = []
            for fr in fact_results:
                logits = fr.get("logits", None)
                if logits is None:
                    continue
                fact_score = expected_score_from_logits(logits)
                fact_scores.append(fact_score)

            pred_score = aggregate_fact_scores(fact_scores, mode="mean")
            if pred_score is None:
                continue

            y_true.append(float(gt))
            y_pred.append(float(pred_score))

    return y_true, y_pred


def evaluate_srcc_plcc(result_path, gt_key="mos_align"):
    y_true, y_pred = load_pred_and_gt(result_path, gt_key=gt_key)

    if len(y_true) < 2:
        raise ValueError("Not enough valid samples.")

    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)

    return srcc, plcc, len(y_true)


if __name__ == "__main__":
    result_path = "results/res_deqa_mix3/output_decomposition.json"

    srcc, plcc, n = evaluate_srcc_plcc(
        result_path,
        gt_key="mos_align",   # 也可以换成 mos_quality
    )

    print(f"N = {n}")
    print(f"SRCC = {srcc:.4f}")
    print(f"PLCC = {plcc:.4f}")