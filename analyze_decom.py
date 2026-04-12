import argparse
import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp-path", type=str, required=True)
    parser.add_argument("--full-path", type=str, default=None)
    args = parser.parse_args()

    data = load_jsonl(args.decomp_path)

    print(f"Total samples: {len(data)}")

    # ===== 1. fact score 分布 =====
    all_fact_scores = []
    num_facts_list = []

    for item in data:
        facts = item.get("fact_results", [])
        num_facts_list.append(len(facts))
        for f in facts:
            all_fact_scores.append(f["fact_score"])

    all_fact_scores = np.array(all_fact_scores)

    print("\n=== Fact Score Distribution ===")
    print(f"Mean: {all_fact_scores.mean():.4f}")
    print(f"Std:  {all_fact_scores.std():.4f}")
    print(f"Min:  {all_fact_scores.min():.4f}")
    print(f"Max:  {all_fact_scores.max():.4f}")

    print("\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"P{p}: {np.percentile(all_fact_scores, p):.4f}")

    print("\nFacts per sample:")
    print(f"Mean: {np.mean(num_facts_list):.2f}")

    # ===== 2. fact vs GT 相关性 =====
    fact_scores_avg = []
    fact_scores_min = []
    gts = []

    for item in data:
        facts = item.get("fact_results", [])
        if len(facts) == 0:
            continue

        scores = [f["fact_score"] for f in facts]
        fact_scores_avg.append(np.mean(scores))
        fact_scores_min.append(np.min(scores))
        gts.append(item["mos_align"])

    print("\n=== Correlation (Decomp vs GT) ===")
    print("Mean:")
    print("  SRCC:", spearmanr(fact_scores_avg, gts)[0])
    print("  PLCC:", pearsonr(fact_scores_avg, gts)[0])

    print("Min:")
    print("  SRCC:", spearmanr(fact_scores_min, gts)[0])
    print("  PLCC:", pearsonr(fact_scores_min, gts)[0])

    # ===== 3. 每个样本内部 variance =====
    variances = []
    for item in data:
        facts = item.get("fact_results", [])
        if len(facts) > 1:
            scores = [f["fact_score"] for f in facts]
            variances.append(np.std(scores))

    print("\n=== Intra-sample Fact Variance ===")
    print(f"Mean std: {np.mean(variances):.4f}")

    # ===== 4. 找 worst samples =====
    diffs = []
    for item in data:
        gt = item["mos_align"]
        pred = item["pred_score"]
        diffs.append((abs(gt - pred), item))

    diffs.sort(reverse=True, key=lambda x: x[0])

    print("\n=== Top 5 Worst Cases ===")
    for i in range(5):
        d, item = diffs[i]
        print(f"\nCase {i+1}: diff={d:.4f}")
        print("Prompt:", item["prompt"])
        print("GT:", item["mos_align"])
        print("Pred:", item["pred_score"])
        print("Facts:")
        for f in item["fact_results"]:
            print(f"  - {f['fact']} | score={f['fact_score']:.3f}")

    # ===== 5. 如果有 full score，对比 =====
    if args.full_path:
        full_data = load_jsonl(args.full_path)
        full_map = {x["id"]: x for x in full_data}

        full_scores = []
        decomp_scores = []
        gts2 = []

        for item in data:
            sid = item["id"]
            if sid not in full_map:
                continue

            full_scores.append(full_map[sid]["pred_score"])
            decomp_scores.append(item["pred_score"])
            gts2.append(item["mos_align"])

        print("\n=== Full vs Decomp Correlation ===")
        print("Full:")
        print("  SRCC:", spearmanr(full_scores, gts2)[0])
        print("  PLCC:", pearsonr(full_scores, gts2)[0])

        print("Decomp:")
        print("  SRCC:", spearmanr(decomp_scores, gts2)[0])
        print("  PLCC:", pearsonr(decomp_scores, gts2)[0])

        print("\nFull vs Decomp consistency:")
        print("  SRCC:", spearmanr(full_scores, decomp_scores)[0])


if __name__ == "__main__":
    main()