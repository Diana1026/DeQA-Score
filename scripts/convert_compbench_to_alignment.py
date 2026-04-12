#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CompBench annotation_all.json to DeQA-Score alignment metas."
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="./data/CompBench/annotation_all.json",
        help="Path to CompBench annotation_all.json",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./data/CompBench/images",
        help="Directory of CompBench images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/CompBench/metas",
        help="Output directory for converted metas.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="../T2I-CompBench/examples/dataset",
        help="Directory containing T2I-CompBench *train.txt/*val.txt files.",
    )
    return parser.parse_args()


def _norm_prompt(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def main():
    args = parse_args()
    ann_path = Path(args.annotation_path)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    split_dir = Path(args.split_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_to_file_stem = {
        "color": "color",
        "shape": "shape",
        "texture": "texture",
        "2d-spatial relationship": "spatial",
        "non-spatial relationship": "non_spatial",
        "complex": "complex",
        "numeracy": "numeracy",
        "3d-spatial relationship": "3d_spatial",
    }

    split_map = {}
    for cat_name, stem in cat_to_file_stem.items():
        train_path = split_dir / f"{stem}_train.txt"
        val_path = split_dir / f"{stem}_val.txt"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Missing split files for '{cat_name}': {train_path} / {val_path}"
            )
        train_prompts = {
            _norm_prompt(line.split("\t")[0])
            for line in train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        val_prompts = {
            _norm_prompt(line.split("\t")[0])
            for line in val_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        split_map[cat_name] = (train_prompts, val_prompts)

    records = []
    missing = 0
    unmatched = 0
    for item in data:
        image_name = item["image_name"]
        image_path = image_dir / image_name
        if not image_path.exists():
            missing += 1
            continue

        score = float(item["score"])
        rec = {
            "id": image_name,
            "image": str(image_path).replace("\\", "/"),
            "prompt": item["prompt"],
            "mos_align_raw": score,
            "mos_align": score,
            "std_align": 0.0,
            "task_type": "alignment",
            "align_score": score,
            "category": item.get("category", ""),
            "source_model": item.get("method", ""),
        }
        records.append(rec)

    train_records = []
    val_records = []
    for rec in records:
        category = rec.get("category", "")
        prompt = _norm_prompt(rec.get("prompt", ""))
        train_prompts, val_prompts = split_map.get(category, (set(), set()))
        if prompt in train_prompts:
            train_records.append(rec)
        elif prompt in val_prompts:
            val_records.append(rec)
        else:
            # This meta bundle is evaluation-centric; unmatched prompts stay in val.
            val_records.append(rec)
            unmatched += 1

    train_path = out_dir / "train_alignment.json"
    val_path = out_dir / "val_alignment.json"
    all_path = out_dir / "all_alignment.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Loaded annotations: {len(data)}")
    print(f"Valid records: {len(records)} | Missing images skipped: {missing}")
    print(f"Unmatched prompts fallback-to-val: {unmatched}")
    print(f"Train: {len(train_records)} | Val: {len(val_records)}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
