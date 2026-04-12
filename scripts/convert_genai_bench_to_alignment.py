#!/usr/bin/env python3
import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_KEYS = [
    "DALLE_3",
    "DeepFloyd_I_XL_v1",
    "Midjourney_6",
    "SDXL_2_1",
    "SDXL_Base",
    "SDXL_Turbo",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GenAI-Bench parquet to DeQA-Score alignment json meta."
    )
    parser.add_argument(
        "--genai-root",
        type=str,
        default="/home/ubuntu/GenAI-Bench",
        help="Root directory of downloaded GenAI-Bench dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./data/GenAI-Bench",
        help="Output root for extracted images and metas.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=-1,
        help="If >0, only process first N prompts for quick testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    genai_root = Path(args.genai_root)
    parquet_files = sorted(glob.glob(str(genai_root / "data" / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {genai_root / 'data'}")

    out_root = Path(args.output_root)
    image_dir = out_root / "images"
    meta_dir = out_root / "metas"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    records = []
    processed_prompts = 0

    for pfile in parquet_files:
        df = pd.read_parquet(pfile)
        for _, row in df.iterrows():
            if args.max_prompts > 0 and processed_prompts >= args.max_prompts:
                break

            idx = int(row["Index"])
            prompt = str(row["Prompt"])
            ratings_dict = row["HumanRatings"]

            for model_name in MODEL_KEYS:
                img_obj = row[model_name]
                if not isinstance(img_obj, dict) or "bytes" not in img_obj:
                    continue

                img_bytes = img_obj["bytes"]
                filename = f"{idx:04d}_{model_name}.jpg"
                out_img_path = image_dir / filename
                if not out_img_path.exists():
                    with open(out_img_path, "wb") as f:
                        f.write(img_bytes)

                model_ratings = np.array(ratings_dict[model_name], dtype=np.float32)
                mos_align = float(model_ratings.mean())
                std_align = float(model_ratings.std())
                if mos_align < 1.0 or mos_align > 5.0:
                    continue

                record = {
                    "id": f"{idx:04d}_{model_name}.jpg",
                    "image": f"./data/GenAI-Bench/images/{filename}",
                    "prompt": prompt,
                    "mos_align_raw": mos_align,
                    "mos_align": mos_align,
                    "std_align": std_align,
                    "task_type": "alignment",
                    "align_score": mos_align,
                    "source_model": model_name,
                }
                records.append(record)

            processed_prompts += 1

        if args.max_prompts > 0 and processed_prompts >= args.max_prompts:
            break

    random.seed(args.seed)
    random.shuffle(records)
    split = int(len(records) * args.train_ratio)
    train_records = records[:split]
    val_records = records[split:]

    train_path = meta_dir / "train_alignment.json"
    val_path = meta_dir / "val_alignment.json"
    all_path = meta_dir / "all_alignment.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Processed prompts: {processed_prompts}")
    print(f"Total samples (prompt x model): {len(records)}")
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")
    print(f"Saved images to: {image_dir}")
    print(f"Saved metas to: {meta_dir}")


if __name__ == "__main__":
    main()
