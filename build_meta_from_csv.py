import argparse
import json
import os
import pandas as pd


def map_0to5_to_1to5(x):
    return 1.0 + 4.0 * float(x) / 5.0


def scale_std_0to5_to_1to5(x):
    return 4.0 * float(x) / 5.0


def zscore_normalize(series):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return [0.0] * len(series)
    return ((series - mean) / std).tolist()


def main(args):
    df = pd.read_csv(args.csv_path)

    required_cols = ["name", "prompt", "mos_quality", "std_quality", "mos_align", "std_align"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    meta = []
    for _, row in df.iterrows():
        image_rel_path = os.path.join(args.image_prefix, row["name"]).replace("\\", "/")

        item = {
            "id": row["name"],
            "image": image_rel_path,
            "prompt": str(row["prompt"]) if not pd.isna(row["prompt"]) else "",
            "mos_quality": float(row["mos_quality"]),
            "std_quality": float(row["std_quality"]),
            "mos_align_raw": float(row["mos_align"]),
            "mos_align": map_0to5_to_1to5(row["mos_align"]),
            "align_score": map_0to5_to_1to5(row["mos_align"]),
            "std_align": scale_std_0to5_to_1to5(row["std_align"]),
        }

        meta.append(item)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(meta)} samples to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--image-prefix", type=str, default="AGIQA3K/images")
    args = parser.parse_args()
    main(args)
