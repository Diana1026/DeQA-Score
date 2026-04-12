#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Q-Eval-100K image train annotations to DeQA metas."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="./data/Q-Eval-100K/Images_train_jsons",
        help="Directory containing image_alignment_train.json and image_quality_train.json.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./data/Q-Eval-100K/metas",
        help="Output meta directory.",
    )
    parser.add_argument(
        "--repo-image-prefix",
        type=str,
        default="./data/Q-Eval-100K",
        help="Prefix prepended to image_path when writing `image`.",
    )
    return parser.parse_args()


def convert_alignment(items, image_prefix):
    out = []
    for i, x in enumerate(items):
        prompt = x.get("prompt", "")
        gt = float(x.get("gt_score", 0.0))
        image_rel = x.get("image_path", "")
        out.append(
            {
                "id": f"qeval_img_align_train:{i}",
                "image": f"{image_prefix}/{image_rel}" if image_rel else "",
                "prompt": prompt,
                "mos_align": gt,
                "std_align": 0.0,
                "task_type": "alignment",
                "align_score": gt,
                "model": x.get("model", ""),
                "split": "train",
                "source": "AGI-Eval-Official/Q-Eval-100K",
            }
        )
    return out


def convert_quality(items, image_prefix):
    out = []
    for i, x in enumerate(items):
        prompt = x.get("prompt", "")
        gt = float(x.get("gt_score", 0.0))
        image_rel = x.get("image_path", "")
        out.append(
            {
                "id": f"qeval_img_quality_train:{i}",
                "image": f"{image_prefix}/{image_rel}" if image_rel else "",
                "prompt": prompt,
                "mos_quality": gt,
                "std_quality": 0.0,
                "task_type": "quality",
                "quality_score": gt,
                "model": x.get("model", ""),
                "split": "train",
                "source": "AGI-Eval-Official/Q-Eval-100K",
            }
        )
    return out


def main():
    args = parse_args()
    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    align_path = in_root / "image_alignment_train.json"
    qual_path = in_root / "image_quality_train.json"

    with open(align_path, "r", encoding="utf-8") as f:
        align_raw = json.load(f)
    with open(qual_path, "r", encoding="utf-8") as f:
        qual_raw = json.load(f)

    align_meta = convert_alignment(align_raw, args.repo_image_prefix)
    qual_meta = convert_quality(qual_raw, args.repo_image_prefix)

    with open(out_root / "train_alignment.json", "w", encoding="utf-8") as f:
        json.dump(align_meta, f, ensure_ascii=False, indent=2)
    with open(out_root / "all_alignment.json", "w", encoding="utf-8") as f:
        json.dump(align_meta, f, ensure_ascii=False, indent=2)

    with open(out_root / "train_quality.json", "w", encoding="utf-8") as f:
        json.dump(qual_meta, f, ensure_ascii=False, indent=2)
    with open(out_root / "all_quality.json", "w", encoding="utf-8") as f:
        json.dump(qual_meta, f, ensure_ascii=False, indent=2)

    summary = {
        "alignment_train": len(align_meta),
        "quality_train": len(qual_meta),
        "note": "Q-Eval-100K public release provides train annotations only.",
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(summary)
    print(f"saved metas to: {out_root}")


if __name__ == "__main__":
    main()
