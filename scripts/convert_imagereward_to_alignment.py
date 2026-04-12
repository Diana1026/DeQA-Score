#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from datasets import Image, load_dataset


def to_1to5_from_1to7(x: int) -> float:
    # Linear map: 1..7 -> 1..5
    return 1.0 + (float(x) - 1.0) * (4.0 / 6.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download THUDM/ImageRewardDB and convert to DeQA alignment metas."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/ImageRewardDB",
        help="Output root directory.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="If set, save decoded images to output-dir/images/<split>/.",
    )
    parser.add_argument(
        "--repo-image-prefix",
        type=str,
        default="./data/ImageRewardDB/images",
        help="Prefix written into meta `image` field.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="1k",
        help="ImageRewardDB config: 1k/2k/4k/8k (or *_group/*_pair).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    meta_dir = out_root / "metas"
    img_root = out_root / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        img_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": "train_alignment.json",
        "validation": "val_alignment.json",
        "test": "test_alignment.json",
    }

    all_records = []
    summary = {}

    for split_name, out_name in split_map.items():
        print(f"[ImageRewardDB] processing split: {split_name}", flush=True)
        ds = load_dataset(
            "THUDM/ImageRewardDB",
            args.config_name,
            split=split_name,
            streaming=False,
        )
        ds = ds.cast_column("image", Image(decode=False))
        split_count = 0
        split_skip = 0

        split_img_dir = img_root / split_name
        if args.save_images:
            split_img_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = meta_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fw:
            for row in ds:
                try:
                    pid = str(row["prompt_id"])
                    src_path = (
                        row.get("image", {}).get("path", "")
                        if isinstance(row.get("image", {}), dict)
                        else getattr(row.get("image", None), "filename", "")
                    )
                    src_name = Path(src_path).name if src_path else f"{pid}.webp"
                    image_name = f"{pid}__{src_name}"
                    rec_id = f"{split_name}:{pid}:{Path(src_name).stem}"
                    image_abs = split_img_dir / image_name
                    if args.save_images:
                        image_path_in_meta = (
                            f"{args.repo_image_prefix}/{split_name}/{image_name}"
                        )
                    else:
                        image_path_in_meta = str(src_path)

                    if args.save_images and not image_abs.exists():
                        if not src_path:
                            raise RuntimeError("missing source image path")
                        shutil.copyfile(src_path, image_abs)

                    align_1to7 = int(row["image_text_alignment_rating"])
                    align_1to5 = to_1to5_from_1to7(align_1to7)

                    rec = {
                        "id": rec_id,
                        "image": image_path_in_meta,
                        "prompt": row["prompt"],
                        "mos_align_raw": align_1to7,
                        "mos_align": align_1to5,
                        "std_align": 0.0,
                        "task_type": "alignment",
                        "align_score": align_1to5,
                        "split": split_name,
                        "source": "THUDM/ImageRewardDB",
                        "classification": row["classification"],
                    }
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    split_count += 1

                    if split_count % 2000 == 0:
                        print(
                            f"  {split_name}: kept={split_count} skipped={split_skip}",
                            flush=True,
                        )
                except Exception:
                    split_skip += 1
                    if split_skip % 100 == 0:
                        print(
                            f"  {split_name}: kept={split_count} skipped={split_skip}",
                            flush=True,
                        )

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        with open(meta_dir / out_name, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        all_records.extend(records)
        summary[split_name] = split_count
        summary[f"{split_name}_skipped"] = split_skip
        print(
            f"[ImageRewardDB] {split_name} done: kept={split_count}, skipped={split_skip}",
            flush=True,
        )

    with open(meta_dir / "all_alignment.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    with open(meta_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[ImageRewardDB] completed.")
    print(summary)
    print(f"saved metas: {meta_dir}")
    if args.save_images:
        print(f"saved images: {img_root}")


if __name__ == "__main__":
    main()
