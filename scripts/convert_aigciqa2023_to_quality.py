#!/usr/bin/env python3
import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AIGCIQA2023 into quality/authenticity/alignment metas for DeQA-Score."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/AIGCIQA2023",
        help="Root dir of AIGCIQA2023",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_vec(path: Path, key: str):
    arr = sio.loadmat(path)[key]
    return np.asarray(arr).reshape(-1).astype(np.float32)


def ensure_unzip(zip_path: Path, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir.parent)


def dump_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_records(records, train_ratio: float, seed: int):
    rec = list(records)
    random.seed(seed)
    random.shuffle(rec)
    split = int(len(rec) * train_ratio)
    return rec[:split], rec[split:]


def main():
    args = parse_args()
    root = Path(args.root)

    image_zip = root / "Image" / "allimg.zip"
    image_dir = root / "Image" / "allimg"
    ensure_unzip(image_zip, image_dir)

    # Dataset dimensions (by user requirement):
    # mosz1/SD1 -> quality
    # mosz2/SD2 -> authenticity
    # mosz3/SD3 -> correspondence (alignment)
    mos_quality = load_vec(root / "DATA" / "MOS" / "mosz1.mat", "MOSz")
    mos_auth = load_vec(root / "DATA" / "MOS" / "mosz2.mat", "MOSz")
    mos_align = load_vec(root / "DATA" / "MOS" / "mosz3.mat", "MOSz")
    std_quality = load_vec(root / "DATA" / "STD" / "SD1.mat", "SD")
    std_auth = load_vec(root / "DATA" / "STD" / "SD2.mat", "SD")
    std_align = load_vec(root / "DATA" / "STD" / "SD3.mat", "SD")

    raw_json_path = root / "AIGIQA2023.json"
    align_raw_map = {}
    if raw_json_path.exists():
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        for item in raw_items:
            img_name = str(item["img"])
            if img_name not in align_raw_map:
                align_raw_map[img_name] = float(item["moz3"])

    n = len(mos_quality)
    assert len(mos_auth) == n and len(mos_align) == n
    assert len(std_quality) == n and len(std_auth) == n and len(std_align) == n

    all_multi = []
    all_quality = []
    all_auth = []
    all_alignment = []

    missing = 0
    for i in range(n):
        img_name = f"{i}.png"
        img_path = image_dir / img_name
        if not img_path.exists():
            missing += 1
            continue

        # Keep relative path style consistent with existing metas.
        rel_img = str(img_path).replace("\\", "/")
        base = {
            "id": img_name,
            "image": rel_img,
            # Prompt text is not provided by this package.
            "prompt": "",
            "mos_quality": float(mos_quality[i]),
            "std_quality": float(std_quality[i]),
            "mos_authenticity": float(mos_auth[i]),
            "std_authenticity": float(std_auth[i]),
            "mos_align_raw": align_raw_map.get(img_name, float(mos_align[i])),
            "mos_align": float(mos_align[i]),
            "std_align": float(std_align[i]),
        }
        all_multi.append(base)

        rec_quality = dict(base)
        rec_quality.update(
            {
                "task_type": "score",
                "gt_score": float(mos_quality[i]),
                "std": float(std_quality[i]),
            }
        )
        all_quality.append(rec_quality)

        rec_auth = dict(base)
        rec_auth.update(
            {
                "task_type": "score",
                "gt_score": float(mos_auth[i]),
                "std": float(std_auth[i]),
                "authenticity_score": float(mos_auth[i]),
            }
        )
        all_auth.append(rec_auth)

        rec_align = dict(base)
        rec_align.update(
            {
                "task_type": "alignment",
                "align_score": float(mos_align[i]),
            }
        )
        all_alignment.append(rec_align)

    out_dir = root / "metas"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Multidimensional master meta (one-to-one image mapping, all 3 labels).
    dump_json(out_dir / "all_multidim.json", all_multi)

    # 2) Quality task metas.
    q_train, q_val = split_records(all_quality, args.train_ratio, args.seed)
    dump_json(out_dir / "all_quality.json", all_quality)
    dump_json(out_dir / "train_quality.json", q_train)
    dump_json(out_dir / "val_quality.json", q_val)

    # 3) Authenticity task metas.
    a_train, a_val = split_records(all_auth, args.train_ratio, args.seed)
    dump_json(out_dir / "all_authenticity.json", all_auth)
    dump_json(out_dir / "train_authenticity.json", a_train)
    dump_json(out_dir / "val_authenticity.json", a_val)

    # 4) Alignment/correspondence metas.
    al_train, al_val = split_records(all_alignment, args.train_ratio, args.seed)
    dump_json(out_dir / "all_alignment.json", all_alignment)
    dump_json(out_dir / "train_alignment.json", al_train)
    dump_json(out_dir / "val_alignment.json", al_val)

    print(f"Total records: {len(all_multi)} | Missing images: {missing}")
    print(f"Saved metas to: {out_dir}")
    print(
        "Generated files: all_multidim_nodecomp + "
        "quality/authenticity/alignment train/val/all."
    )
    print("Note: prompt is empty because AIGCIQA2023 package has no prompt text.")


if __name__ == "__main__":
    main()
