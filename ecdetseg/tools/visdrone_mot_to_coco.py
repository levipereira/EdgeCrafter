"""Convert VisDrone MOT dataset to COCO detection JSON format.

VisDrone MOT annotation format (per line):
    frame_idx, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
    score, object_category, truncation, occlusion

Creates a flat image directory (symlinks) and COCO JSON annotation file.
Categories 0 (ignored) and 11 (others) are skipped.

Usage:
    conda activate ec
    python ecdetseg/tools/visdrone_mot_to_coco.py \
        --src /mnt/f/dataset_uav/visdrone \
        --dst /mnt/f/dataset_uav/visdrone_coco
"""
import argparse
import json
import os
from pathlib import Path


VISDRONE_ORIGINAL_CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

# Remap 1-10 → 0-9 so labels match model output space [0, num_classes-1]
VISDRONE_REMAP = {orig: orig - 1 for orig in VISDRONE_ORIGINAL_CATEGORIES}
VISDRONE_CATEGORIES = {v: name for (k, name), v in
                       zip(VISDRONE_ORIGINAL_CATEGORIES.items(), VISDRONE_REMAP.values())}

SKIP_CATEGORIES = {0, 11}


def convert_split(src_dir: Path, dst_dir: Path, split: str) -> dict:
    """Convert one split (train/val) from VisDrone MOT to COCO format.

    Args:
        src_dir: Root of VisDrone MOT dataset (contains train/, val/).
        dst_dir: Output directory (will contain images/, annotations/).
        split: 'train' or 'val'.

    Returns:
        COCO annotation dict.
    """
    seq_dir = src_dir / split / "sequences"
    ann_dir = src_dir / split / "annotations"
    img_out = dst_dir / split / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cat_id, "name": name}
            for cat_id, name in sorted(VISDRONE_CATEGORIES.items())
        ],
    }

    image_id = 0
    ann_id = 0

    sequences = sorted(seq_dir.iterdir())
    for seq_path in sequences:
        if not seq_path.is_dir():
            continue
        seq_name = seq_path.name
        ann_file = ann_dir / f"{seq_name}.txt"

        if not ann_file.exists():
            print(f"  WARNING: no annotation for {seq_name}, skipping")
            continue

        frame_annotations: dict[int, list] = {}
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                frame_idx = int(parts[0])
                cat_id = int(parts[7])
                if cat_id in SKIP_CATEGORIES:
                    continue
                if frame_idx not in frame_annotations:
                    frame_annotations[frame_idx] = []
                frame_annotations[frame_idx].append(parts)

        frames = sorted(seq_path.glob("*.jpg"))
        for frame_path in frames:
            frame_num = int(frame_path.stem)
            image_id += 1

            flat_name = f"{seq_name}_{frame_path.name}"
            link_path = img_out / flat_name

            if not link_path.exists():
                os.symlink(frame_path.resolve(), link_path)

            from PIL import Image
            with Image.open(frame_path) as im:
                w, h = im.size

            coco["images"].append({
                "id": image_id,
                "file_name": flat_name,
                "width": w,
                "height": h,
            })

            if frame_num not in frame_annotations:
                continue

            for parts in frame_annotations[frame_num]:
                orig_cat = int(parts[7])
                cat_id = VISDRONE_REMAP[orig_cat]
                x, y, bw, bh = (
                    float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5]),
                )
                if bw <= 0 or bh <= 0:
                    continue

                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })

    return coco


def main():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone MOT to COCO detection format"
    )
    parser.add_argument("--src", type=str, required=True,
                        help="VisDrone MOT root (contains train/, val/)")
    parser.add_argument("--dst", type=str, required=True,
                        help="Output directory for COCO format")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    for split in ["train", "val"]:
        print(f"\nConverting {split}...")
        ann_out_dir = dst / split / "annotations"
        ann_out_dir.mkdir(parents=True, exist_ok=True)

        coco = convert_split(src, dst, split)

        ann_path = ann_out_dir / f"{split}.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f)

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"  {split}: {n_img} images, {n_ann} annotations -> {ann_path}")


if __name__ == "__main__":
    main()
