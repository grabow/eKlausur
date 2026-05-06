"""Deterministic train/validation split preparation and YOLOv5 training entrypoint."""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

DEFAULT_DATASET_DIR = os.getenv("YOLO_HG_DATASET_DIR", "/workspace/eKlausurData/YoloMultiClassGenerated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic train/val split and train YOLOv5.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Folder containing *.txt + image files.")
    parser.add_argument("--image-ext", default=".png", help="Image extension used in dataset-dir.")
    parser.add_argument("--split-percentage", type=int, default=90, help="Train split percentage (0-100).")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for split and training reproducibility.")
    parser.add_argument("--data-config", default="dataset_hg_multiclass.yaml", help="YOLO data config yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Train/val image size.")
    parser.add_argument("--weights", default="yolov5s.pt", help="Initial weights.")
    parser.add_argument("--hyp", default="hyp_hg_table.yaml", help="Hyperparameter yaml.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience (epochs).")
    parser.add_argument("--batch", type=int, default=-1, help="YOLO batch arg (-1=autobatch).")
    parser.add_argument("--device", default="", help="Training device, e.g. '', 'cpu', 'mps', '0'.")
    parser.add_argument("--noval", action="store_true", help="Skip validation during training.")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare and print split, do not train.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def discover_samples(dataset_dir: Path, image_ext: str) -> List[str]:
    samples: List[str] = []
    for txt_path in sorted(dataset_dir.glob("*.txt")):
        stem = txt_path.stem
        img_path = dataset_dir / f"{stem}{image_ext}"
        if img_path.exists():
            samples.append(stem)
    return samples


def reset_split_dirs(dataset_dir: Path) -> Tuple[Path, Path, Path, Path]:
    images_path = dataset_dir / "images"
    labels_path = dataset_dir / "labels"
    if images_path.exists():
        shutil.rmtree(images_path)
    if labels_path.exists():
        shutil.rmtree(labels_path)

    training_images_path = images_path / "training"
    validation_images_path = images_path / "validation"
    training_labels_path = labels_path / "training"
    validation_labels_path = labels_path / "validation"
    training_images_path.mkdir(parents=True, exist_ok=True)
    validation_images_path.mkdir(parents=True, exist_ok=True)
    training_labels_path.mkdir(parents=True, exist_ok=True)
    validation_labels_path.mkdir(parents=True, exist_ok=True)
    return training_images_path, validation_images_path, training_labels_path, validation_labels_path


def copy_split(
    dataset_dir: Path,
    image_ext: str,
    train_stems: List[str],
    val_stems: List[str],
    train_img_dir: Path,
    val_img_dir: Path,
    train_lbl_dir: Path,
    val_lbl_dir: Path,
) -> None:
    for stem in train_stems:
        shutil.copy(dataset_dir / f"{stem}{image_ext}", train_img_dir)
        shutil.copy(dataset_dir / f"{stem}.txt", train_lbl_dir)
    for stem in val_stems:
        shutil.copy(dataset_dir / f"{stem}{image_ext}", val_img_dir)
        shutil.copy(dataset_dir / f"{stem}.txt", val_lbl_dir)


def write_split_manifest(dataset_dir: Path, train_stems: List[str], val_stems: List[str], seed: int) -> None:
    manifest = dataset_dir / "split_manifest_seed.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"seed={seed}\n")
        f.write(f"train_count={len(train_stems)}\n")
        f.write(f"val_count={len(val_stems)}\n")
        f.write("train:\n")
        for stem in train_stems:
            f.write(f"{stem}\n")
        f.write("val:\n")
        for stem in val_stems:
            f.write(f"{stem}\n")


def prepare_split(dataset_dir: Path, image_ext: str, split_percentage: int, seed: int) -> Tuple[List[str], List[str]]:
    samples = discover_samples(dataset_dir, image_ext)
    if not samples:
        raise RuntimeError(f"No samples found in {dataset_dir} for image extension {image_ext}")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    split_idx = int(split_percentage * len(shuffled) / 100)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    train_stems = shuffled[:split_idx]
    val_stems = shuffled[split_idx:]
    return train_stems, val_stems


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if args.split_percentage <= 0 or args.split_percentage >= 100:
        raise ValueError("--split-percentage must be between 1 and 99")

    set_global_seed(args.seed)
    train_stems, val_stems = prepare_split(dataset_dir, args.image_ext, args.split_percentage, args.seed)
    train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir = reset_split_dirs(dataset_dir)
    copy_split(
        dataset_dir,
        args.image_ext,
        train_stems,
        val_stems,
        train_img_dir,
        val_img_dir,
        train_lbl_dir,
        val_lbl_dir,
    )
    write_split_manifest(dataset_dir, train_stems, val_stems, args.seed)

    print("Prepared deterministic split")
    print(f"dataset_dir: {dataset_dir}")
    print(f"seed: {args.seed}")
    print(f"train samples: {len(train_stems)}")
    print(f"val samples: {len(val_stems)}")

    if args.dry_run:
        print("dry-run: training skipped")
        return 0

    import torch

    # Compatibility with torch>=2.6 where torch.load defaults to weights_only=True.
    orig_torch_load = torch.load

    def patched_torch_load(*load_args, **load_kwargs):
        if "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = False
        return orig_torch_load(*load_args, **load_kwargs)

    torch.load = patched_torch_load

    import train

    train.run(
        data=args.data_config,
        imgsz=args.imgsz,
        weights=args.weights,
        hyp=args.hyp,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        noval=args.noval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
