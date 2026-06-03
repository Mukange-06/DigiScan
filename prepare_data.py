"""
Run this ONCE before training.
Converts Pascal VOC XML annotations to YOLO txt format and creates dataset.yaml.
"""
import os
import random
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path

IMAGE_DIR = Path("Data/images")
ANNOTATION_DIR = Path("Data/annotations")
LABELS_DIR = Path("Data/labels")
SEED = 42


def xml_to_yolo(xml_path, img_width, img_height):
    # YOLO format: class_id  cx  cy  w  h  (all normalised 0-1)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        cx = (xmin + xmax) / 2 / img_width
        cy = (ymin + ymax) / 2 / img_height
        w  = (xmax - xmin) / img_width
        h  = (ymax - ymin) / img_height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    LABELS_DIR.mkdir(exist_ok=True)

    all_files = [f.stem for f in ANNOTATION_DIR.glob("*.xml")]

    for name in all_files:
        xml_path = ANNOTATION_DIR / f"{name}.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        width  = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        lines = xml_to_yolo(xml_path, width, height)
        (LABELS_DIR / f"{name}.txt").write_text("\n".join(lines))

    print(f"Converted {len(all_files)} annotations -> '{LABELS_DIR}'")

    # 80/20 train/val split
    random.seed(SEED)
    random.shuffle(all_files)
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files   = all_files[split:]

    # Ultralytics resolves labels by replacing 'images' with 'labels' in the path,
    # so absolute paths keep everything unambiguous on Windows
    abs_image_dir = IMAGE_DIR.resolve()

    Path("train.txt").write_text("\n".join(str(abs_image_dir / (n + ".png")) for n in train_files))
    Path("val.txt").write_text("\n".join(str(abs_image_dir / (n + ".png")) for n in val_files))

    dataset_cfg = {
        "train": str(Path("train.txt").resolve()),
        "val":   str(Path("val.txt").resolve()),
        "nc": 1,
        "names": ["licence"],
    }
    with open("dataset.yaml", "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    print(f"Train: {len(train_files)} images | Val: {len(val_files)} images")
    print("dataset.yaml written. Run train.py next.")


if __name__ == "__main__":
    main()
