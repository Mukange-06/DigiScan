import os
import random
import cv2
from pathlib import Path
from model import get_model

WEIGHTS = "runs/car_plate/weights/best.pt"
OUTPUT_DIR = "predictions"
SCORE_THRESHOLD = 0.5
SEED = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = get_model(WEIGHTS)

    # --- Metrics via ultralytics built-in validation ---
    print("Running validation metrics...\n")
    metrics = model.val(data="dataset.yaml", conf=SCORE_THRESHOLD)
    print(f"mAP@50      : {metrics.box.map50:.4f}")
    print(f"mAP@50-95   : {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    # --- OpenCV visualisation on 5 random val images ---
    val_paths = Path("val.txt").read_text().strip().splitlines()
    random.seed(SEED)
    sample = random.sample(val_paths, min(5, len(val_paths)))

    for img_path in sample:
        results = model(img_path, conf=SCORE_THRESHOLD)[0]
        img = cv2.imread(img_path)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{score:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_name = Path(img_path).stem + "_pred.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), img)

    print(f"\nSample predictions saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
