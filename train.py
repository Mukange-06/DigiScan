from model import get_model

EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640  # pixels — YOLO resizes all images to this during training


def main():
    model = get_model("yolov8n.pt")  # downloads pretrained weights on first run

    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project="runs",
        name="car_plate",
        exist_ok=True,   # overwrite previous run folder if re-training
    )

    print("\nTraining complete.")
    print("Best weights saved to: runs/car_plate/weights/best.pt")


if __name__ == "__main__":
    main()
