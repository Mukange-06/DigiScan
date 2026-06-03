from ultralytics import YOLO


def get_model(weights="yolov8n.pt"):
    """
    Load a YOLO model.
    Pass a pretrained weights file (e.g. 'yolov8n.pt') for transfer learning,
    or a trained checkpoint (e.g. 'runs/car_plate/weights/best.pt') for inference.
    yolov8n = nano (fastest/smallest), yolov8s = small, yolov8m = medium
    """
    return YOLO(weights)
