import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CarPlateDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, file_list, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]

        img = Image.open(os.path.join(self.image_dir, name + ".png")).convert("RGB")
        img_tensor = F.to_tensor(img)

        tree = ET.parse(os.path.join(self.annotation_dir, name + ".xml"))
        root = tree.getroot()

        boxes, labels = [], []
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text)
            ymax = float(b.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # class 1 = licence plate, 0 is reserved for background by Faster R-CNN

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, target
