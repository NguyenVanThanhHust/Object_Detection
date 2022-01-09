import os 
from os.path import join
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class VocDataset(Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, transform=None, S=7, B=2, C=20,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        assert transform, "Provide transform"

    def __len__(self):
        return 10
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            cls_label, x, y, w, h = box.tolist()
            cls_label = int(cls_label)
            i, j = int(self.S * y), int(self.S * x) 
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                w * self.S, 
                h * self.S, 
            )
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coords = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coords
                label_matrix[i, j, cls_label] = 1
        return image, label_matrix