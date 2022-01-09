import os
from os.path import join
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import VocDataset
from model import Yolo
from loss import YoloLoss

from utils import (
    intersection_over_union, 
    mean_average_precision, 
    cellboxes_to_boxes,
    # plot_image, 
)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS=0

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

base_transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_one_epoch(model, data_loader, optimizer, criterion):
    model = model.to(DEVICE)
    model.train()
    mean_loss = []
    
    for idx, (images, tgts) in enumerate(data_loader):
        images = images.to(DEVICE)
        tgts = tgts.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, tgts)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, data_loader, criterion, output_dir):
    model = model.to(DEVICE)
    model.eval()
    mean_loss = []
    all_pred_boxes = []
    all_true_boxes = []

    for idx, (images, tgts) in enumerate(data_loader):
        images = images.to(DEVICE)
        tgts = tgts.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, tgts)
        mean_loss.append(loss.item())
        print(outputs.shape)
        print(images.shape)
        pred_boxes = cellboxes_to_boxes(outputs)
        tgt_boxes = cellboxes_to_boxes(tgts)
        map = mean_average_precision(outputs, tgts)


def main():
    model = Yolo(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = YoloLoss()
    data_folder = "./datasets/"
    csv_file = join(data_folder, "train.csv")
    img_dir = join(data_folder, "images")
    lbl_dir = join(data_folder, "labels")

    train_dataset = VocDataset(csv_file=csv_file, img_dir=img_dir, label_dir=lbl_dir, transform=base_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):
        train_one_epoch(model, train_loader, optimizer, criterion)
        evaluate(model, train_loader, criterion, output_dir)

if __name__ == "__main__":
    main()