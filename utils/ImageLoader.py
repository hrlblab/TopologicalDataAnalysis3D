import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PersistenceImageDataset(Dataset):
    def __init__(self, root_dir, pred_idx, transform = None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.file_paths = []
        self.labels = []
        self.transform = transform

        for classes_name in sorted(os.listdir(root_dir)):
            label = int(classes_name)
            class_dir = os.path.join(root_dir, classes_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.png'):
                    self.file_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

        if pred_idx == None:
            self.pred_idx = self.labels
        else:
            self.pred_idx = pred_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
