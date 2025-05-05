"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 05/05/25
"""
import torch
import torch.utils.data
import pandas as pd
import os

from PIL import Image


class ChexpertTest(torch.utils.data.Dataset):
    def __init__(self, root, processor, labels):
        self.root = root
        self.processor = processor
        self.labels = labels

        self.df = pd.read_csv(os.path.join(root, 'test_labels.csv'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]

        image = Image.open(os.path.join(self.root, entry["Path"]))
        inputs = self.processor(images=image, return_tensors="pt")
        label = entry[self.labels].tolist()

        return inputs['pixel_values'][0], label


