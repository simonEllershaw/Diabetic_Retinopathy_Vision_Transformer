import pandas as pd
import os
import torch
import json
import numpy as np

from torch.utils.data import Dataset
import torchvision
import transforms
import cv2
from transforms import GrahamPreprocessing
import random
from PIL import Image
import sys
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from copy import deepcopy

# Test 
import visualisation
import matplotlib.pyplot as plt
import time

class Coloured_Squares_Dataset(Dataset):
    def __init__(self, max_length, square_size, img_size=224, labels=None):
        # Setup differing transforms for training and testing
        self.augment = False        
        self.img_size = img_size
        self.square_size = square_size
        self.fill = 0
        self.length = max_length

        # Load and extract config variables
        self.labels = self.generate_labels()
        self.tl_square_coords = self.generate_tl_square_coords()
        self.class_names = ["Red", "Green"]
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extract sample's metadata
        label = self.labels[idx]
        img = self.generate_img(self.tl_square_coords[idx], label)
        # Load and transform img
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)(img)
        return img, label, ""

    def generate_labels(self, random_state=None):
        labels = np.zeros(len(self), dtype=int)
        labels[:len(self)//2] = 1
        np.random.shuffle(labels)
        return labels

    def generate_tl_square_coords(self):
        tl = np.random.randint(0, self.img_size-self.square_size, (len(self),2))
        return tl

    def generate_img(self, tl, label):
        img = np.zeros((self.img_size,self.img_size,3), dtype="float32")        
        br = tl + self.square_size
        img[tl[0]:br[0], tl[1]:br[1], label] = 1
        return img

    def get_labels(self):
        return self.labels
    
    def augmentation(self, img):
        augment_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            # torchvision.transforms.RandomAffine(degrees=10, fill=self.fill),#, translate=(0.1,0.1), scale=(0.75,1.25), fill=self.fill),
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2),
        ])
        return augment_transforms(img)

    def select_subset_of_data(self, subset_start, subset_end):
        self.labels = self.labels[subset_start:subset_end]
        self.tl_square_coords = self.tl_square_coords[subset_start:subset_end]
        self.length = len(self.labels)

    def create_train_val_test_datasets(self, proportions, dataset_names):
        subsets = {subset: deepcopy(self) for subset in dataset_names}

        lengths = (proportions*len(self)).astype(int)
        split_indicies = np.cumsum(lengths)
        split_indicies = np.insert(split_indicies, 0, 0)
        for idx, subset in enumerate(subsets.values()):
            subset.select_subset_of_data(split_indicies[idx], split_indicies[idx+1])
        return subsets
                
if __name__ == "__main__":
    np.random.seed(13)
    full_dataset = Coloured_Squares_Dataset(3000, 64)
    print(len(full_dataset))
    # data.augment = True
    idx = 45
    sample = full_dataset[idx]
    fig, ax = plt.subplots()
    visualisation.imshow(sample[0], ax)
    # plt.show()

    dataset_names = ["train", "val", "test"]    
    dataset_proportions = np.array([0.6, 0.2, 0.2])
    # full_dataset = Coloured_Squares_Dataset(1000, 1)
    class_names = full_dataset.class_names
    datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, dataset_names)
    # fig, ax = plt.subplots()
    # visualisation.imshow(datasets["val"][0][0], ax)
    # # plt.show()

    dataloaders = {name: torch.utils.data.DataLoader(datasets[name], batch_size=64,
                                            shuffle=False, num_workers=4)
                        for name in ["val", "test"]}   
    dataloaders["train"] = torch.utils.data.DataLoader(datasets["train"], batch_size=64, shuffle=True, num_workers=4, drop_last = True)

    batch = next(iter(dataloaders["train"]))
    print(batch[0].dtype)
    print(batch[1].dtype)

    # fig = visualisation.sample_batch(dataloaders["train"], class_names)
    # plt.show()