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

class EyePACS_Dataset(Dataset):
    def __init__(self, data_directory, max_length=None, random_state=None, img_size=384, remove_ungradables=True, labels_to_binary=True):
        # Load and extract config variables
        self.data_directory = data_directory
        self.labels_df = self.load_labels(random_state, max_length, remove_ungradables, labels_to_binary)
        self.img_dir = os.path.join(data_directory, "train", "train")
        self.img_dir_preprocessed = os.path.join(self.data_directory, "preprocessed_448")
        self.class_names = ["Healthy", "Refer"]#["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        # Setup differing transforms for training and testing
        self.augment = False        
        self.img_size = img_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform img
        img_path = os.path.join(self.img_dir_preprocessed, metadata.image + ".jpeg")
        img = Image.open(img_path)
        img = torchvision.transforms.Resize(self.img_size)(img)
        if self.augment:
            img = self.augmentation(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)(img)
        return img, label, metadata.image

    def load_labels(self, random_state=None, max_length=None, remove_ungradables=True, labels_to_binary=True):
        label_fname = os.path.join(self.data_directory, "trainLabels.csv", "trainLabels.csv")
        labels_df = pd.read_csv(label_fname)
        if remove_ungradables:
            gradability_fname = os.path.join(self.data_directory, "eyepacs_gradability_grades.csv")
            gradabilty_df = pd.read_csv(gradability_fname, delimiter = " ")
            labels_df = pd.merge(labels_df, gradabilty_df, how="inner", left_on=labels_df.columns[0], right_on=gradabilty_df.columns[0])
            labels_df = labels_df.drop(labels_df[labels_df["gradability"]==0].index)
            labels_df = labels_df.drop(columns=['image_name', 'gradability'])
        if labels_to_binary:
            labels_df.level = np.where(labels_df.level>1, 1, 0)
        labels_df = labels_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        labels_df = labels_df.iloc[:max_length] if max_length is not None else labels_df
        return labels_df

    def get_labels(self):
        return self.labels_df.level
    
    def augmentation(self, img):
        augment_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            # torchvision.transforms.RandomAffine(degrees=10, fill=self.fill),#, translate=(0.1,0.1), scale=(0.75,1.25), fill=self.fill),
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2),
        ])
        return augment_transforms(img)

    def select_subset_of_data(self, subset_start, subset_end):
        self.labels_df = self.labels_df.iloc[subset_start:subset_end].reset_index(drop=True)

    def create_train_val_test_datasets(self, proportions, dataset_names):
        subsets = {subset: deepcopy(self) for subset in dataset_names}

        lengths = (proportions*len(self)).astype(int)
        split_indicies = np.cumsum(lengths)
        split_indicies = np.insert(split_indicies, 0, 0)
        for idx, subset in enumerate(subsets.values()):
            subset.select_subset_of_data(split_indicies[idx], split_indicies[idx+1])
        return subsets
                
if __name__ == "__main__":
    data = EyePACS_Dataset("diabetic-retinopathy-detection", random_state=13)
    print(len(data))
    data.augment = True
    idx = 15
    start_time = time.time()
    sample = data[idx]
    fig, ax = plt.subplots()
    visualisation.imshow(sample[0], ax)
    plt.show()
    
    # torch.set_printoptions(precision=10)
    # print(sample[2])
    # print(torch.mean(sample[0]), torch.std(sample[0]))
    # np_image = sample[0].flatten().numpy()
    # print(np_image.shape)
    # plt.hist(np_image)
    # plt.show()
