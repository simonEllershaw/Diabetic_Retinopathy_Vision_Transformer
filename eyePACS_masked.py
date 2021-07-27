import pandas as pd
import os
import torch
import json
import numpy as np
from copy import deepcopy

from torch.utils.data import Dataset
import torchvision
import cv2
import random
from PIL import Image
import sys
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

# Test 
import visualisation
import matplotlib.pyplot as plt
import time

class EyePACS_Masked_Dataset(Dataset):
    def __init__(self, data_directory, img_size=384, mask_size=4, max_length=None, random_state=None):
        # Load and extract config variables
        self.augment = False        
        self.img_size = img_size
        self.mask_size = mask_size
        self.data_directory = data_directory
        self.labels_df = self.load_labels(random_state, max_length)
        self.img_dir = os.path.join(data_directory, "train", "train")
        self.img_dir_preprocessed = os.path.join(self.data_directory, "preprocessed_448")
        self.class_names = ["Healthy", "Refer"]#["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        # Setup differing transforms for training and testing
        

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.label
        # Load and transform img
        img_path = os.path.join(self.img_dir_preprocessed, metadata.image + ".jpeg")
        img = Image.open(img_path)
        if label == 1:
            img = Image.fromarray(self.mask_img(np.array(img), metadata.tl_x, metadata.tl_y))
        img = torchvision.transforms.Resize(self.img_size)(img)
        if self.augment:
            img = self.augmentation(img) 
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)(img)
        return img, label, metadata.image

    def load_labels(self, random_state=None, max_length=None, remove_ungradables=True, labels_to_binary=True):
        label_fname = os.path.join(self.data_directory, "trainLabels.csv", "trainLabels.csv")
        labels_df = pd.read_csv(label_fname)
        
        labels_df = labels_df[labels_df.level==0]
        labels_df["label"] = self.generate_labels(len(labels_df))
        labels_df["tl_x"], labels_df["tl_y"] = self.generate_tl_square_coords(len(labels_df))

        labels_df = labels_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        labels_df = labels_df.iloc[:max_length] if max_length is not None else labels_df
        return labels_df

    def generate_labels(self, num_samples):
        labels = np.ones(num_samples, dtype=int)
        labels[:num_samples//2] = 0
        np.random.shuffle(labels)
        return labels.tolist()

    def generate_tl_square_coords(self, num_samples):
        tl_x = np.random.randint(0, self.img_size-self.mask_size, num_samples).tolist()
        tl_y = np.random.randint(0, self.img_size-self.mask_size, num_samples).tolist()
        return tl_x, tl_y

    def mask_img(self, img, tl_x, tl_y):
        br_x = tl_x + self.mask_size
        br_y = tl_y + self.mask_size
        img[tl_x:br_x, tl_y:br_y, :] = 255
        return img

    def get_labels(self):
        return self.labels_df.label
    
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
    np.random.seed(13)
    data = EyePACS_Masked_Dataset("diabetic-retinopathy-detection", random_state=13)
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
