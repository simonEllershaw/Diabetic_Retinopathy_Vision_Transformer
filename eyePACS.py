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

# Test 
import visualisation
import matplotlib.pyplot as plt
import time

class EyePACS_Dataset(Dataset):
    def __init__(self, data_directory, indices=None, max_length=None):
        # Load and extract config variables
        labels_fname = os.path.join(data_directory, "trainLabels.csv", "trainLabels.csv")
        labels_df_full = pd.read_csv(labels_fname)
        if indices is not None:
            self.labels_df = labels_df_full.iloc[indices].reset_index(drop=True)
        elif max_length is not None:
            self.labels_df = labels_df_full.iloc[:max_length]
        else:
            self.labels_df = labels_df_full
        self.data_directory = data_directory
        self.img_dir = os.path.join(data_directory, "train", "train")
        self.img_dir_preprocessed = os.path.join(self.data_directory, "preprocessed")
        self.class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        # Setup differing transforms for training and testing
        self.augment = False        
        self.img_size = 224

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform img
        img_path = os.path.join(self.img_dir_preprocessed, metadata.image + ".jpeg")
        img = Image.open(img_path)
        if self.augment:
            img = self.augmentation(img)
        img = torchvision.transforms.ToTensor().__call__(img)
        return img, label, metadata.image

    def get_labels(self):
        return self.labels_df.level

    def preprocess_image(self, img):
        # Crop image according to bounding box then resize imge
        left, top, box_size = self.calc_cropbox_dim(img)
        img = torchvision.transforms.functional.crop(img, top, left, box_size, box_size)
        img = torchvision.transforms.Resize((self.img_size, self.img_size)).forward(img)
        return img

    def preprocess_all_images(self):
        if not os.path.exists(self.img_dir_preprocessed):
            os.makedirs(self.img_dir_preprocessed)
        for idx in self.indices:
            fname = self.labels_df.loc[idx].image + ".jpeg"
            img = Image.open(os.path.join(self.img_dir, fname))
            img = self.preprocess_image(img)
            img.save(os.path.join(self.img_dir_preprocessed, fname))

    def calc_cropbox_dim(self, img):
        # Sum over colour channels and threshold to give
        # segmentation map. Only look at every 100th row/col
        # to reduce compute at cost of accuracy
        stride = 100
        img_np = np.asarray(img)[::stride,::stride].sum(2) # converting to np array slow but necessary
        img_np = np.where(img_np>img_np.mean()/5, 1, 0)
        # Find nonzero rows and columns (convert back to org indexing)
        non_zero_rows = np.nonzero(img_np.sum(1))[0]*stride
        non_zero_columns = np.nonzero(img_np.sum(0))[0]*stride
        # Boundaries given first and last non zero rows/columns 
        boundary_coords = np.zeros((2,2))
        boundary_coords[:, 0] = non_zero_columns[[0, -1]] # x coords
        boundary_coords[:, 1] = non_zero_rows[[0, -1]] # y coords
        # Center is middle of the non zero values
        center = np.zeros(2)
        center[0], center[1] = np.median(non_zero_columns), np.median(non_zero_rows)
        # Radius is max boundary difference, add stride to conservatively account
        # for uncertainity due to it's use and pad by 5%
        radius = ((max(boundary_coords[1] - boundary_coords[0])/2)+stride)*1.05
        top_left_coord = np.round((center - radius))
        return top_left_coord[0], top_left_coord[1], round(radius*2)
    
    def augmentation(self, img):
        augment_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomAffine(degrees=180, translate=(0.1,0.1), scale=(0.8,1.2)),
        ])
        return augment_transforms(img)

    def create_train_val_test_datasets(data_directory, proportions, dataset_names, max_length=None):
        full_dataset = EyePACS_Dataset(data_directory, max_length=max_length)
        indicies = np.arange(len(full_dataset))
        np.random.shuffle(indicies)
        proportions = (proportions*len(indicies)).astype(int)
        print(proportions)
        test = np.cumsum(proportions[:2])
        print(test)
        split_indicies = np.split(indicies, test)
        print([i.shape for i in split_indicies])
        datasets = {dataset_names[i]: EyePACS_Dataset(data_directory, split_indicies[i]) for i in range(len(split_indicies))}
        dataset_indicies = {dataset_names[i]: split_indicies[i] for i in range(len(split_indicies))}
        return datasets, dataset_indicies
                
if __name__ == "__main__":
    start_time = time.time()
    data_directory = sys.argv[1]
    data = EyePACS_Dataset(data_directory)
    data.preprocess_all_images()
    # print(time.time()-start_time)

    # data.augment = True
    # print(len(data))
    # for i in range(10):
    #     start_time = time.time()
    #     x = data[i]
    #     print(time.time()-start_time)
    # data.is_train_set = False
    # start_time = time.time()
    # data = data[2]
    # print(time.time()-start_time)
    # fig, ax = plt.subplots()
    # idx = 12
    # start_time = time.time()
    # sample = data[idx]
    # print(time.time()-start_time)
    # visualisation.imshow(sample[0], ax)
    # plt.show()

