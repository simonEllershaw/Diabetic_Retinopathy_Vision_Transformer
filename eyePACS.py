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

# Test 
import visualisation
import matplotlib.pyplot as plt
import time

class EyePACS_Dataset(Dataset):
    def __init__(self, data_directory):
        # Load and extract config variables
        labels_fname = os.path.join(data_directory, "trainLabels.csv", "trainLabels.csv")
        self.labels_df = pd.read_csv(labels_fname)
        self.img_dir = os.path.join(data_directory, "train", "train")
        self.class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        # Setup differing transforms for training and testing
        self.augment = False        

    def __len__(self):
        # return 100
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform img
        img_path = os.path.join(self.img_dir, metadata.image + ".jpeg")
        img = Image.open(img_path)
        if self.augment:
            img = self.augmentation(img)
        img = self.preprocess_image(img)
        return img, label

    def preprocess_image(self, img):
        # img = GrahamPreprocessing(np.array(img))
        # Crop image according to bounding box
        left, top, box_size = self.calc_cropbox_dim(img)
        img = torchvision.transforms.functional.crop(img, top, left, box_size, box_size)
        # Resize and convert to tensor
        preprocessing_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor()
        ])
        return preprocessing_transforms(img)

    def calc_cropbox_dim(self, img):
        # Sum over colour channels and threshold to give
        # segmentation map. Only look at every 100th row/col
        # to reduce compute at cost of accuracy
        stride = 100
        img_np = np.array(img)[::stride,::stride].sum(2) 
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
        # for uncertainity due to it's use
        radius = (max(boundary_coords[1] - boundary_coords[0])/2)+stride
        top_left_coord = np.round((center - radius))
        return top_left_coord[0], top_left_coord[1], round(radius*2)
    
    def augmentation(self, img):
        # Pad image so that image is not augment out of frame
        width, height = img.size
        padded_img_size = round(max(height,width)*1.1)
        padding_tb = (padded_img_size-height)//2
        padding_lr = (padded_img_size-width)//2
        # Apply standard data augmentations
        augment_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Pad((padding_lr, padding_tb)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomAffine(degrees=180, translate=(0.1,0.1), scale=(0.8,1.2)),
        ])
        return augment_transforms(img)

    def split_train_test_val_sets(self, prop_train, prop_val, prop_test):
        # Calc split in terms of num samples instead of proportions
        num_train = round(prop_train*len(self))
        num_val = round(prop_val*len(self))
        num_test = len(self) - num_train - num_val
        # Split and set training set bool so correct transform occurs
        dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(self, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(1))
        dataset_train.train_set = True
        return dataset_train, dataset_val, dataset_test
                
if __name__ == "__main__":
    data = EyePACS_Dataset("diabetic-retinopathy-detection")
    data.augment = True
    print(len(data))
    # for i in range(10):
    #     start_time = time.time()
    #     x = data[i]
    #     print(time.time()-start_time)
    # data.is_train_set = False
    # start_time = time.time()
    # data = data[2]
    # print(time.time()-start_time)
    fig, ax = plt.subplots()
    idx = 64
    start_time = time.time()
    sample = data[idx][0]
    print(time.time()-start_time)
    visualisation.imshow(data[idx][0], ax)
    plt.show()

