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
        return 800
        # return len(self.labels_df)

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform img
        img_path = os.path.join(self.img_dir, metadata.image + ".jpeg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment:
            img = self.augmentation(img)
        img = GrahamPreprocessing(img)
        img = cv2.resize(img, (224,224))
        img = np.divide(img, 255.0, dtype=np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        # Return sample
        return img, label
    
    def augmentation(self, img):
        # Random flip
        flip_code = random.randrange(-1,2) # -1 both, 0 x-axis, 1 y-axis, 2 none
        if flip_code < 2:
            img = cv2.flip(img, flip_code)
        rows, cols, _ = img.shape
        # Random translation
        translation_x = random.uniform(-1, 1)*0.2*img.shape[0] 
        translation_y = random.uniform(-1, 1)*0.2*img.shape[1]
        translation_matrix = np.float32(([[1,0,translation_x],[0,1,translation_y]]))
        img = cv2.warpAffine(img,translation_matrix,(cols,rows))
        # Random rotation
        angle = random.random()*360
        rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2), random.uniform(-1, 1)*45, 1)
        return cv2.warpAffine(img,rotation_matrix,(cols,rows))

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
    data = EyePACS_Dataset("config.json")
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
    idx = 10
    visualisation.imshow(data[idx][0], ax)
    plt.show()
