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
    def __init__(self, data_directory, labels=None, max_length=None, random_state=None):
        # Load and extract config variables
        self.data_directory = data_directory
        self.labels_df = labels if labels is not None else self.load_labels(random_state)
        self.length = max_length if max_length is not None else len(self.labels_df)
        self.img_dir = os.path.join(data_directory, "train", "train")
        self.img_dir_preprocessed = os.path.join(self.data_directory, "preprocessed")
        self.class_names = ["Healthy", "Refer"]#["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        # Setup differing transforms for training and testing
        self.augment = False        
        self.img_size = 224
        self.fill = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform img
        img_path = os.path.join(self.img_dir_preprocessed, metadata.image + ".jpeg")
        img = Image.open(img_path)
        if self.augment:
            img = self.augmentation(img)
        img = torchvision.transforms.ToTensor()(img)
        return img, label, metadata.image

    def load_labels(self, random_state=None):
        label_fname = os.path.join(self.data_directory, "trainLabels.csv", "trainLabels.csv")
        gradability_fname = os.path.join(self.data_directory, "eyepacs_gradability_grades.csv")

        labels_df = pd.read_csv(label_fname)
        gradabilty_df = pd.read_csv(gradability_fname, delimiter = " ")
        labels_df = pd.merge(labels_df, gradabilty_df, how="inner", left_on=labels_df.columns[0], right_on=gradabilty_df.columns[0])
        
        labels_df = labels_df.drop(labels_df[labels_df["gradability"]==0].index)
        labels_df = labels_df.drop(columns=['image_name', 'gradability'])
        labels_df = labels_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        labels_df.level = np.where(labels_df.level>1, 1, 0)
        return labels_df

    def get_labels(self):
        return self.labels_df.level

    def preprocess_image(self, img):
        # Crop image according to bounding box then resize imge
        return transforms.GrahamPreprocessing(img)

    def preprocess_all_images(self):
        if not os.path.exists(self.img_dir_preprocessed):
            os.makedirs(self.img_dir_preprocessed)
        for idx in range(len(self)):
            fname = self.labels_df.loc[idx].image + ".jpeg"
            img = cv2.imread(os.path.join(self.img_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                img = self.preprocess_image(img)
                cv2.imwrite(os.path.join(self.img_dir_preprocessed, fname), img)
            except:
                print(fname)

    
    def augmentation(self, img):
        augment_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomAffine(degrees=10, fill=self.fill),#, translate=(0.1,0.1), scale=(0.75,1.25), fill=self.fill),
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2),
        ])
        return augment_transforms(img)

    def create_train_val_test_datasets(self, proportions, dataset_names):
        lengths = (proportions*len(self)).astype(int)
        split_indicies = np.cumsum(lengths)

        labels_subset = {}
        labels_subset["train"] = self.labels_df.iloc[:split_indicies[0]].reset_index(drop=True)
        labels_subset["val"] = self.labels_df.iloc[split_indicies[0]:split_indicies[1]].reset_index(drop=True)
        labels_subset["test"] = self.labels_df.iloc[split_indicies[1]:].reset_index(drop=True)
        subsets = {subset: EyePACS_Dataset(self.data_directory, labels=labels_subset[subset]) for subset in dataset_names}
        return subsets
                
if __name__ == "__main__":
    # start_time = time.time()
    # data_directory = "diabetic-retinopathy-detection"#sys.argv[1]
    # data = EyePACS_Dataset(data_directory)
    # data.preprocess_all_images()

    data = EyePACS_Dataset("diabetic-retinopathy-detection", random_state=13)
    print(len(data))
    data.augment = True
    idx = 15
    start_time = time.time()
    sample = data[idx]
    # torch.set_printoptions(precision=10)
    # print(sample[2])
    # print(torch.mean(sample[0]), torch.std(sample[0]))
    # np_image = sample[0].flatten().numpy()
    # print(np_image.shape)
    # plt.hist(np_image)
    # plt.show()
    # print(time.time()-start_time)
    # print(sample[2])
    fig, ax = plt.subplots()
    visualisation.imshow(sample[0], ax)
    plt.show()
    
