from copy import deepcopy
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class EyePACS_Dataset(Dataset):
    def __init__(self, data_directory, img_size=384, use_inception_norm=True, random_state=None, remove_ungradables=True, labels_to_binary=True, max_length=None):
        # Setup file structure
        self.data_directory = data_directory
        self.img_dir = os.path.join(data_directory, "train", "train")
        self.img_dir_preprocessed = os.path.join(self.data_directory, "preprocessed_images")
        self.labels_fname = os.path.join(self.data_directory, "trainLabels.csv", "trainLabels.csv")
        self.gradability_fname = os.path.join(self.data_directory, "eyepacs_gradability_grades.csv")
        # Load labels
        self.class_names = ["Healthy", "Refer"]
        self.labels_df = self.load_labels(random_state, max_length, remove_ungradables, labels_to_binary)
        # Setup differing transforms for training and testing
        self.augment = False        
        self.img_size = img_size
        self.std = IMAGENET_INCEPTION_STD if use_inception_norm else IMAGENET_DEFAULT_STD
        self.mean = IMAGENET_INCEPTION_MEAN if use_inception_norm else IMAGENET_DEFAULT_MEAN

    def __len__(self):
        return len(self.labels_df)

    def get_labels(self):
        return self.labels_df.level

    def load_labels(self, random_state=None, max_length=None, remove_ungradables=True, labels_to_binary=True):
        # Load label csv to dataframe
        labels_df = pd.read_csv(self.labels_fname)
        if remove_ungradables:
            labels_df = self.remove_ungradables(labels_df)
        if labels_to_binary:
            labels_df = self.labels_to_binary(labels_df)
        # Random shuffle
        labels_df = labels_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        # Choose number of samples to keep
        labels_df = labels_df.iloc[:max_length] if max_length is not None else labels_df
        return labels_df

    def remove_ungradables(self, labels_df):
        # Load gradabilty csv to dataframe
        gradabilty_df = pd.read_csv(self.gradability_fname, delimiter = " ")
        # Merge label and gradabilty dataframes
        labels_df = pd.merge(labels_df, gradabilty_df, how="inner", left_on=labels_df.columns[0], right_on=gradabilty_df.columns[0])
        # Drop ungradables and non-required columns
        labels_df = labels_df.drop(labels_df[labels_df["gradability"]==0].index)
        labels_df = labels_df.drop(columns=['image_name', 'gradability'])
        return labels_df

    def labels_to_binary(self, labels_df):
        # Threshold
        labels_df.level = np.where(labels_df.level>1, 1, 0)
        return labels_df

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform img
        img_path = os.path.join(self.img_dir_preprocessed, metadata.image + ".jpeg")
        img = Image.open(img_path)
        # Augment image to correct size, type and norm
        img = torchvision.transforms.Resize(self.img_size)(img)
        # Add random augmentations if training
        if self.augment:
            img = self.augmentation(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(self.mean, self.std)(img)
        return img, label, metadata.image
    
    def augmentation(self, img):
        augment_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomAffine(degrees=180, translate=(0.05,0.05), scale=(0.95,1.05), fill=0),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.02),
            torchvision.transforms.GaussianBlur(5),
        ])
        return augment_transforms(img)

    def select_subset_of_data(self, subset_start, subset_end):
        # Use pandas indexing to select subset
        self.labels_df = self.labels_df.iloc[subset_start:subset_end].reset_index(drop=True)

    def create_train_val_test_datasets(self, proportions, dataset_names):
        # Copy subset (so all settings the same) then select subset
        subsets = {subset: deepcopy(self) for subset in dataset_names}
        
        # Calc how long each subset is and then cumulative sum to find inidices
        lengths = (proportions*len(self)).astype(int)
        split_indicies = np.cumsum(lengths)
        # Add 0 index for first split
        split_indicies = np.insert(split_indicies, 0, 0)
        for idx, subset in enumerate(subsets.values()):
            subset.select_subset_of_data(split_indicies[idx], split_indicies[idx+1])
        return subsets

