import os
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import matplotlib.pyplot as plt

from utils import visualisation

class Abstract_DR_Dataset(Dataset, metaclass = ABCMeta):
    def __init__(self, data_directory, img_size, use_inception_norm, max_length, **kwargs):
        # Setup file structure
        self.data_directory = data_directory
        self.img_dir_preprocessed = os.path.join(self.data_directory, "preprocessed_images")
        # Load labels
        self.class_names = ["Healthy", "Refer"]
        self.labels_df = self.load_labels(max_length, **kwargs)
        # Setup differing transforms for training and testing
        self.augment = False        
        self.img_size = img_size
        self.std = IMAGENET_INCEPTION_STD if use_inception_norm else IMAGENET_DEFAULT_STD
        self.mean = IMAGENET_INCEPTION_MEAN if use_inception_norm else IMAGENET_DEFAULT_MEAN

    @abstractmethod
    def load_labels(self, max_length, **kwargs):
        pass

    def __len__(self):
        return len(self.labels_df)

    def get_labels(self):
        return self.labels_df.level

    def labels_to_binary(self, labels_df):
        # Threshold
        labels_df.level = np.where(labels_df.level>1, 1, 0)
        return labels_df

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform img
        img_path = os.path.join(self.img_dir_preprocessed, metadata.image_name)
        img = Image.open(img_path)
        img = self.get_augmentations()(img)
        return img, label, metadata.image_name
    
    def get_augmentations(self):
        # Resize 1st to minimise computation
        augmentations = [torchvision.transforms.Resize(self.img_size)]
        # Add random augmentations if training
        if self.augment:
            augmentations += [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomAffine(degrees=180, translate=(0.05,0.05), scale=(0.95,1.05), fill=0),
                torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.02),
                torchvision.transforms.GaussianBlur(5),
            ]
        # Always to tensor and normalise
        augmentations += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ]
        return torchvision.transforms.Compose(augmentations)
    
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

    def select_subset_of_data(self, subset_start, subset_end):
        # Use pandas indexing to select subset
        self.labels_df = self.labels_df.iloc[subset_start:subset_end].reset_index(drop=True)

    def visualise_sample(self, idx):
        img, label, fname = self[idx]
        fig, ax = plt.subplots()
        visualisation.imshow(img, ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Fname: {fname}, Label: {self.class_names[label]}")