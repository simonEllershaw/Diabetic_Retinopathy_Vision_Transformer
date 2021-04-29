import pandas as pd
import os
import torch
import json
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class EyePACS_Dataset(Dataset):
    def __init__(self, config_fname):
        # Load and extract config variables
        with open(config_fname) as json_file:
            config = json.load(json_file)["eyePACS"]
        self.labels_df = pd.read_csv(config["label_file"])
        self.img_dir = config["img_dir"]
        self.class_names = config["class_names"]
        # Setup differing transforms for training and testing
        self.is_train_set = False
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Extract sample's metadata
        metadata = self.labels_df.loc[idx]
        label = metadata.level
        # Load and transform image
        img_path = os.path.join(self.img_dir, metadata.image + ".jpeg")
        image = Image.open(img_path)
        image = self.transform_train(image) if self.is_train_set else self.transform_test(image)
        # Return sample
        return image, label

    def split_train_test_val_sets(self, prop_train, prop_val, prop_test):
        # Calc split in terms of num samples instead of proportions
        num_train = round(prop_train*len(self))
        num_val = round(prop_val*len(self))
        num_test = len(self) - num_train - num_val
        # Split and set training set bool so correct transform occurs
        dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(self, [num_train, num_val, num_test])
        dataset_train.train_set = True
        return dataset_train, dataset_val, dataset_test