import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torch

class EyePACS_Dataset(Dataset):
    def __init__(self, label_file, img_dir, transform_train, transform_test):
        self.labels_df = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.is_train_set = False

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        meta_data = self.labels_df.loc[idx]
        img_path = os.path.join(self.img_dir, meta_data.image + ".jpeg")
        image = read_image(img_path)
        image = self.transform_train(image) if self.is_train_set else self.transform_test(image)
        label = meta_data.level
        sample = {"image": image, "label": label}
        return sample

    def split_train_test_val_sets(self, prop_train, prop_val, prop_test):
        num_train = round(prop_train*len(self))
        num_val = round(prop_val*len(self))
        num_test = len(self) - num_train - num_val
        print(num_train, num_val, num_test)
        dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(self, [num_train, num_val, num_test])
        dataset_train.train_set = True
        return dataset_train, dataset_val, dataset_test