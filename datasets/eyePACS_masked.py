import os
import numpy as np
from PIL import Image

from datasets.eyePACS import EyePACS_Dataset

class EyePACS_Masked_Dataset(EyePACS_Dataset):
    def __init__(self, data_directory, mask_size=4, img_size=384, use_inception_norm=True, random_state=None, max_length=None, remove_ungradables=True):
        self.mask_size = mask_size
        super().__init__(data_directory, img_size, use_inception_norm, random_state, max_length, remove_ungradables)
        
    def __getitem__(self, idx):
        # Overwrite superclass method to add mask
        # Extract sample's metadata
        sample = self.labels_df.loc[idx]
        label = sample.level
        # Load and transform img
        img_path = os.path.join(self.img_dir_preprocessed, sample.image_name)
        img = Image.open(img_path)
        # Mask image if positive label
        if label == 1:
            img = Image.fromarray(self.mask_img(np.array(img), sample.tl_x, sample.tl_y))
        img = self.get_augmentations()(img)
        return img, label, sample.image_name

    def load_labels(self, max_length, **kwargs):
        labels_df = super().load_labels(max_length, **kwargs)
        # Keep only healthy labels
        labels_df = labels_df[labels_df.level<=1]
        # Generate random masks and labels
        np.random.seed(kwargs["random_state"])
        labels_df["level"] = self.generate_labels(len(labels_df))
        labels_df["tl_x"], labels_df["tl_y"] = self.generate_tl_square_coords(len(labels_df))
        return labels_df

    def generate_labels(self, num_samples):
        # Generate random list of half 1s and half 0s
        labels = np.ones(num_samples, dtype=int)
        labels[:num_samples//2] = 0
        np.random.shuffle(labels)
        return labels.tolist()

    def generate_tl_square_coords(self, num_samples):
        # Random top left x and y coordinates of mask. Bounded by image size
        tl_x = np.random.randint(0, self.img_size-self.mask_size, num_samples).tolist()
        tl_y = np.random.randint(0, self.img_size-self.mask_size, num_samples).tolist()
        return tl_x, tl_y

    def mask_img(self, img, tl_x, tl_y):
        br_x = tl_x + self.mask_size
        br_y = tl_y + self.mask_size
        img[tl_x:br_x, tl_y:br_y, :] = 255
        return img
