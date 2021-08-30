import os
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from datasets.abstract_DR import Abstract_DR_Dataset
from utils import visualisation


class IDRiD_Dataset(Abstract_DR_Dataset):
    def __init__(self, data_directory, img_size=384, patch_size=16, use_inception_norm=None, max_length=None):
        # Setup class specific variables
        self.patch_size = patch_size
        self.img_dir_preprocessed = os.path.join(data_directory, "preprocessed_images")
        self.seg_dir_preprocessed = os.path.join(data_directory, "preprocessed_seg")
        super().__init__(data_directory, img_size, use_inception_norm, max_length)

    def load_labels(self, max_length, **kwargs):
        length = len(os.listdir(self.img_dir_preprocessed)) if max_length is None else max_length
        labels_df = pd.DataFrame({"index": np.arange(length, dtype=int), 'label': np.ones(length, dtype=int)})
        # All examples are referable DR cases (as have annoated lesions!)
        return labels_df

    def __getitem__(self, idx):
        # Extract sample's metadata
        sample = self.labels_df.loc[idx]
        idx = sample["index"]
        label = sample["label"]
        fname = self.get_fname(idx)
        img_path = os.path.join(self.img_dir_preprocessed, fname + ".jpg")
        img = Image.open(img_path)
        img = self.get_augmentations()(img)

        seg_path = os.path.join(self.seg_dir_preprocessed, fname + ".tif")
        seg = Image.open(seg_path)
        seg = torchvision.transforms.Resize(self.img_size)(seg)
        seg = np.array(seg, dtype="float")
        seg[seg>0] = 1.0
        seg_cuml = self.generate_seg_cum(seg)
        return img, seg, seg_cuml, label, fname

    def get_fname(self, idx):
        # Indexed from 01
        return f"IDRiD_{idx+1:0>2d}"

    def generate_seg_cum(self, seg):
        num_patches = self.img_size // self.patch_size
        seg_cum = seg.reshape(num_patches, self.patch_size, num_patches, self.patch_size, 3).max(axis=(1, 3, -1))
        return seg_cum

    def visualise_sample(self, idx):
        fig, axes = plt.subplots(1, 3)
        titles = ["Input Image", "Annotations", "Patched Annotations"]
        image, seg, seg_cum, label, fname = self[idx]
        # Get rid of axis ticks, set label and add background image
        for ax, title in zip(axes, titles):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(title)
            visualisation.imshow(image, ax)
        # Plot seg map
        axes[1].imshow(seg, alpha=0.5)
        # Upsample seg_cum maps then plot
        seg_cum = seg_cum.repeat(16, axis=0).repeat(16, axis=1)
        axes[2].imshow(seg_cum, alpha=0.5)    
        fig.suptitle(f"Fname: {fname}, Label: {self.class_names[label]}")