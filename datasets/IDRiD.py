import os
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from datasets.abstract_DR import Abstract_DR_Dataset
from utilities import visualisation


class IDRiD_Dataset(Abstract_DR_Dataset):
    def __init__(self, data_directory, img_size=384, patch_size=16, use_inception_norm=None, max_length=None):
        # Setup class specific variables
        self.patch_size = patch_size
        self.seg_dir_preprocessed = os.path.join(data_directory, "preprocessed_seg")
        super().__init__(data_directory, img_size, use_inception_norm, max_length)

    def load_labels(self, max_length, **kwargs):
        length = len(os.listdir(self.img_dir_preprocessed)) if max_length is None else max_length
        fnames = [self.get_fname(idx) for idx in np.arange(length, dtype=int)]
        image_names = [fname + ".jpg" for fname in fnames]
        seg_names = [fname + ".tif" for fname in fnames]
        # All examples are referable DR cases (as have annoated lesions!)
        levels = np.ones(length, dtype=int)
        labels_df = pd.DataFrame({"image_name": image_names, "seg_name": seg_names, "level": levels})
        return labels_df

    def __getitem__(self, idx):
        # Overwrites super class def as seg maps loaded as well
        img, label, image_name = super().__getitem__(idx)

        # Get seg maps
        metadata = self.labels_df.loc[idx]
        seg_path = os.path.join(self.seg_dir_preprocessed, metadata.seg_name)
        seg = Image.open(seg_path)
        seg = torchvision.transforms.Resize(self.img_size)(seg)
        seg = np.array(seg)
        seg_cuml = self.generate_seg_cum(seg)
        return img, seg, seg_cuml, label, image_name

    def get_fname(self, idx):
        # Indexed from 01
        return f"IDRiD_{idx+1:0>2d}"

    def generate_seg_cum(self, seg):
        # Binary map in patch_sizexpatch_size grid
        # If pixel in element contains a lesion whole patch is positive
        num_patches = self.img_size // self.patch_size
        seg_cum = seg.reshape(num_patches, self.patch_size, num_patches, self.patch_size, 3).max(axis=(1, 3, -1))
        seg_cum[seg_cum>0] = 1
        return seg_cum

    def visualise_sample(self, idx):
        # Overides superclass method to show seg maps as well
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