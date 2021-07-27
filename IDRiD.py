import pandas as pd
import os
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
import torchvision
import cv2
import random
from PIL import Image
import sys
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from copy import deepcopy

# Test 
import visualisation
import matplotlib.pyplot as plt
import time

class IDRiD_Dataset(Dataset):
    def __init__(self, data_directory, img_size=384, patch_size=16):
        # Load and extract config variables
        self.data_directory = data_directory
        self.img_size = img_size
        self.patch_size = patch_size
        self.img_dir = os.path.join(self.data_directory, "Images")
        self.seg_dir = os.path.join(self.data_directory, "Segmentation")
        self.img_dir_preprocessed = os.path.join(self.data_directory, "Images_Preprocessed")
        self.seg_dir_preprocessed = os.path.join(self.data_directory, "Segmentation_Preprocessed")
        self.length = len([name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, name))])
        self.indicies = np.arange(self.length)
        np.random.shuffle(self.indicies)

    def get_fname(self, idx):
        return f"IDRiD_{idx+1:0>2d}"

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        # Extract sample's metadata
        idx = self.indicies[idx]
        fname = self.get_fname(idx)
        img_path = os.path.join(self.img_dir_preprocessed, fname + ".jpg")
        img = Image.open(img_path)
        img = torchvision.transforms.Resize(self.img_size)(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)(img)

        seg_path = os.path.join(self.seg_dir_preprocessed, fname + ".tif")
        seg = Image.open(seg_path)
        seg = torchvision.transforms.Resize(self.img_size)(seg)
        seg = np.array(seg, dtype="float")
        seg[seg>0] = 1.0

        seg_cuml = self.generate_seg_cum(seg)

        return img, seg, seg_cuml, fname

    def generate_segmentation_masks(self):
        ma_dir = os.path.join(self.seg_dir, "1. Microaneurysms")
        he_dir = os.path.join(self.seg_dir, "2. Haemorrhages")
        ex_dir = os.path.join(self.seg_dir, "3. Hard Exudates")
        mask_dir = os.path.join(self.seg_dir, "4. Mask")
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        for sample_num in range(0, len(self)):
            fname = self.get_fname(sample_num)
            seg_mask = cv2.imread(os.path.join(ma_dir, f"{fname}_MA.tif"))
            seg_mask[:,:,0] = cv2.imread(os.path.join(ex_dir, f"{fname}_EX.tif"))[:,:,2]
            try:
                seg_mask[:,:,1] = cv2.imread(os.path.join(he_dir, f"{fname}_HE.tif"))[:,:,2]
            except:
                print(f"{fname} has no haemorrage mask")
            cv2.imwrite(os.path.join(mask_dir, f"{fname}.tif"), seg_mask)

    def generate_seg_cum(self, seg):
        num_patches = self.img_size // self.patch_size
        seg_cum = seg.reshape(num_patches, self.patch_size, num_patches, self.patch_size, 3).sum(axis=(1, 3))
        seg_cum[seg_cum>0] = 1
        return seg_cum

    def select_subset_of_data(self, subset_start, subset_end):
        self.indicies = self.indicies[subset_start:subset_end]

    def create_train_val_test_datasets(self, proportions, dataset_names):
        subsets = {subset: deepcopy(self) for subset in dataset_names}

        lengths = (proportions*len(self)).astype(int)
        split_indicies = np.cumsum(lengths)
        split_indicies = np.insert(split_indicies, 0, 0)
        for idx, subset in enumerate(subsets.values()):
            subset.select_subset_of_data(split_indicies[idx], split_indicies[idx+1])
        return subsets
                
if __name__ == "__main__":
    data = IDRiD_Dataset(r'data\idrid')
    fig, axes = plt.subplots(1,3)
    image, seg, seg_cum, _ = data[0]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        visualisation.imshow(image, ax)
    axes[1].imshow(seg, alpha=0.5)
    seg_cum = seg_cum.repeat(16, axis=0).repeat(16, axis=1)
    axes[2].imshow(seg_cum, alpha=0.5)    
    plt.show()

    # print(len(data))
    
    # idx = 0
    # sample = data[idx]
    # fig, ax = plt.subplots()
    # visualisation.imshow(sample[0], ax)
    # ax.imshow(sample[1], alpha=0.4)
    # plt.show()

    # print(sample[2])
    # print(torch.mean(sample[0]), torch.std(sample[0]))
    # np_image = sample[0].flatten().numpy()
    # print(np_image.shape)
    # plt.hist(np_image)
    # plt.show()
