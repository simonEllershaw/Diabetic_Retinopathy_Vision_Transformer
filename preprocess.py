from PIL import ImageFilter, Image
import numpy as np
import math
import cv2  

import matplotlib.pyplot as plt
import time
import sys
import os 

def preprocess_all_images(img_dir, img_dir_preprocessed, image_format, seg_dir=None, seg_dir_preprocessed=None, seg_format=None):
    if not os.path.exists(img_dir_preprocessed):
        os.makedirs(img_dir_preprocessed)

    for fname in os.listdir(img_dir):
        if fname.endswith(image_format):
            img = cv2.imread(os.path.join(img_dir, fname))
            img, x_min, y_min, radius_inital = GrahamPreprocessing(img)
            cv2.imwrite(os.path.join(img_dir_preprocessed, fname), img)
            if seg_dir is not None:
                if not os.path.exists(seg_dir_preprocessed):
                    os.makedirs(seg_dir_preprocessed)
                fname = fname[:-3] + seg_format
                seg = cv2.imread(os.path.join(seg_dir, fname))
                seg, _, _, _ = GrahamPreprocessing(seg, x_min, y_min, radius_inital)
                cv2.imwrite(os.path.join(seg_dir_preprocessed, fname), seg)


def GrahamPreprocessing(img, x_min_no_pad=None, y_min_no_pad=None, radius_inital=None):
    if x_min_no_pad is None:
        x_min_no_pad, y_min_no_pad, radius_inital = calc_cropbox_dim(img)

    if radius_inital < 10:
        raise ValueError("Image radius could not be found")

    img, x_min, y_min = pad_image(img, x_min_no_pad, y_min_no_pad)
    img = img[y_min:y_min+radius_inital*2, x_min:x_min+radius_inital*2]
    radius_scaled = 500       
    img = rescale_image(img, radius_inital, radius_scaled)
    # img = subtract_average_local_colour(img)
    # img = threshold_boundary(img, round(radius_scaled*0.9))
    img = cv2.resize(img, [448,448])
    return img, x_min_no_pad, y_min_no_pad, radius_inital

def estimate_radius(image):
    # Take central row and sum accross channels
    x = image[image.shape[0]//2].sum(1)
    # ROI defined as non zero pixels
    radius = (x>x.mean()/10).sum()/2
    return radius

def rescale_image(image, radius_intial, radius_scaled):
    width, height, _ = image.shape
    # Resize so radius of eye is = radius_scaled
    s = radius_scaled/radius_intial
    return cv2.resize(image, (0,0), fx=s, fy=s)

def subtract_average_local_colour(image):
    average_local_colour = cv2.GaussianBlur(image, (0,0), 10)
    image = cv2.addWeighted(image, 4, average_local_colour, -4, 128)
    return image

def threshold_boundary(img, radius_boundary):
    boundary = np.zeros(img.shape)
    cv2.circle(boundary, (img.shape[1]//2, img.shape[0]//2), radius_boundary, (1,1,1), -1)
    boundary = boundary.astype("uint8")
    img = cv2.multiply(img, boundary) + 128*(1-boundary)
    return img

def crop_image(image, radius_boundary):
    # Crop to square the size of scale
    centre_h = image.shape[0]//2
    centre_w = image.shape[1]//2
    if centre_h < radius_boundary:
        image, centre_h = pad_height(image, 2*radius_boundary)
    y_min = centre_h - radius_boundary
    y_max = centre_h + radius_boundary
    x_min = centre_w - radius_boundary
    x_max = centre_w + radius_boundary
    return image[y_min:y_max, x_min:x_max]

def pad_image(image, x_min, y_min):
    h_padding, v_padding = (0,0)
    if x_min < 0:
        h_padding = -x_min
        x_min = 0
    if y_min < 0:
        v_padding = -y_min
        y_min = 0
    image = cv2.copyMakeBorder(image, v_padding, v_padding, h_padding, h_padding, cv2.BORDER_CONSTANT, value=[0,0,0])
    return image, x_min, y_min

def calc_cropbox_dim(img):
    # Sum over colour channels and threshold to give
    # segmentation map. Only look at every 100th row/col
    # to reduce compute at cost of accuracy
    stride = 100
    img_np = np.asarray(img)[::stride,::stride].sum(2) # converting to np array slow but necessary
    img_np = np.where(img_np>img_np.mean()/5, 1, 0)
    # axes[1].imshow(img_np)
    # Find nonzero rows and columns (convert back to org indexing)
    non_zero_rows = np.nonzero(img_np.sum(1))[0]*stride
    non_zero_columns = np.nonzero(img_np.sum(0))[0]*stride
    # Boundaries given first and last non zero rows/columns 
    boundary_coords = np.zeros((2,2))
    boundary_coords[:, 0] = non_zero_columns[[0, -1]] # x coords
    boundary_coords[:, 1] = non_zero_rows[[0, -1]] # y coords
    # Center is middle of the non zero values
    center = np.zeros(2)
    center[0], center[1] = np.median(non_zero_columns), np.median(non_zero_rows)
    # Radius is max boundary difference, add stride to conservatively account
    # for uncertainity due to it's use and pad by 5%
    radius = (max(boundary_coords[1] - boundary_coords[0])/2) + stride
    top_left_coord = ((center - radius)).astype(int)
    return top_left_coord[0], top_left_coord[1], round(radius)

if __name__ == "__main__":    
    # for sub_dir in os.listdir(img_dir):
    #     print(sub_dir)
    #     if "Base" in sub_dir:
    #         img_dir_preprocessed = os.path.join(img_dir, "preprocessed_448", sub_dir)
    #         preprocess_all_images(os.path.join(img_dir, sub_dir), img_dir_preprocessed, ".tif")

    data_directory = r'data\idrid'
    img_dir = os.path.join(data_directory, "Images")
    seg_dir = os.path.join(data_directory, "Segmentation", "5. Mask")
    img_preprocessed_dir = os.path.join(data_directory, "Images_Preprocessed")
    seg_preprocessed_dir = os.path.join(data_directory, "Segmentation_Preprocessed")

    preprocess_all_images(img_dir, img_preprocessed_dir, "jpg", seg_dir=seg_dir, seg_dir_preprocessed=seg_preprocessed_dir, seg_format="tif")

    # fig, axes = plt.subplots(1,3, figsize=(12,6))
    # start_time = time.time()
    # image = cv2.imread(r"C:\\Users\\rmhisje\Documents\\medical_ViT\\data\\eyePACs\\train\\train\\10_left.jpeg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # axes[0].imshow(image)
    # out = GrahamPreprocessing(image)
    # # print(image)
    # # print(time.time()-start_time)
    # axes[2].imshow(out[0])
    # titles = ["Input Image", "Threshold Mask", "Preprocessed Image"]
    # for ax, title in zip(axes, titles):
    #     ax.set_xlabel(title)
    
    # plt.show()