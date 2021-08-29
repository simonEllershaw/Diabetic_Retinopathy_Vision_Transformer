import os
import numpy as np
from cv2 import cv2

def preprocess_all_images(img_dir, image_format, img_dir_preprocessed, preprocessed_size, store_crop_boxes=False):
    # Create directories
    if not os.path.exists(img_dir_preprocessed):
        os.makedirs(img_dir_preprocessed)
    crop_boxes={}

    # Preprocess each image
    for fname in os.listdir(img_dir):
        if fname.endswith(image_format):
            # Read in, calc crop box, preprocess and save to file
            img = cv2.imread(os.path.join(img_dir, fname))
            x_min, y_min, radius_inital = calc_cropbox_dim(img)
            if radius_inital < 10:
                raise ValueError("Image radius could not be found")
            img = preprocess_img(img, x_min, y_min, radius_inital, preprocessed_size)
            cv2.imwrite(os.path.join(img_dir_preprocessed, fname), img)
            # If seg maps use crop box defined by RGB image
            if store_crop_boxes:
                # Remove file type from fname
                fname = fname[:-len(image_format)]
                crop_boxes[fname] = {"x_min": x_min, "y_min": y_min, "radius_inital": radius_inital}
    return crop_boxes    

def preprocess_img(img, x_min, y_min, radius_inital, preprocessed_size):
    # Pad so crop box in RGB image
    img, x_min_pad, y_min_pad = pad_image(img, x_min, y_min)
    # Crop
    img = img[y_min_pad:y_min_pad+radius_inital*2, x_min_pad:x_min_pad+radius_inital*2]
    # Resize
    img = cv2.resize(img, [preprocessed_size, preprocessed_size])
    return img

def pad_image(image, x_min, y_min):
    # If x_min or y_min pad so they are at 0
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

def preprocess_seg_map(fname, seg_sub_dir, seg_dir_preprocessed, x_min, y_min, radius_inital, preprocessed_size):
    seg_map = create_seg_map(fname, seg_sub_dir)
    seg_map_preprocessed = preprocess_img(seg_map, x_min, y_min, radius_inital, preprocessed_size)
    cv2.imwrite(os.path.join(seg_dir_preprocessed, fname + seg_image_format), seg_map_preprocessed)

def create_seg_map(fname, seg_dir):
    # Define lesion directories
    dir_microaneurysms = os.path.join(seg_dir, "1. Microaneurysms")
    dir_haemorrhages = os.path.join(seg_dir, "2. Haemorrhages")
    dir_hard_exudates = os.path.join(seg_dir, "3. Hard Exudates")
    dir_soft_exudates = os.path.join(seg_dir, "4. Soft Exudates")
    
    # Add each lesion to channel of seg map (BGR as using cv2)
    # Red
    seg_map = cv2.imread(os.path.join(dir_microaneurysms, f"{fname}_MA.tif"))
    # Green
    haemorrhages_fname = os.path.join(dir_haemorrhages, f"{fname}_HE.tif")
    seg_map = add_lesion_to_seg_map(seg_map, haemorrhages_fname, [1])
    # Blue
    hard_exudates_fname = os.path.join(dir_hard_exudates, f"{fname}_EX.tif")
    seg_map = add_lesion_to_seg_map(seg_map, hard_exudates_fname, [0])
    # Pink
    soft_exudates_fname = os.path.join(dir_soft_exudates, f"{fname}_SE.tif")
    seg_map = add_lesion_to_seg_map(seg_map, soft_exudates_fname, [0,2])
    # Clip required due to pink channel
    seg_map = np.clip(seg_map, 0, 255)
    return seg_map

def add_lesion_to_seg_map(seg_map, lesion_fname, channels):
    # All lesions held in red channel add to segmap channel of choice
    if os.path.isfile(lesion_fname):
        for channel in channels:
            seg_map[:,:,channel] += cv2.imread(lesion_fname)[:,:,2]
    return seg_map

if __name__ == "__main__":
    preprocessed_size = 448

    # # EyePACs Preprocessing
    data_directory = os.path.join("data", "eyePACs")
    img_dir = os.path.join(data_directory, "train", "train")
    image_format = ".jpeg"
    img_dir_preprocessed = os.path.join(data_directory, "preprocessed_images")
    preprocess_all_images(img_dir, image_format, img_dir_preprocessed, preprocessed_size)

    # Messidor Preprocessing
    data_directory = os.path.join("data", "messidor")
    image_format = ".tif"
    preprocessed_size = 448
    for sub_dir in os.listdir(data_directory):
        img_dir = os.path.join(data_directory, sub_dir)
        if os.path.isdir(img_dir) and "Base" in sub_dir:
            img_dir_preprocessed = os.path.join(data_directory, "preprocessed_images", sub_dir)
            preprocess_all_images(img_dir, image_format, img_dir_preprocessed, preprocessed_size)

    # IDRiD Processing- Includes creating and preprocessing seg_maps as well
    data_directory = os.path.join("data", "idrid")
    img_dir = os.path.join(data_directory, "A. Segmentation", "A. Segmentation", "1. Original Images")
    img_dir_preprocessed = os.path.join(data_directory, "preprocessed_images")
    image_format = ".jpg"
    seg_dir = os.path.join(data_directory, "A. Segmentation", "A. Segmentation", "2. All Segmentation Groundtruths")
    seg_dir_preprocessed = os.path.join(data_directory, "preprocessed_seg")
    seg_image_format = ".tif"

    for sub_dir in os.listdir(img_dir):
        img_sub_dir = os.path.join(img_dir, sub_dir)
        seg_sub_dir = os.path.join(seg_dir, sub_dir)
        crop_boxes = preprocess_all_images(img_sub_dir, image_format, img_dir_preprocessed, preprocessed_size, store_crop_boxes=True)
        for fname, crop_boxes_dim in crop_boxes.items():
            x_min, y_min, radius_inital = crop_boxes_dim.values()
            preprocess_seg_map(fname, seg_sub_dir, seg_dir_preprocessed, x_min, y_min, radius_inital, preprocessed_size)