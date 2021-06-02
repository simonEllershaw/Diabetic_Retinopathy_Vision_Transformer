from PIL import ImageFilter, Image
import numpy as np
import math
import cv2  

def GrahamPreprocessing(image, radius_intial):
    radius_scaled = 300       
    image = rescale_image(image, radius_intial, radius_scaled)
    image = subtract_average_local_colour(image)
    radius_boundary = round(radius_scaled*0.9)
    image = threshold_boundary(image, radius_boundary)
    # image = crop_image(image, radius_boundary)
    return image

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

def threshold_boundary(image, radius_boundary):
    boundary = np.zeros(image.shape)
    cv2.circle(boundary, (image.shape[1]//2, image.shape[0]//2), radius_boundary, (1,1,1), -1)
    return cv2.multiply(image, boundary.astype("uint8"))

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

def pad_height(image, desired_h):
    h_padding = math.ceil((desired_h-image.shape[0])/2.0)
    image = cv2.copyMakeBorder(image, h_padding, h_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    return image, image.shape[0]//2

if __name__ == "__main__":
    start_time = time.time()
    image = cv2.imread(r"C:\\Users\\rmhisje\Documents\\medical_ViT\\diabetic-retinopathy-detection\\train\\train\\16_left.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = GrahamPreprocessingCV2(image)
    print(image.min())
    print(time.time()-start_time)
    axes[1].imshow(image)
    plt.show()