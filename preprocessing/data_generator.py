import argparse
import os
from glob import glob 
from tqdm import tqdm 
import cv2
import numpy as np
from skimage import measure, io, transform, exposure
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

def get_cdf_hist(image_input):
    """
    Method to compute histogram and cumulative distribution function
    :param image_input: input image
    :return: cdf
    """
    #image_input = image_input[image_input > 15]
    hist, bins = np.histogram(image_input.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return cdf_normalized

def clahe_enhancement(image, threshold, grid_size=(16, 16)):
    """
    Adaptive histogram equalization to enhance the input image
    :param image: input image
    :param threshold: clipping threshold
    :param grid_size: local neighbourhood
    :return: enhanced image
        """
    clahe = cv2.createCLAHE(clipLimit=threshold, tileGridSize=grid_size)
    enhanced_image = clahe.apply(image)
    cdf = get_cdf_hist(enhanced_image)
    return enhanced_image, cdf

def process_image(input_image_path, sub_processed_dir, processed_dir, NoEllipse_found_folder, process_if_exists=False):
    I_OK = cv2.imread(input_image_path)
    I_gray = cv2.cvtColor(I_OK, cv2.COLOR_BGR2GRAY)
    TH_BLACK = 10
    _, bw = cv2.threshold(I_gray, TH_BLACK, 255, cv2.THRESH_BINARY_INV)

    labels = measure.label(bw, connectivity=2)
    props = measure.regionprops(labels)

    max_fitness = -1
    best_prop = None
    for prop in props:
        bb = prop.bbox
        width = bb[3] - bb[1]
        height = bb[2] - bb[0]
        fitness = (width * height) / (1 + abs(width - height))
        if fitness > max_fitness:
            max_fitness = fitness
            best_prop = prop
    if best_prop is None:
        logger.error(f"--> !!!!! No suitable component found in image: {input_image_path}")

        if not os.path.exists(NoEllipse_found_folder):
            os.makedirs(NoEllipse_found_folder)
        save_path = os.path.join(NoEllipse_found_folder, os.path.basename(input_image_path))
        cv2.imwrite(save_path, I_OK)
        return

    # Get bounding box and adjust to square.
    min_row, min_col, max_row, max_col = best_prop.bbox
    width = max_col - min_col
    height = max_row - min_row
    size = max(width, height)

    # Center the bounding box.
    center_row = min_row + height // 2
    center_col = min_col + width // 2

    # Calculate new bounding box coordinates
    new_min_row = max(center_row - size // 2, 0)
    new_max_row = min(center_row + size // 2, I_OK.shape[0])
    new_min_col = max(center_col - size // 2, 0)
    new_max_col = min(center_col + size // 2, I_OK.shape[1])

    # Create an elliptical mask
    mask = np.zeros_like(I_gray, dtype=np.uint8)
    ellipse_center = (center_col, center_row)
    ellipse_axes = (width // 2, height // 2)
    cv2.ellipse(mask, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1) 

    # Crop the image to the bounding box
    I_OK = I_OK[new_min_row:new_max_row, new_min_col:new_max_col]
    mask = mask[new_min_row:new_max_row, new_min_col:new_max_col]
    
    # Save non normalized image
    I_out = np.zeros_like(I_OK)
    I_out[mask != 0] = I_OK[mask != 0]
    resized = cv2.resize(I_out, (250, 250), interpolation=cv2.INTER_AREA)
    if not os.path.exists(sub_processed_dir):
        os.makedirs(sub_processed_dir)
    save_path = os.path.join(sub_processed_dir, os.path.splitext(os.path.basename(input_image_path))[0] + ".png")
    cv2.imwrite(save_path, resized)
    print(f"Processed and saved image: {save_path}")

    #Clache RGB
    I_masked = np.zeros_like(I_OK)
    img = cv2.cvtColor(I_OK, cv2.COLOR_RGB2Lab)
    clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(2,2))
    img[:,:,0] = clahe.apply(img[:,:,0])
    I_OK = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    I_masked[mask != 0] = I_OK[mask != 0]
    
    # Resize to 250x250
    resized = cv2.resize(I_masked, (250, 250), interpolation=cv2.INTER_AREA)

    # Save the processed image
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    save_path = os.path.join(processed_dir, os.path.splitext(os.path.basename(input_image_path))[0] + ".png")
    cv2.imwrite(save_path, resized)
    print(f"Processed and saved image: {save_path}")

if __name__ == "__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help='Path to input folder')
    parser.add_argument("--output_non_norm_folder", help='Path to output folder')
    parser.add_argument("--output_folder", help='Path to output folder')
    parser.add_argument("--error_folder", help='Path to output error folder')
    args = parser.parse_args()
    
    for image_path in glob(os.path.join(args.input_folder,'*')):
        process_image(input_image_path=image_path,
                      sub_processed_dir=args.output_non_norm_folder,
                      processed_dir=args.output_folder,
                      NoEllipse_found_folder=args.error_folder,
                      process_if_exists=True
                      )
                      
                      
