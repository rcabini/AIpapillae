import argparse
import os
from glob import glob 
from tqdm import tqdm 
import cv2
import numpy as np
from skimage import measure, io, transform, exposure
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

def create_dots(args):
    image_path = args.input_folder
    label_path = args.label_folder
    output_path = args.output_dots_folder

    # Check if output_path exists, and create it if it doesn't
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"creare la directory: {output_path}")
    else:
        print(f"la directory esiste già: {output_path}")

    image_files = glob(os.path.join(image_path) + '/*-OK.png')
    label_files = glob(os.path.join(label_path) + '/*-MELI.png') 

    couple_images_labels_paths = [(image_file, os.path.join(label_path, os.path.basename(image_file).replace('-OK', '-MELI'))) for image_file in image_files]

    for image_file, label_file in couple_images_labels_paths:
        image = io.imread(image_file)
        label = io.imread(label_file)
        
        test = image - label
        out = test[..., 0]
        out = (out - out.min()) / (out.max() - out.min()) * 255.
        filename = os.path.basename(image_file).replace('-OK.png', '.png')
        output_file = os.path.join(output_path, filename)
        io.imsave(output_file, out.astype(np.uint8))
        
def create_spots(args):
    output_path = args.output_spots_folder
    label_path = args.output_dots_folder
    # Assicurazione che la directory di output esista
    os.makedirs(output_path, exist_ok=True)

    # Trovare tutte le immagini che finiscono con ".png"
    label_files = glob(os.path.join(label_path, '*.png'))

    # Ordinare il file del label(puntini)
    label_files.sort()

    TH = 40

    # creare un loop per creare un immagine che contiene spots per tutta la dataset
    for lbl_file in label_files:
        base = lbl_file.split('/')[-1]
        print(base)

        label = io.imread(lbl_file)
        centers = peak_local_max(label, min_distance=2)
        output = np.zeros(label.shape+(len(centers),))
        for k, (x, y) in enumerate(centers):
          lab = np.zeros(label.shape)
          lab[x,y] = 255. 
          output[:,:,k]  = cv2.GaussianBlur(lab, ksize=(0,0), sigmaX=5, borderType=cv2.BORDER_ISOLATED)
        smooth = np.max(output, axis=-1) 

        # Normalizzo l'output
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min()) * 255.

        # Costruisco il nome del file di output
        output_file = os.path.join(output_path, base)

        # Salvo l'immagine di output
        io.imsave(output_file, smooth.astype(np.uint8))

        print(f"Salvato: {output_file}")

if __name__ == "__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help='Path to input folder')
    parser.add_argument("--label_folder", help='Path to label folder')
    parser.add_argument("--output_dots_folder", help='Path to output dots folder')
    parser.add_argument("--output_spots_folder", help='Path to output spots folder')
    args = parser.parse_args()
    
    create_dots(args)
    create_spots(args)
