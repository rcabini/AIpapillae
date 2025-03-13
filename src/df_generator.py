import argparse
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from glob import glob
from sklearn.model_selection import KFold

def main(names, gt_path, img_path, out_path):
    framse = []
    kf = KFold(n_splits=5)
    
    for i, (train_index, test_index) in enumerate(kf.split(names)):
        k_names = [names[r] for r in test_index]
        for fname in k_names:
            base = fname.split('/')[-1]
            image = base.replace('.png', '-OK.png')
            framse.append([i, img_path+image, fname])
    df=pd.DataFrame(framse, columns=['k', 'image', 'label'])
    df.to_excel(out_path+"dataset.xlsx", index=False)
    
# ------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # patches files
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help='Path to input images')
    parser.add_argument("--labels_path", help='Path to output labels')
    parser.add_argument("--output_folder", help='Path to output folder to save xlsx file')
    args = parser.parse_args()
    
    # file name
    names = glob(gt_path+"*.png")
    main(names, args.labels_path, args.input_path, args.output_folder)
