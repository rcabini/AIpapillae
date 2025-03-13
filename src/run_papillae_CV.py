import argparse
import os, sys
#sys.path.insert(0,'../')
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import img_to_array, load_img
import imageio
import matplotlib.pyplot as plt
import skimage.transform as trans
import skimage.io as io
import glob
import pandas as pd
import skimage.io as io
from skimage.feature import peak_local_max
from training_papillae_CV import train_generator
import cv2
from skimage import metrics
from tqdm import tqdm
from errors import error_dice, error_papillae
    
#---------------------------------------------------------------------------

def plot_results(image, labels, gtlabels, dstFolder, base):
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(gtlabels, vmin=0, vmax=255)
    ax[1].set_title('Groud-Truth')
    ax[2].imshow(labels, vmin=0, vmax=255)
    ax[2].set_title('Predicted')
    for a in ax:
        a.axis("off")
    plt.savefig(dstFolder + os.path.sep + "{}.png".format(base), bbox_inches='tight', dpi=200)
    plt.close()

#---------------------------------------------------------------------------

def main(DATA_PATH, WEIGHTS_DIR):

    pred_path = os.path.join(WEIGHTS_DIR, 'pred')
    res_path = os.path.join(WEIGHTS_DIR, 'res')
    err_path = os.path.join(WEIGHTS_DIR, 'err')
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(err_path, exist_ok=True)
    
    K_FOLDS = 5
    WINDOW_SIZE = (512, 512, 3) 
    
    for k in range(K_FOLDS):
        print("Fold {}/{}".format(k+1, K_FOLDS))
        df = pd.read_excel(DATA_PATH)
        df = df[[os.path.isfile(i) for i in df['image']]]
        test_df = df[df['k']==k][["image", "label"]]
        
        tf.keras.backend.clear_session()
        MODEL_PATH = WEIGHTS_DIR +os.path.sep+'Unet_kfold{}.hdf5'.format(k)
        model = tf.keras.models.load_model(MODEL_PATH)
        
        ssim, mae, dice, errTP = [], [], [], []
        nTP, nFN, nFP, nN = 0, 0, 0, 0
        i, b = 0, 0
        for index, row in tqdm(test_df.iterrows()):
            base = row['image'].split('/')[-1].split('.png')[0]
            x_img0 = io.imread(row['image'], as_gray = False)
            im_height, im_width = x_img0.shape[0], x_img0.shape[1]
            x_img = load_img(row['image'], color_mode="rgb", target_size=WINDOW_SIZE[:-1])
            x_img = img_to_array(x_img)
            X = np.expand_dims(x_img, axis=0)
            X = X.astype(float) / 255.
            
            x_mask = io.imread(row['label'], as_gray = True)
            
            results = model.predict(X, verbose=0)
            results = trans.resize(results, (1, im_height, im_width, results.shape[-1]), anti_aliasing=True)
            res = results[0,:,:,0]*255.
            res[res<0.]=0.
            res[res>255.]=255.
        
            plt.imsave(os.path.join(pred_path,'{}.png'.format(base)),(res).astype(np.uint8), vmin=0, vmax=255)
            plot_results(x_img0, res, x_mask, res_path, base)
            
            #Structural similarity index
            ssim_idx = metrics.structural_similarity(x_mask, res, full=True, data_range=256)
            ssim.append(ssim_idx[0])
            #MAE
            mae.append(np.abs(x_mask/255.-res/255.).mean())        
            
            #Normalization
            gt = x_mask / 255.
            pred_orig = res / 255.
            pred = pred_orig.copy()
            pred[pred<0.15] = 0.
            
            #Dice
            dice.append(error_dice(gt, pred, 10*2)) # Threshold sulla grandezza della papilla
            dist_TP, TP_gt, TP_pred, FN, FP, N = error_papillae(gt, pred, 10*2) # Threshold sulla grandezza della papilla
            if N!=0:
                nN += N
                nTP += len(TP_gt)
                nFN += len(FN)
                nFP += len(FP)
                errTP.append(np.mean(dist_TP))
            
            coordinates_TP = np.array(TP_pred)
            coordinates_FN = np.array(FN)
            coordinates_FP = np.array(FP)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(x_img0)
            ax[0].autoscale(False)
            ax[0].axis('off')
            ax[0].set_title('Input Image')
            ax[1].imshow(gt, cmap=plt.cm.gray)
            ax[1].autoscale(False)
            ax[1].axis('off')
            ax[1].set_title('GT')
            ax[2].imshow(pred, cmap=plt.cm.gray)
            ax[2].autoscale(False)
            if len(coordinates_TP)!=0:
                ax[2].plot(coordinates_TP[:, 0], coordinates_TP[:, 1], 'g.', label='True positive')
            if len(coordinates_FP)!=0:
                ax[2].plot(coordinates_FP[:, 0], coordinates_FP[:, 1], 'r.', label='Misdetected papillae')
            if len(coordinates_FN)!=0:
                ax[2].plot(coordinates_FN[:, 0], coordinates_FN[:, 1], 'b.', label='False negative')
            ax[2].axis('off')
            ax[2].set_title('Pred')
            ax[2].legend()
            fig.tight_layout()
            plt.savefig(os.path.join(err_path, '{}.png'.format(base)))
            plt.close()
        
        print(DATA_PATH)
        print(MODEL_PATH)
        print('SSIM', np.mean(ssim), 'MAE', np.mean(mae), 'Dice', np.mean(dice))
        print('Standard deviation: ','SSIM', np.std(ssim), 'MAE', np.std(mae), 'Dice', np.std(dice))
        print('numero totale di cellule vere', nN)
        errs = np.array(errTP)
        print('errore medio TP', np.mean(errs[~np.isnan(errs)]), np.std(errs[~np.isnan(errs)]))
        print('numero Veri Positivi', nTP)
        print('numero Falsi Negativi', nFN)
        print('numero Falsi Positivi', nFP)
        tf.keras.backend.clear_session()
        del model
    
if __name__=="__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help='Path to xlsx file')
    parser.add_argument("--weights_folder", help='Path to input weights')
    args = parser.parse_args()
    main(args)
    
    DATA_PATH = args.input_file
    WEIGHTS_DIR = os.path.join(args.weights_folder, 'unet_weights')
    main(DATA_PATH, WEIGHTS_DIR)
    print('***********************************')

    WEIGHTS_DIR = os.path.join(args.weights_folder, 'unet_optimized_weights')
    main(DATA_PATH, WEIGHTS_DIR)
    print('***********************************')

    WEIGHTS_DIR = os.path.join(args.weights_folder, 'multires_weights')
    main(DATA_PATH, WEIGHTS_DIR)
    print('***********************************')
