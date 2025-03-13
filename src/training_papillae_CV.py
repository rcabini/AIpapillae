import argparse
import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras         # Keras must be loaded from tensorflow directly
from tensorflow.keras import layers  # Layers
from tensorflow.keras.optimizers import Adam # Optimizer 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Unet import *
from multiUnet import *

#---------------------------------------------------------------------------

def train_generator(data_frame, batch_size, aug_dict,
                    target_size,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    flag_multi_class=True,
                    save_to_dir=None,
                    seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "image",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "label",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        #img, mask = adjust_data(img, mask)
        yield (img,mask)

#---------------------------------------------------------------------------
        
train_generator_args = dict(rescale=1./255., #intensity normalization
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            rotation_range=90.,
                            shear_range=5.,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='constant',
                            cval = 0.0)
             
#---------------------------------------------------------------------------
                     
def plot_history(history, results_path, k):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.semilogy(history.history['loss'], label='Train')
    plt.semilogy(history.history['val_loss'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.semilogy(history.history['mean_absolute_error'], label='Train')
    plt.semilogy(history.history['val_mean_absolute_error'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.savefig(os.path.join(results_path, 'history_kfold{}.png'.format(k)), bbox_inches = "tight", dpi=200)
    
#---------------------------------------------------------------------------

def load_dataset(exceldir, k, WINDOW_SIZE, BATCH_SIZE):
    # Load X and Y vectors
    df = pd.read_excel(exceldir)
    df = df[[os.path.isfile(i) for i in df['image']]]
    df = df[[os.path.isfile(i) for i in df['label']]]
    df = df[df['k']!=k][["image", "label"]]

    unique_ids = np.arange(len(df))
    train_nam, val_nam = train_test_split(unique_ids, test_size=0.1, random_state=1)
    
    train_df = df.iloc[train_nam, :]  
    print(train_df)  
    val_df = df.iloc[val_nam, :]    
    print(val_df)  

    train_gen = train_generator(train_df, BATCH_SIZE, train_generator_args, WINDOW_SIZE)
    #train_gen = train_generator(train_df, BATCH_SIZE, dict(rescale=1./255), WINDOW_SIZE)
    val_gen = train_generator(val_df, BATCH_SIZE, dict(rescale=1./255.), WINDOW_SIZE)

    SET_SIZE = (train_df.shape[0], val_df.shape[0])
    print("****************************")
    print(SET_SIZE)
    print("****************************")
    return train_gen, val_gen, SET_SIZE

#---------------------------------------------------------------------------

def train_model(model, train_gen, val_gen, BATCH_SIZE, WEIGHTS_DIR, DATASET_SIZE, epochs, k):
    #Add a callback for saving model
    model_checkpoint = ModelCheckpoint(WEIGHTS_DIR +os.path.sep+'Unet_kfold{}.hdf5'.format(k), monitor='val_mean_absolute_error', save_best_only=True)
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate = 1e-3), metrics=["mean_absolute_error"])
    
    STEPS_PER_EPOCH = DATASET_SIZE[0] // BATCH_SIZE
    VALIDATION_STEPS = DATASET_SIZE[1] // BATCH_SIZE
    history=model.fit(
                      train_gen,
                      epochs = epochs,
                      steps_per_epoch=STEPS_PER_EPOCH,
                      validation_steps=VALIDATION_STEPS,
                      batch_size = BATCH_SIZE,
                      verbose = 1,
                      validation_data = val_gen,
                      callbacks = [model_checkpoint]
                     )
    
    return model, history
    
#---------------------------------------------------------------------------

def main(args):    
    
    BATCH_SIZE = 2
    EPOCHS = 500 #200 #200
    WINDOW_SIZE = (512, 512, 3)
    K_FOLDS = 5
    
    tf.keras.backend.clear_session()
    DATA_PATH = args.input_file
    WEIGHTS_DIR = os.path.join(args.output_folder, 'unet_weights')
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    for k in range(K_FOLDS):
        print("Fold {}/{}".format(k+1, K_FOLDS))
            
        train_gen, val_gen, DATASET_SIZE = load_dataset(DATA_PATH, k, (WINDOW_SIZE[0],WINDOW_SIZE[1]), BATCH_SIZE)
            
        model = unet(WINDOW_SIZE)
        print(model.summary())
            
        #Fit the model
        model, history = train_model(model, train_gen, val_gen, BATCH_SIZE, WEIGHTS_DIR, DATASET_SIZE, EPOCHS, k)
        plot_history(history, WEIGHTS_DIR, k)
        tf.keras.backend.clear_session()
        #break
        del model

    #---------------------------------------------------------------------------

    tf.keras.backend.clear_session()
    DATA_PATH = args.input_file
    WEIGHTS_DIR = os.path.join(args.output_folder, 'unet_optimized_weights')
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for k in range(K_FOLDS):
        print("Fold {}/{}".format(k+1, K_FOLDS))
        
        train_gen, val_gen, DATASET_SIZE = load_dataset(DATA_PATH, k, (WINDOW_SIZE[0],WINDOW_SIZE[1]), BATCH_SIZE)
        
        model = unet_optimized(WINDOW_SIZE)
        print(model.summary())
        
        #Fit the model
        model, history = train_model(model, train_gen, val_gen, BATCH_SIZE, WEIGHTS_DIR, DATASET_SIZE, EPOCHS, k)
        plot_history(history, WEIGHTS_DIR, k)
        tf.keras.backend.clear_session()
        #break
        del model
    
    #---------------------------------------------------------------------------
    
    tf.keras.backend.clear_session()
    DATA_PATH = args.input_file
    WEIGHTS_DIR = os.path.join(args.output_folder, 'multires_weights')
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for k in range(K_FOLDS):
        print("Fold {}/{}".format(k+1, K_FOLDS))

        train_gen, val_gen, DATASET_SIZE = load_dataset(DATA_PATH, k, (WINDOW_SIZE[0],WINDOW_SIZE[1]), BATCH_SIZE)

        model = MultiResUnet(height=WINDOW_SIZE[0], width=WINDOW_SIZE[1], n_channels=WINDOW_SIZE[2])
        print(model.summary())

        #Fit the model
        model, history = train_model(model, train_gen, val_gen, BATCH_SIZE, WEIGHTS_DIR, DATASET_SIZE, EPOCHS, k)
        plot_history(history, WEIGHTS_DIR, k)
        tf.keras.backend.clear_session()
        #break
        del model
    
if __name__=="__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help='Path to xlsx file')
    parser.add_argument("--output_folder", help='Path to output weights')
    args = parser.parse_args()
    main(args)
    
    
