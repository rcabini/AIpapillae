# AIpapillae: automatic identification of Fungiform Papillae

## 📚 About
This project contains the technical materials used to achieve the results shared in our research project:  
**"Automatic Detection of Fungiform Papillae on the Human Tongue by Deep Learning Approaches."**

By leveraging the code and model within this repository, you can re-run the deep-learning models, conduct experiments, or even use it as a starting point for developing your own model. The source code can be found in the `src/` folder.  

The core objective is to **automate the identification of the Fungiform Papillae on the human tongue**, starting with the cropping of the area of interest on the tongue, followed by enhancement and normalization of images. Subsequently, the Fungiform Papillae are manually annotated using **GIMP software** to create the ground truth data for deep learning applications, specifically for training DL models.  


## 📝 Pipeline

1. **Make Dataset:**
   - Load and read raw data.
   - Preprocess images: crop the circular blue region, resize to 250x250 pixels, and normalize.
   - Manual annotation: mark each Fungiform papilla with a black point at the center using GIMP.
   - Extract Papillae Markers: isolate only the black points indicating each papilla position.
   - Generate ground truth: transform the black points into spots.

2. **Cross-Validation Folds:**
   - Create cross-validation folds with **k=5**.

3. **Model Training:**
   - Train deep learning models using the prepared dataset.

4. **Model Evaluation:**
   - Plot figures of the loss history for each model's fold.

5. **Prediction:**
   - Automatically predict Fungiform Papillae on 175 images using the best model.

6. **Final Results:**
   - Save final results and model outputs.

---

## 🗂️ Project Structure
```
├── data/                                # Data Folder
│   ├── raw/                             # Original data (non-modified)
│   └── processed/                       # Pre-processed data (cropped, normalized, resized)
│   └── ground_truth/                    # Annotated Papillae and Generation of Spots
│
├── models/                              # Trained models for each K-Fold
│   ├── unet_fold1.h5  
│   ├── unet_fold2.h5  
│   ├── unet_fold3.h5  
│   ├── unet_fold4.h5  
│   ├── unet_fold5.h5   
│   ├── multiresunet_fold1.h5  
│   ├── multiresunet_fold2.h5  
│   ├── multiresunet_fold3.h5  
│   ├── multiresunet_fold4.h5  
│   ├── multiresunet_fold5.h5   
│
├── src/                                  # Source Code
│   ├── dataset.py                        # Data loading and pre-processing  
│   ├── classic_unet.py                   # Classic U-Net architecture
│   ├── multiresunet.py                   # MultiResU-Net architecture
│   ├── optimized_unet.py                 # Optimized U-Net architecture
│   ├── train_kfold.py                    # Training with K-Fold Cross Validation 
│   ├── history_kfold.py                  # Training history of each fold 
│   ├── evaluate_kfold.py                 # Evaluation of each fold 
│   └── predict.py                        # Prediction with the best model
│
├── requirements.txt                     # Python requirements  
├── README.md                            # Project documentation 
└── .gitignore                           # Git ignore file for temporary files
```

---

## 💻 Setup Your Environment

1. **Install Python:** Version 3 or higher.  
2. **Install the virtual environment:**   
3. **Clone or Download the project from this repository:**  
   ```
   git clone <https://github.com/lala-sudo/Papillae-project>
   ```
4. **Install required Python libraries:**  

