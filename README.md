# AIpapillae: automatic identification of Fungiform Papillae

## 📚 About
This project contains the technical materials used to achieve the results shared in our research project:  
**"Automatic Detection of Fungiform Papillae on the Human Tongue Via Convolutional Neural Networks and Identification of the Best Performing Model"**

Since image data cannot be included in this repository due to restrictions, this repository contains the necessary preprocessing and model training code. The provided scripts allow you to preprocess the images, generate labels, and train deep-learning models. You will need to provide your own dataset.

The source code can be found in the src/ folder.

The core objective of this project is to **automate the identification of the Fungiform Papillae on the human tongue**, starting with the cropping of the area of interest on the tongue, followed by enhancement and normalization of images. The Fungiform Papillae are manually annotated using **GIMP software**  to create the ground truth data for deep learning applications.


## 📝 Pipeline

1. **Make Dataset:**
   - Load and read raw data: The data_generator.py script processes images, including cropping the circular blue region, resizing to 250x250 pixels, and normalizing. The input data should be placed in a specified folder.
   - Manual annotation: mark each Fungiform papilla with a black point at the center using GIMP.
   - Extract Papillae Markers: The dots indicating the position of each papilla are extracted. This step is handled in label_generator.py.
   - Generate ground truth: The dots are transformed into "spots," which are used as the ground truth. This process is also implemented in label_generator.py.

3. **Cross-Validation Folds:**
   - Create cross-validation folds with **k=5**. This step is handled by src/df_generator.py, which prepares the dataset for cross-validation and splits the data into k-folds.

4. **Model Training:**
   - This process is implemented in src/training_papillae_CV.py, where models like classic U-Net, Optimized U-Net, and MultiResU-Net are trained using k-fold cross-validation.

5. **Model Evaluation:**
   - Loss history plots are generated for each model's fold. This evaluation is done in src/run_papillae_CV.py, which runs cross-validation on all models and computes evaluation metrics.

6. **Prediction:**
   - The best model is used to predict papillae on 175 images. This process is carried out in src/run_papillae_CV.py.

7. **Final Results:**
   - The final results and predictions are saved in output files. This is managed in src/run_papillae_CV.py.

---

## 🗂️ Project Structure
```
├── preprocessing/                       # Data preprocessing Folder
│   ├── data_generator.py                # Image Preprocessing (cropping, normalization, resizing)
│   └── data_generator.py                # Label Generation (dots and spots creation based on annotations)
│
├── src/                                  # Source Code
│   ├── Unet.py                           # Classic U-Net  and Optimized U-Net architecture
│   ├── multiUnet.py                      # MultiResU-Net architecture
│   ├── df_generator.py                   # Dataset generation and KFold cross-validation preparation
│   ├── run_papillae_CV.py                # Run cross-validation on all models and evaluate metrics
│   ├── training_papillae_CV.py           # Train U-Net, optimized U-Net, and MultiResU-Net models using K-fold cross-validation
│   ├── errors.py                         # Error and evaluation metrics 
│
├── requirements.txt                     # Python requirements  
├── README.md                            # Project documentation 
└── .gitignore                           # Git ignore file for temporary files
```

---

## 💻 Setup Your Environment

1. **Install Python:** Version 3.9.
2. **Clone or Download the project from this repository:**  
   ```
   git clone <https://github.com/rcabini/AIpapillae>
   ```
3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```
   
4. **Install the virtual environment with Anaconda:**
   ```
   conda env create -f environment.yml
      conda activate environment
   ```

## 📷 Data Preparation
Since image data cannot be included in the repository, you must prepare your dataset before running the scripts.
1. Input Folder Structure:
Place your images in a folder and provide the path to this folder in the command-line arguments. 
Your images should be labeled with the suffix -OK.png (for input images) and -MELI.png (for annotated images).
2. Running the Preprocessing Scripts:

data_generator.py processes the images by cropping and resizing them, and storing the results in the output folder.
label_generator.py generates the dots (representing papillae) and the spots (ground truth) based on the annotations.
Use the following commands to run the scripts:
```
python preprocessing/data_generator.py --input_folder <input_folder> --output_non_norm_folder <output_non_norm_folder> --output_folder <output_folder> --error_folder <error_folder>
python preprocessing/label_generator.py --input_folder <input_folder> --label_folder <label_folder> --output_dots_folder <output_dots_folder> --output_spots_folder <output_spots_folder>
```

## 🗂️ Model Weights
We have trained the following models:

Classic U-Net
Optimized U-Net
MultiResU-Net
You can download the pre-trained weights for all models from the following link:
https://drive.google.com/drive/folders/1aEp3gHVcENwYwWs2gsma5okkiPP6MenB?usp=sharing

## 💬 We're here to help!
Contact us: raffaellacabini@gmail.com & lala.chaimae.naciri@gmail.com





