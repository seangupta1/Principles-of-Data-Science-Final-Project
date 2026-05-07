# Deep Fake Detection Project

**Sean Gupta**  
**Josh Oursler**  
**Ruixuan Hou**  
**Haoran Zhang**  

University of Missouri–Kansas City (UMKC)

---

# Project Overview

This project focuses on detecting deep fake facial images using deep learning and computer vision techniques. The system was designed to classify images as either **Real** or **Fake** using a supervised learning approach trained on the OpenForensics dataset.

The project combines image preprocessing, face detection, data augmentation, transfer learning, model evaluation, and explainable AI techniques to create a robust deep fake detection pipeline.

The final model achieved strong performance with approximately **95.8% testing accuracy** and an **ROC-AUC score of 0.9716**.

---

# Objective

The goal of this project is to build a machine learning system capable of identifying manipulated or AI-generated facial images. Deep fake technology has become increasingly advanced and accessible, creating challenges related to misinformation, identity fraud, and digital media authenticity.

This project applies Data Science and Deep Learning techniques to analyze facial images and determine whether they are genuine or manipulated.

---

# Dataset

**Dataset Used:** OpenForensics Dataset  

Kaggle Link:  
https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

---

# Data Cleaning and Preprocessing

- Images were loaded directly from zip files
- Faces were detected using OpenCV
- Detected faces were cropped from the original images
- Images were resized to 150x150 pixels while keeping aspect ratio
- Padding was added when needed
- Images were converted into tensors and normalized for training

---

# Model Architecture

- ResNet50 was used as the feature extractor
- The final classification layer was replaced
- Custom fully connected layers were added
- ReLU activation and dropout were used
- Output predicts whether an image is real or fake

---

# Training Configuration

## Hyperparameters

- **Epochs:** 30
- **Batch Size:** 128
- **Learning Rate:** 0.0001
- **Optimizer:** AdamW
- **Loss Function:** BCEWithLogitsLoss
- **Scheduler:** StepLR

## Training Features

- Validation performed every epoch
- Early stopping used to prevent overfitting
- Best model saved automatically

---

# Model Performance

## Results

- **Accuracy:** 95.8%
- **ROC-AUC:** 0.9716
- **Test Loss:** 0.12

## Additional Metrics

- **Precision:** 93.89%
- **Recall:** 87.67%
- **Specificity:** 93.96%

---

# Visualizations Generated

- Loss curves
- Accuracy curves
- ROC curve
- Confusion matrix
- Grad-CAM heatmaps
- Probability distribution graphs

---

# Project Pipeline

1. Load images
2. Detect and crop faces
3. Resize and normalize images
4. Apply augmentations
5. Train the model
6. Evaluate performance
7. Generate Grad-CAM visualizations
