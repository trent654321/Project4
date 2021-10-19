# Breast Cancer Detection using CNN
Data Visualization - Project Four
October 19, 2021

Trent McNabb & Scott Stimpson

## Introduction

Build a breast cancer classifier on the IDC dataset to identify histology images as benign or malignant using Convolutional Neural Network. Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions which contain the IDC. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of IDC inside of a whole mount slide.

## Data

The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. 

277,524 patches of size 50 x 50 were extracted (198,738 IDC negative and 78,786 IDC positive). 

File name format: uxXyYclassC.png — >  10253idx5x1351y1101class0.png 

u = patient ID (10253idx5) 
X = x-coordinate of where this patch was cropped
Y = y-coordinate of where this patch was cropped 
C = the class where 0 is non-IDC and 1 is IDC

### Cleaning & Preprocessing

Using the files names & Keras.preprocessing;

    Split the images into “Training” (80%) and “Testing” (20%) datasets, 
    Label each image data as 0: benign or, 1: malignant
    Using Numpy - transform images into array
    Normalize the array by dividing by 255

## Model
### Training
Training;
    Sequential Class Model
    3 layers of Conv2D & MaxPooling2D
    Activation Relu
    Flatten 3D features into 1D feature vectors
    Activation changed to Softmax for output
### Testing 
Testing;
    Loss - Sparce_categorical_crossentropy
    Optimizer - Adam