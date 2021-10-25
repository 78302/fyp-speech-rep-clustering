# FYP-speech-representation-clustering
This is a Master Project: Clustering the Learned Speech Representations (APC/VQ-APC)

--------------------------

## Required Packages
Torch 1.8.1\
Numpy 1.20.2\
Matplotlib 3.4.2

--------------------------

## File Description
Model files: Define different models used in this Project\
apc.py - Defines the representation model (APC);\
vq-apc_model.py - Defines the representation model (VQ-APC);\
classifier.py - Defines the downstream task (phonetic classification) model.

Training files:Train the model and draw 'Error Rate Curve' for each trained model\
train_apc.py - Training the APC Model;\
train_vq-model.py - Training the VQ-APC Model;\
train_classifier.py - Training the phonetic classifier.

testKM.py - Use K-Means clustering to cluster the generated representations.\
functions.py - Defines basic functions such as loading pretrained models and calculating K-Means that will be utilized in above files.\
kaldiark.py - A speech processing tool.

---------------------------

## Usage
There are 4 files need to be run: train_apc.py, train_vq-model.py, train_classifier.py and testKM.py.\
To run these files, use the command: python *filename* *parameters*.\
To check the needed parameters, use the command: python *filename* -h.
