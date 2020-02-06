# Kuzushiji-Recognition
Kuzushiji Recognition project hosted by Kaggle.

Kuzushiji is an ancient historical text that is almost going into extinct where only less than 0.1% of the Japanese natives can read it fluently.
Hence, with the help of machine learning & deep learning, the aim and goal for this project is to build and train a model that'll detect and classify the Kuzushiji texts into modern Japanese characters as accurate as possible.

To learn more about Kuzushiji Recognition please refer to the link below :
https://www.kaggle.com/c/kuzushiji-recognition

# Overview 
This project is divided into 3 stages. 
Detection, Classification, Pseudolabelling

The visualization of the final result of this model for a random test image as shown beow:
![alt text](https://github.com/wintersin/Kuzushiji-Recognition/raw/master/for_display/example.png)

The final F1 score for the overall model is 0.89 or roughly 89%. Accuray can be improved by training with deeper network.
## Detection
* Kuzushiji characters are detected with a Faster-RCNN model by using ```resnet50``` as backbone trained with torchvision. Only single class is used so it doesn't predict the character class.
* Trains moderately fast and accurate which gives a F1 score around ~0.95 on validation.
* Original page height was around 1500 px, model is trained on 512x384 cropped images and full pages were used for inference
* Albumentations library was used for augmentations. Augmentations used were LongestMaxSize, RandomSizedCrop, HueSaturationValue, RandomBrightnessContrast, RandomGamma

## Classification
* Kuzushijii characters classification is trained on a network which use ```resnet50``` as base from scratch.
* Takes input from a 512x768 cropped image which contains multiple characters and bounding boxes predicted from the detection model. 
* Bounding boxes are frozen, ```layer4``` is discarded, features are extracted for bounding box with ```roi_align``` from ```layer2``` & ```layer3``` and concatenated together, and passed into a classification head with 2 fully connected layer.
* Mixed precision training was used to speed up the training process by freezing the first convolution and whole layer1. 
* Albumentations library was used for augmentations. Augmentations used were LongestMaxSize, RandomSizedCrop, HueSaturationValue, RandomBrightnessContrast, RandomGamma
* Test Time Augmentation (TTA) technique was used. 4 different scales was used.

## Pseudolabelling
* A semi-supervised technique was used to boost the accuracy of the model.
* Confident classification predictions of the model on test data with accuracy more than 95% on local validation were taken and add back to the training dataset and retrain.
* This will allow the model to adapt and generalize better on different paper style, different handwritting styles. 
* Classification model is fine-tuned for 5 epcohs with starting learning rate 10 times smaller than the inital learning rate.
* A result csv file is created after the fine-tuning.

## Visualization
Different styles of pages as demo:
![alt text](https://github.com/wintersin/Kuzushiji-Recognition/raw/master/for_display/example3.png)![alt text](https://github.com/wintersin/Kuzushiji-Recognition/raw/master/for_display/example2.png)![alt text](https://github.com/wintersin/Kuzushiji-Recognition/raw/master/for_display/example1.png)

# Requirements
All the codes are written in Python 3.6.


Install the requirements pacakages: 
``` 
pip install -r requirements.txt
python setup.py develop
``` 
Install libjpeg-turbo for your OS as documented at https://github.com/ajkxyz/jpeg4py

Install apex as documented at https://github.com/NVIDIA/apex (python-only is already good enough)

All the models are written in Pytorch while the detection models in ```frcnn``` folder are based on torchvision. Albumentations is for augmentation purposes and Apex is used for mixed precision training.

# Run
For detection stage:
```
cd sh
./runner.sh
```

For classification stage:
```
cd sh
./classify.sh
```

For pseudolabelling stage:
```
cd sh
./pseudo.sh
```
# Hardware 
All the models are trained on GCP with one Nvidia V100 GPU. Due to the credit limits, all the backbones are trained with ```resnet50``` only as training larger and deeper networks required huge amount of resources which I don't have.
