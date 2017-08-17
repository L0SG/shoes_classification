# Shoes classification
Simple shoes classification with ResNet50 in Keras for Deepest Hackaton Challenge
Achieves ~77% accuracy in validation dataset.

This code is written in Deepest Hackaton challenge. 

It uses pre-trained ResNet50 for transfer learning. 

It unfreezes the last 2 redisual blocks for fine-tuning.

## How to use
Requires Keras and Scikit-learn

Run train.py to fine-tune ResNet50 with the dataset.
You could use any arbitaray image dataset, by appropriately constructing your images with subfolders and modifying the class names in the code

Run inferencce.py to validate the model with test dataset.
