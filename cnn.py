#import libraries and dependencies
import tensorflow as tf
import pickle
import matplotlib
import numpy as np

#dataFile_training = "" 
#need filepath for data

#dataFile_testing = "" 
#need filepath for data

	
#loading the dataset with both training and testing

with open(dataFile_training, mode = 'rb') as file:
	training_data = pickle.load(file)

with open(dataFile_testing, mode - 'rb') as file:
	testing_data = pickle.load(file)
	
X_training_data, y_training_data = training_data['features'], training_data['labels']
X_testing_data, y_testing_data = testing_data['features'], testing_data['labels']

#finished loading dataset

import cv2 #importing computer vision library to help grayscale all of the images

def grayScale(image_data):
	grayscaled = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
	return grayscaled
	
X_training_data = np.array([grayScale(image_data) for image_data in X_training_data]) #grayscales images in training data
X_testing_data = np.array([grayScale(image_data for image_data in X_testing_data]) #grayscales images in testing dataset