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
	training_data = pickle.load(file)#reads from the open file stream after loading the training data file

with open(dataFile_testing, mode = 'rb') as file:
	testing_data = pickle.load(file)#reads from file stream object used for testing against neural network
	
X_training_data, y_training_data = training_data['features'], training_data['labels']#loading datasets to memory 
X_testing_data, y_testing_data = testing_data['features'], testing_data['labels']
#features is an array which contains the pixel data of the images
#labels contains information about the traffic sign such as the picture ID

#finished loading dataset

import cv2 #importing computer vision library to help grayscale all of the images

def grayScale(image_data):
	grayscaled = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
	return grayscaled
	
X_training_data = np.array([grayScale(image_data) for image_data in X_training_data]) #grayscales images in training data
X_testing_data = np.array([grayScale(image_data for image_data in X_testing_data]) #grayscales images in testing dataset

#end of preprocessing dataset


#beginning of training code for CNN model

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	
	print("Training the model")

