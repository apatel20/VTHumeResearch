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

with open(dataFile_training, mode = 'rb') as f:
	training_data = pickle.load(f)

with open(dataFile_testing, mode - 'rb') as f:
	testing_data = pickle.load(f)
	
X_training_data, y_training_data = training_data['features'], training_data['labels']
X_testing_data, t_testing_data = testing_data['features'], testing_data['labels']