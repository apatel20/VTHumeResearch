#import libraries and dependencies
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import random
from input_pipeline.py import label, feature

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

#exploring the data 
number_training_samples, number_testing_samples = X_training_data.shape[0], X_testing_data.shape[0] ##extracting info from the vector

#this function plots the data on a bar chart by counting the number of images for each class
def plotting_histogram(data2plot, label):
	images = data2plot.tolist()
	count_per_class = [images.count(c) for c in range(NUM_CLASSES)]
	plt.bar(range(NUM_CLASSES), count_per_class)
	plt.xlabel(label)
	plt.show()
#end function

plotting_histogram(y_training_data, label="Training Set Data")
plotting_historgram(y_testing_data, label="Testing Set Data")

#finished loading dataset

import cv2 #importing computer vision library to help grayscale all of the images

def grayScale(image_data):
	grayscaled = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY) #color conversion from RGB to grayscale
	return grayscaled
	
X_training_data = np.array([grayScale(image_data) for image_data in X_training_data]) #grayscales images in training data
X_testing_data = np.array([grayScale(image_data for image_data in X_testing_data]) #grayscales images in testing dataset

#end of preprocessing dataset

num_training = len(training_data)

BATCH_SIZE = 150
EPOCHS = num_training / 150

#evaluating neural network model for functions, predictions, and accuracy

#setting my placeholders for training/testing the network
#will be official once Cristian finishes the network

#logits = network() or the function where the network is defined

#LR = tf.placeholder(tf.float32)
saver = tf.train.Saver()#saves variables and checkpoint filenames

#inference_operation
#cross_entropy

#loss_operation = tf.reduce_mean(cross_entropy)
#training_operation = optimizer.minimize(loss_operation)
#optimizer = tf.train.AdamOptimizer(learning_rate = LR)

#correct_prediction = tf.equal(tf.argmax()
#accuracy_operation = tf.reduce_mean(tf.cast())

def evaluation(X_data, y_data):
	number_examples = len(X_data)
	total_accuracy = 0
	
	sess = tf.get_default_session()#returns default session for current
	
	for start in range(0, number_examples, BATCH_SIZE)
		batch_x, batch_y = X_data[start:start+BATCH_SIZE], y_data[start:start+BATCH_SIZE] 
		accuracy = sess.run(accuracy_operation, feed_dict = {x: batch_x, y: batch_y}, keep_prob: 1.0)
		total_accuracy = total_accuracy + (accuracy * len(batch_x))
	
	return total_accuracy / number_examples
	

#beginning of training code for CNN model

from sklearn.utils import shuffle#shuffling matrices consistently for training

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	number_examples = len(x_training)
	
	print("Training the model")
	
	for epoch in range(EPOCHS):
		X_preprocessed, y_preprocessed = shuffle(X_preprocessed, y_preprocessed) #randomizing the data for every iteration in the training data
		for start in range(0, number_examples, BATCH_SIZE):
			end = start + BATCH_SIZE
			batch_x, batch_y = get_batch(X_preprocessed, y_preprocessed, start, BATCH_SIZE)
			
		print("Accuracy during Validation Process")
		#see how accurate the model is when training
		evaluation_training_accuracy = evaluate(X_training_data, y_training_data)
		print("EPOCH {} ..." .format(epoch+1))
			
	saver.save(sess, './cnnModel')