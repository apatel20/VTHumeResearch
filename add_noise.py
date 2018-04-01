import numpy as np
import cv2

def add_noise(X_image)
	#create copy as to not modify the actual image
	X_image_copy = X_image.copy()
	height,width,channel = X_image_copy.shape #gather dimensions and channel number for shape
	mean = 0
	stdev = 0.4
	sigma = stdev**0.5
	generated_noise = np.random.normal(mean, sigma, (height,width,channel))
	#gaussian noise is a statistical method used to generate noise (distribution curve)
	gaussian_noise = generated_noise.reshape(height,width,channel)
	image_with_noise = X_image_copy + gaussian_noise
	
return image_with_noise