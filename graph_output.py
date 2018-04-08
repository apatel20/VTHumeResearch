#see the output of image and its prediction from the model when going into testing
#uses the data from the estimator api

# Predict single images
number_images = 3
# Get images from test set
test_images = features[:number_images]
# Prepare the input data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, shuffle=False)
	
# Use the model to predict the images class
preds = list(model.predict(input_fn))

# Display each image with its prediction
for i in range(number_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("What the Model Predicted: ", preds[i])



