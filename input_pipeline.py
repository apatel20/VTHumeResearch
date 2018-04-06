import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import glob, csv, PIL
from tensorflow.contrib.data import Dataset, Iterator
NUM_CLASSES = 47
IMAGE_SIZE = 28
BATCH_SIZE = 100

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
# Parses a list of file names and converts them into Tensors
# @param file is a 1-D Tensor which is a list of files, and label is a 1-D Tensor representing the respective
# class number of each file
# @return the label tensor and the transformed feature tensor
def input_parser(file,label):
    #convert label to one-hot vector
    #one_hot = tf.one_hot(label,NUM_CLASSES)
    #Decodes the image to transform it into a tensor
    image = tf.read_file(file)
    decoded_image = tf.image.decode_png(image,channels=3)
    #Convert from uint8 to float
    decoded_image = tf.cast(decoded_image,tf.float32)
    grayscale_image = tf.image.rgb_to_grayscale(decoded_image)
    image_resize = tf.image.resize_image_with_crop_or_pad(grayscale_image,IMAGE_SIZE,IMAGE_SIZE)
    return label, image_resize
# Maps an input function so that it can be used in the Estimator
# @param features is a 3-D Tensor with dim [width, height, channels], labels is a 1-D Tensor, train
# is a boolean indicating whether training is true, and batch_size is the batch_size
# @ return two elemens from a dataset. The feature will be a dictionary and the label a number.
def input_function(features, labels, train,batch_size):
    # Creates a dataset from features and labels
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if train:
        # if train, repeat the dataset an infinte amount of times
        # and shuffle dataset
        num_repeat = -1
        # To avoid not iterating through the whole dataset, make the
        # buffer size the same length as the number of training samples
        buffer_size = len(features)
        dataset = dataset.shuffle(buffer_size=buffer_size)
    else:
        #if not train, go through dataset once
        num_repeat =1
    #maps the training data features from filenames to tensors
    dataset = dataset.map(input_parser)
    dataset = dataset.repeat(num_repeat)
    # Applies a batch to th` dataset and transforms each element into a 4-D Tensor with dim [batch, width, height, channels]
    dataset = dataset.batch(batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    label, feature = iterator.get_next()
    # converts feature into a dict and labels into y for easier understanding
    x = {'image': feature}
    y = label
    return x,y
# prepares the input function for the estimator in training mode
def training_function():
    return input_function(features=X_train,labels=y_train,train=True, batch_size=BATCH_SIZE)
# prepares the input function for the estimator in testing mode
def testing_function():
    return input_function(features=X_test,labels=y_test,train=False, batch_size=BATCH_SIZE)



#creates list with file paths
path_list = glob.glob('Data/annotations/*.png')
#creates list of classes
class_list = []
for path in path_list:
    fixedpath = path.split('_')
    class_list.append(fixedpath[1])
#Removes duplicates from class_list
class_list = f7(class_list)
#creates list of indexes
index_list = []
for path in path_list:
    fixedpath = path.split('_')
    index_list.append(class_list.index(fixedpath[1])+1)

#Splits features and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(path_list, index_list, test_size=0.2, random_state=42)
print(index_list)

with tf.Session() as sess:
    while True:
            try:
                x,y = input_function(features=path_list,labels=index_list,train=True, batch_size=BATCH_SIZE)
                elem = sess.run(x['image'])
                assert not np.any(np.isnan(elem))
                print("Still here")
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break