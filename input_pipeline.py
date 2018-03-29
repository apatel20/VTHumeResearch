import tensorflow as tf
import numpy as np
import glob, csv, PIL
from tensorflow.contrib.data import Dataset, Iterator
NUM_CLASSES = 47
def input_parser(file,label):
    #convert label to one-hot vector
    one_hot = tf.one_hot(label,NUM_CLASSES)
    #Decodes the image to transform it into a tensor
    image = tf.read_file(file)
    decoded_image = tf.image.decode_image(image,channels=3)
    grayscale_image = tf.image.rgb_to_grayscale(decoded_image)
    image_resize = tf.image.resize_image_with_crop_or_pad(grayscale_image,28,28)
    return one_hot, image_resize
#creates list of classes
csvfile = open('Data/allAnnotations.csv')
reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
class_list =[]
for row in reader:
    fixedrow = row[0].split(';')
    if fixedrow[1]!="Annotation":
        class_list.append(fixedrow[1])
class_list = list(set(class_list))
csvfile.close()
#Creates list with indexes
csvfile = open('Data/allAnnotations.csv')
reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
index_list = []
for row in reader:
    fixedrow = row[0].split(';')
    if fixedrow[1]!="Annotation":
        index_list.append(class_list.index(fixedrow[1]))
tensor_index_list = tf.convert_to_tensor(index_list)
#creates tensor with file paths
path_tensor = tf.constant(tf.gfile.Glob('Data/annotations/*.png'))
#Creates data sets
training_data = tf.data.Dataset.from_tensor_slices((path_tensor,tensor_index_list))
training_data = training_data.map(input_parser)
iterator = Iterator.from_structure(training_data.output_types,training_data.output_shapes)
training_init_op = iterator.make_initializer(training_data)
training_data = training_data.batch(5)
label, feature = iterator.get_next()

with tf.Session() as sess:
    sess.run(training_init_op)
    print("VERSION", tf.Session(config=tf.ConfigProto(log_device_placement=True)))
    while True:
            try:
                elem = sess.run(feature)
                print(sess.run(tf.shape(elem)))
                print("batch")
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break

