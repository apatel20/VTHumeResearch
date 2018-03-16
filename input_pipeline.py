import tensorflow as tf
import numpy as np
import glob, csv, PIL
from tensorflow.contrib.data import Dataset, Iterator
NUM_CLASSES = 47

def input_parser(file,label):
    #convert label to one-hot vector
    one_hot = tf.one_hot(label,NUM_CLASSES)

    image = tf.read_file(file)
    decoded_image = tf.image.decode_image(image,channels=3)
    return one_hot, decoded_image
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
next_element = iterator.get_next()
training_init_op = iterator.make_initializer(training_data)
with tf.Session() as sess:
    sess.run(training_init_op)
    while True:
            try:
                elem = sess.run(next_element)
                print(elem)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break

