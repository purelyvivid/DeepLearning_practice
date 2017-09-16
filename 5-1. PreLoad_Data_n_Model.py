import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.metrics import accuracy

def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


x_train_, y_train_ = read_and_decode("train.cifar10", True)
x_test_, y_test_   = read_and_decode("test.cifar10", False)

batch_size = 128
x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                      batch_size=batch_size,
                                                      capacity=2000,
                                                      min_after_dequeue=1000,
                                                      num_threads=10) # set the number of threads here
    # for testing, uses batch instead of shuffle_batch
x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
                                            batch_size=batch_size,
                                            capacity=10000,
                                            num_threads=10)
   
def model_1(inputs):
    with slim.arg_scope([slim.conv2d],
                        kernel_size=3,
                        stride=1):
        
        v1 = slim.conv2d(inputs,64,scope='conv1')
        v1 = slim.batch_norm(v1)
        v2 = slim.max_pool2d(v1,[2, 2], scope='pool1')
        v3 = slim.conv2d(v2,128,scope='conv2')
        v3 = slim.batch_norm(v3)
        v4 = slim.max_pool2d(v3, [2, 2], scope='pool2')
        v5 = slim.conv2d(v4,256,scope='conv3')
        v5 = slim.batch_norm(v5)
        v6 = slim.max_pool2d(v5,[2, 2], scope='pool3')
        v7 = slim.flatten(v6, scope='flatten')
        v8 = slim.fully_connected(v7, 400, scope='fc1')
        v9 = slim.fully_connected(v8, 200, scope='fc2')
        v_list = [v1, v2, v3, v4, v5, v6, v7, v8, v9]
        outputs = slim.fully_connected(v9, 10, activation_fn=None, scope='fc3')
        return outputs, v_list
    
def model_2(inputs):
    with slim.arg_scope([slim.conv2d],
                        kernel_size=7,
                        stride=1):
        
        v1 = slim.conv2d(inputs,64,scope='conv1')
        v1 = slim.batch_norm(v1)
        v2 = slim.max_pool2d(v1,[2, 2], scope='pool1')
        v3 = slim.conv2d(v2,128,scope='conv2')
        v3 = slim.batch_norm(v3)
        v4 = slim.max_pool2d(v3, [2, 2], scope='pool2')
        v5 = slim.conv2d(v4,256,scope='conv3')
        v5 = slim.batch_norm(v5)
        v6 = slim.max_pool2d(v5,[2, 2], scope='pool3')
        v7 = slim.flatten(v6, scope='flatten')
        v8 = slim.fully_connected(v7, 400, scope='fc1')
        v9 = slim.fully_connected(v8, 200, scope='fc2')
        v_list = [v1, v2, v3, v4, v5, v6, v7, v8, v9]
        outputs = slim.fully_connected(v9, 10, activation_fn=None, scope='fc3')
        return outputs, v_list
    
def model_3(inputs):
    with slim.arg_scope([slim.conv2d],
                        kernel_size=3,
                        stride=2):
        
        v1 = slim.conv2d(inputs,64,scope='conv1')
        v1 = slim.batch_norm(v1)
        v2 = slim.max_pool2d(v1,[2, 2], scope='pool1')
        v3 = slim.conv2d(v2,128,scope='conv2')
        v3 = slim.batch_norm(v3)
        #v4 = slim.max_pool2d(v3, [2, 2], scope='pool2')
        v5 = slim.conv2d(v3,256,scope='conv3')
        v5 = slim.batch_norm(v5)
        v6 = slim.max_pool2d(v5,[2, 2], scope='pool3')
        v7 = slim.flatten(v6, scope='flatten')
        v8 = slim.fully_connected(v7, 400, scope='fc1')
        v9 = slim.fully_connected(v8, 200, scope='fc2')
        v_list = [v1, v2, v3, v5, v6, v7, v8, v9]
        outputs = slim.fully_connected(v9, 10, activation_fn=None, scope='fc3')
        return outputs, v_list

    
def model_4(inputs):
    with slim.arg_scope([slim.conv2d],
                        kernel_size=7,
                        stride=2):
        
        v1 = slim.conv2d(inputs,64,scope='conv1')
        v1 = slim.batch_norm(v1)
        v2 = slim.max_pool2d(v1,[2, 2], scope='pool1')
        v3 = slim.conv2d(v2,128,scope='conv2')
        v3 = slim.batch_norm(v3)
        #v4 = slim.max_pool2d(v3, [2, 2], scope='pool2')
        v5 = slim.conv2d(v3,256,scope='conv3')
        v5 = slim.batch_norm(v5)
        v6 = slim.max_pool2d(v5,[2, 2], scope='pool3')
        v7 = slim.flatten(v6, scope='flatten')
        v8 = slim.fully_connected(v7, 400, scope='fc1')
        v9 = slim.fully_connected(v8, 200, scope='fc2')
        v_list = [v1, v2, v3, v5, v6, v7, v8, v9]
        outputs = slim.fully_connected(v9, 10, activation_fn=None, scope='fc3')
        return outputs, v_list
  