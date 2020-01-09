import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

def get_vgg16_weights(input_shape):
    """

    """
    #Get the VGG16 Model from keras
    vgg16 = keras.applications.vgg16.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    #Initialize the weights list
    weights = []
    #For every layer
    for layer in vgg16.layers:
        #If its a convolution layer
        if 'Conv2D' in str(layer):
            #Get the weights and biases!
            weights.append(layer.get_weights())
    #Return the list of weights and biases from VGG16 ImageNet
    return weights   

def conv_layer(input, filters):
    """
    A convolution block which contains a convolution.
    Then a Batch Normalization.
    Then a ReLU activation.

    :param input: A Tensor, which is the output of the previous layer
    :param filters: The number of channels it should output
    :param training: Defaults as True, mainly for Batch, if validation
    """
    #Do the Convolution
    conv = keras.layers.Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same')(input)
    #Do a Batch Normalization
    norm = keras.layers.BatchNormalization()(conv)
    #Then a ReLU activation
    relu = keras.activations.relu(norm)
    #Return the result of the block
    return relu

def max_pool(inputs):
    """

    """
    value, index = tf.nn.max_pool_with_argmax(tf.to_double(inputs), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.to_float(value), index, inputs.get_shape().as_list()

def bayes_segnet(input_size=(1024,1024,3)):
    '''
    In the original github repo here, https://github.com/toimcio/SegNet-tensorflow/blob/master/SegNet.py
    Here I will mix keras and tf.nn layers to produce the same model. Using parts of the github repo above
    to fill in with tf.nn when necessary. This makes the other convolutions a little easier to read I reckon.

    :param input_size: By default (1024, 1024,3)

    :returns model: A Bayesian Segmentation Network as per the paper https://arxiv.org/abs/1511.02680
    '''
    ###-----INPUTS-----###

    #Setup the inputs layer
    inputs = keras.Input(shape=input_size)
    #Do the Local Response Normalization
    lrn = tf.nn.local_response_normalization(inputs, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)

    ###-----ENCODER-----###
    #This is essentially VGG16 with some minor additions

    #First Block of Encoder, 2 Convolutions and then a Max Pooling Layer (which takes the indices)
    conv1 = conv_layer(lrn, 64)
    conv2 = conv_layer(conv1, 64)
    pool1, pool1_indices, pool1_shape = max_pool(conv2)

    #Second Block of Encoder, same as the first, but 128 channels
    conv3 = conv_layer(pool1, 128)
    conv4 = conv_layer(conv3, 128)
    pool2, pool2_indices, pool2_shape = max_pool(conv4)

    # Third Block of Encoder, 3 convolutions with 256 channels and a max pooling
    conv5 = conv_layer(pool2, 256)
    conv6 = conv_layer(conv5, 256)
    conv7 = conv_layer(conv6, 256)
    pool3, pool3_indices, pool3_shape = max_pool(conv7)

    #Fourth Block of Encoder, A Dropout Layer, 3 Convolutions with 512 channels, then a pooling layer
    drop1 = keras.layers.Dropout(0.5)(pool3)
    conv8 = conv_layer(drop1, 512)
    conv9 = conv_layer(conv8, 512)
    conv10 = conv_layer(conv9, 512)
    pool4, pool4_indices, pool4_shape = max_pool(conv10)

    #Fifth Block of Encoder, Same as Foruth
    drop2 = keras.layers.Dropout(0.5)(pool4)
    conv11 = conv_layer(drop2, 512)
    conv12 = conv_layer(conv11, 512)
    conv13 = conv_layer(conv12, 512)
    pool5, pool5_indices, pool5_shape = max_pool(conv13)

    ###-----DECODER-----###

    #First Block of Decoder, A Dropout, Upsampling, then 3 Convolutions











