import tensorflow as tf
from tensorflow import keras

def entry_block(input_size):
    #Input Layer
    inputs = keras.Input(input_shape=input_size)

    #First convolution
    conv1 = keras.layers.Conv2D(32, 3, strides=2, padding='valid')(inputs)
    norm1 = keras.layers.BatchNormalization()(conv1)
    relu1 = keras.activations.relu()(norm1)
    #Second Convolution
    conv2 = keras.layers.Conv2D(64, 3, strides=1, padding='valid')(relu1)
    norm2 = keras.layers.BatchNormalization()(conv2)
    relu2 = keras.activations.relu()(norm2)

    #First Separable Convolution
    sep_conv1 = keras.layers.SeparableConv2D(128, 3, strides=1, padding='same')(relu2)
    norm3 = keras.layers.BatchNormalization()(sep_conv1)
    relu3 = keras.activations.relu()(norm3)
    #Second Separable Convolution
    sep_conv2 = keras.layers.SeparableConv2D(128, 3, strides=1, padding='same')(relu3)
    norm4 = keras.layers.BatchNormalization()(sep_conv2)
    relu4 = keras.activations.relu()(norm4)
    #Instead of a max pooling layer, we do a another Third SeparableConv, but with stride 2
    sep_conv3 = keras.layers.SeparableConv2D(128, 3, strides=2, padding='same')(relu4)
    norm5 = keras.layers.BatchNormalization()(sep_conv3)

    #Third Convolution (Done in parallel to SeparableConv2D Layers above)
    conv3 = keras.layers.Conv2D(128, 1, strides=2, padding='same')(relu2)
    norm6 = keras.layers.BatchNormalization()(conv3)

    #Do a skip connection
    add1 = keras.layers.Add(norm5, norm6)
    relu5 = keras.activations.relu()(add1)

    #Fourth Separable Convolution
    sep_conv4 = keras.layers.SeparableConv2D(256, 3, strides=1, padding='same')(relu2)
    norm7 = keras.layers.BatchNormalization()(sep_conv1)
    relu5 = keras.activations.relu()(norm3)
    #Fifth Separable Convolution
    sep_conv2 = keras.layers.SeparableConv2D(256, 3, strides=1, padding='same')(relu3)
    norm4 = keras.layers.BatchNormalization()(sep_conv2)
    relu4 = keras.activations.relu()(norm4)
    #Sixth Separable Convolution
    sep_conv3 = keras.layers.SeparableConv2D(256, 3, strides=2, padding='same')(relu4)
    norm5 = keras.layers.BatchNormalization()(sep_conv3)

    #Fourth Convolution (Done in parallel to SeparableConv2D Layers above)
    conv3 = keras.layers.Conv2D(256, 1, strides=2, padding='same')(relu2)
    norm6 = keras.layers.BatchNormalization()(conv3)

    #Do a skip connection
    add1 = keras.layers.Add(norm5, norm6)



def model(input_size=(1024,1024,3)):


    
