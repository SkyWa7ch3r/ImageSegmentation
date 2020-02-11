import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import UpSampling2D, Conv2D, SeparableConv2D, concatenate, Dropout, MaxPooling2D, BatchNormalization, ReLU


def conv_layer(inputs, filters, kernel_size, separable=False, padding='same', kernel_initializer='he_normal'):
    if separable:
        conv = SeparableConv2D(filters, kernel_size, padding=padding,
                               kernel_initializer=kernel_initializer)(inputs)
    else:
        conv = Conv2D(filters, kernel_size, padding=padding,
                      kernel_initializer=kernel_initializer)(inputs)
    batch = BatchNormalization()(conv)
    relu = ReLU()(batch)
    return relu


def model(num_classes=20, input_size=(1024, 2048, 3)):
    """
    This model came about due to my need to have training be fast, within one day
    at high resolution (2048x1024 - cityscape images). So a few things inspired my decisions
    below:
    1. Batch Normalization has improved countless models in the past. Its a given for a model in 2019
    so let's try it here.
    2. Separable Convolution has been shown to be very powerful, Using a fraction of the parameters
    while keeping similar or the same accuracy. So I will use this for the convolutions in the encoder
    and decoder. All decoder convolutions will be replaced by separable convolutions. Every convolution
    before a pooling layer will be replaced.
    The middle bottleneck will also consist of only Separable Convolution.
    """
    ### Beginning of encoder ###
    inputs = keras.Input(input_size)
    # Encoder/Scaling Down
    conv1 = conv_layer(inputs, 64, 3)
    conv2 = conv_layer(conv1, 64, 3, separable=True)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_layer(pool1, 128, 3)
    conv4 = conv_layer(conv3, 128, 3, separable=True)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_layer(pool2, 256, 3)
    conv6 = conv_layer(conv5, 256, 3, separable=True)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
    conv7 = conv_layer(pool3, 512, 3)
    conv8 = conv_layer(conv7, 512, 3, separable=True)
    drop1 = Dropout(0.5)(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop1)

    # The middle
    conv9 = conv_layer(pool4, 1024, 3, separable=True)
    conv10 = conv_layer(conv9, 1024, 3, separable=True)
    drop2 = Dropout(0.5)(conv10)

    # Decoding/Scaling Up
    up1 = conv_layer(UpSampling2D(size=(2, 2))(drop2), 512, 2, separable=True)
    merge1 = concatenate([drop1, up1], axis=3)
    conv11 = conv_layer(merge1, 512, 3, separable=True)
    conv12 = conv_layer(conv11, 512, 3, separable=True)

    up2 = conv_layer(UpSampling2D(size=(2, 2))(conv12), 256, 2, separable=True)
    merge2 = concatenate([conv6, up2], axis=3)
    conv13 = conv_layer(merge2, 256, 3, separable=True)
    conv14 = conv_layer(conv13, 256, 3, separable=True)

    up3 = conv_layer(UpSampling2D(size=(2, 2))(conv14),
                     128, 2, separable=True, padding='same')
    merge3 = concatenate([conv4, up3], axis=3)
    conv15 = conv_layer(merge3, 128, 3, separable=True)
    conv16 = conv_layer(conv15, 128, 3, separable=True)

    up4 = conv_layer(UpSampling2D(size=(2, 2))(conv16), 64, 2, separable=True)
    merge4 = concatenate([conv2, up4], axis=3)
    conv17 = conv_layer(merge4, 64, 3, separable=True)
    conv18 = conv_layer(conv17, 64, 3, separable=True)
    conv19 = conv_layer(conv18, 2, 3)
    conv20 = Conv2D(num_classes, kernel_size=(1, 1), strides=(
        1, 1), activation='softmax', dtype=tf.float32)(conv19)

    model = keras.Model(inputs=inputs, outputs=conv20)

    return model
