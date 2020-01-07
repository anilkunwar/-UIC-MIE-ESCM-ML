import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers,regularizers,initializers
from keras.layers.advanced_activations import PReLU



weight_decay=True

def activation_function(x,name=None):
    act_fnct = PReLU(shared_axes=[1,2], name=name+"/PReLU", alpha_initializer=initializers.Constant(0.01))
    return act_fnct(x)

def regularization():
    if weight_decay:
        return regularizers.l2(weight_decay)
    else:
        return None

def convolutional_layer(x, channels, kernel_size=3, name='convolutional_layer'):

    convolutional_layer = layers.Conv2D(channels, kernel_size, padding='same',
                             kernel_regularizer=regularization(),
                             kernel_initializer='RandomNormal',
                             bias_initializer=initializers.Constant(0.1),
                             name=name)
    x = activation_function(convolutional_layer(x),name=name)
    x = layers.BatchNormalization(name=name+'_batch_normalization')(x)
    return x

def skip_connection(x, y,name=None):
    return layers.add([x, y])

def residual_block(x, channels, name='residual_block'):

    y = convolutional_layer(x, channels, name=name+'/convolution_in_residual_1')
    y = convolutional_layer(y, channels, name=name+'/convolution_in_residual_2')
    y = convolutional_layer(y, channels, name=name+'/convolution_in_residual_3')
    connection_x_y = skip_connection(x, y, name=name+'/element_wise_addition')
    return connection_x_y

def convolution_block(x, channels, name='convolution_block'):

    x = convolutional_layer(x, channels, name=name+"/convolution_1")
    x = residual_block(x, channels, name=name+"/residual_block")
    x =convolutional_layer(x,channels, name=name+"/convolution_2")
    return(x)


def max_pooling_layer(x, name='max_pooling'):
    return layers.MaxPooling2D(pool_size=2, padding='same', name=name)(x)

def bilinear_upsampling(x):

    original_shape = K.int_shape(x)
    factor=2
    new_shape = tf.shape(x)[1:3]
    new_shape *= tf.constant(np.array([factor, factor]).astype('int32'))
    x = tf.compat.v1.image.resize_bilinear(x, new_shape)
    x.set_shape((None, original_shape[1] * factor if original_shape[1] is not None else None,
                 original_shape[2] * factor if original_shape[2] is not None else None, None))
    return x

def BilinearUpSampling2D(**kwargs):
    return layers.Lambda(bilinear_upsampling, **kwargs)


def upsampling_layer(x, channels, name='upsampling_layer'):

    x = BilinearUpSampling2D(name=name+'/upsampling')(x)
    x = convolutional_layer(x, channels, kernel_size=1, name=name+'/upsampling_convolutional')
    return (x)

def score_layer(x, channels, kernel_size=1, name="score_layer"):

    final_convolution = layers.Conv2D(channels, kernel_size,
                             activation='relu',
                             padding='same',
                             kernel_regularizer=regularization(),
                             kernel_initializer='RandomNormal',
                             bias_initializer=initializers.Constant(0.1),
                             name=name)
    return final_convolution(x)
