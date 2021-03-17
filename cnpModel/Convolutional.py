#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model
from keras.constraints import NonNeg



class ConvolutionalEncoder(Layer):
    """
    The Encoder which is to be shared across all context points.
    Instantiate with list of target number of nodes per layer.
    """
    def __init__(self, number_filters, kernel_size):
        super(ConvolutionalEncoder, self).__init__()
        self.conv1 =  tf.keras.layers.SeparableConv2D(number_filters, kernel_size, activation = None, use_bias = False, padding = 'same', depthwise_constraint= NonNeg(), pointwise_constraint=NonNeg(), bias_constraint=NonNeg())
        #self.conv2 =  tf.keras.layers.SeparableConv2D(number_filters, kernel_size, activation = None, use_bias = False, padding = 'same', depthwise_constraint= NonNeg(), pointwise_constraint=NonNeg(), bias_constraint=NonNeg())


    def conv_func(self, M, x):
        # M : mask (1 binary channel)
        # x : image (1 grey or 3 RGB channels)
        M = tf.broadcast_to(M,x.shape)
        density = self.conv1(M)
        signal = self.conv1(x)
        signal = tf.math.divide(signal,tf.math.maximum(density,1e-3))
        return tf.concat([signal,density], axis = -1) # not sure if we should concatenate or not

    def call(self, inputs):
        # process data for encoder
        mask = inputs[0]
        image_context = inputs[1]

        encoder_output = self.conv_func(mask, image_context)
        return encoder_output


class ConvolutionalDecoder(keras.layers.Layer):
    """
    The Decoder to be shared amongst targets.
    Instantiate with list of target number of nodes per layer.
    For 1D regression need final layer to have 2 units
    """
    def __init__(self, number_filters, kernel_size, number_residual_blocks = 4, convolutions_per_block = 1,  output_channels = 2):
        super(ConvolutionalDecoder, self).__init__()
        # initialize the layers for each residual blocks
        self.c = []
        for i in range(number_residual_blocks):
            block_layers = []
            for j in range(convolutions_per_block):
                block_layers.append(tf.keras.layers.SeparableConv2D(number_filters, kernel_size, activation = 'relu', padding = 'same'))
            self.c.append(block_layers)
        self.add = tf.keras.layers.Add()

        self.d1 = tf.keras.layers.Conv2D(number_filters, 1, activation = None)
        self.d2 = tf.keras.layers.Conv2D(2 * output_channels, 1, activation = None)



    def rho_func(self, x):
        x = self.d1(x)
        for block in self.c:
            x_input = x
            for layer in block:
                x = layer(x)
            x = self.add([x,x_input])
        # output layer
        x = self.d2(x)
        return x

    def call(self, x):
        """
        Takes the output of the encoder and outputs the mean and standard deviation for all pixels at each channels
        """

        decoder_output = self.rho_func(x)

        # split into 2 tensors, one with means and one with stds
        # floor the variance to avoid pathological solutions
        means, log_stds = tf.split(decoder_output, 2, axis=-1)
        stds = 0.01 + 0.99 * tf.nn.softplus(log_stds)

        return means, stds