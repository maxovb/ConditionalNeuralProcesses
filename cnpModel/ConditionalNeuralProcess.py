import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model
from GaussianProcesses.GaussianProcessSampler import GaussianProcess
import tensorflow_probability as tfp
from cnpModel.Attention import SelfAttentionEncoder, AttentionDecoder
from cnpModel.Convolutional import ConvolutionalEncoder, ConvolutionalDecoder

# will write class such that it takes in a single [xc, yc] and [xt]
# to train train on each batch
# then get new batch and retrain
class ConditionalNeuralProcess(Model):
    def __init__(self, encoder_layer_widths, decoder_layer_widths, attention = False, attention_params = {}, convolutional = False, convolutional_params = {}):
        super(ConditionalNeuralProcess, self).__init__()
        if attention and convolutional:
            raise Error('Cannot handle both attention and convolutional at the moment')

        # define the right encoder and decoder
        if not(attention) and not(convolutional):
            self._encoder = Encoder(encoder_layer_widths)
            self._decoder = Decoder(decoder_layer_widths)
        elif attention:
            self._encoder = SelfAttentionEncoder(encoder_layer_widths, attention_params['num_heads'], attention_params['num_self_attention_blocks'])
            self._decoder = AttentionDecoder(decoder_layer_widths, attention_params['embedding_layer_width'], attention_params['num_heads'])
        elif convolutional:
            self._encoder = ConvolutionalEncoder(convolutional_params["number_filters"], convolutional_params["kernel_size_encoder"])
            self._decoder = ConvolutionalDecoder(convolutional_params["number_filters"], convolutional_params["kernel_size_decoder"], convolutional_params["number_residual_blocks"], convolutional_params["convolutions_per_block"],  convolutional_params["output_channels"])
        
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.convolutional = convolutional

    def call(self, inputs):

        # encoder
        encoder_output = self._encoder(inputs)

        #decoder
        if self.convolutional:
            means, stds = self._decoder(encoder_output)
        else:
            means, stds = self._decoder(encoder_output, inputs)

        return means, stds

    def loss_func(self, means, stds, targets):
        # want distribution of all target points
        dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=stds)
        log_prob = dist.log_prob(targets)
        avg_batch_loss = tf.reduce_mean(log_prob, axis=1)
        return -tf.reduce_mean(avg_batch_loss)

    def train_step(self, inputs, targets):

        with tf.GradientTape() as tape:
            means, stds = self(inputs)
            #tf.print('means = ', means[0,0,0,0])
            #tf.print('stds = ', stds[0,0,0,0])
            #print('target',targets[0,0,0,0])
            loss = self.loss_func(means, stds, targets)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss
    
    def model(self):
        M = keras.Input(shape=(32,32,1))
        img = keras.Input(shape=(32,32,3))
        return keras.Model(inputs=[M,img], outputs=self.call([M,img]))


class Encoder(Layer):
    """
    The Encoder which is to be shared across all context points.
    Instantiate with list of target number of nodes per layer.
    """
    def __init__(self, encoder_layer_widths):
        super(Encoder, self).__init__()
        # add the hidden layers
        self.h = []
        for layer_width in encoder_layer_widths[:-1]:
            self.h.append(Dense(layer_width, activation='relu'))
        # no activation for the final layer
        self.h.append(Dense(encoder_layer_widths[-1], activation=None))

    def h_func(self, x):
        for layer in self.h:
            x = layer(x)

        return x

    def call(self, inputs):
        """
        Inputs is tuple (x_context, y_context, x_data), each shape [batch_size , num_points, dimension]
        Need to reshape inputs to be pairs of [x_context, y_context],
        Pass through NN,
        Compute a representation by aggregating the outputs.
        """

        # process data for encoder
        x_context, y_context = inputs[0], inputs[1]

        # # grab shapes
        # batch_size, num_context_points = x_context.shape
        #
        # # reshape to [batch_size, num_points, 1]
        # x_context = tf.reshape(x_context, (batch_size, num_context_points, 1))
        # y_context = tf.reshape(y_context, (batch_size, num_context_points, 1))

        # concatenate to form inputs [x_context, y_context], overall shape [batch_size, num_context, dim_x + dim_y]
        encoder_input = tf.concat([x_context, y_context], axis=-1)

        encoder_output = self.h_func(encoder_input)

        # now compute representation vector, average across context points, so axis 1
        representation = tf.reduce_mean(encoder_output, axis=1)

        return representation


class Decoder(keras.layers.Layer):
    """
    The Decoder to be shared amongst targets.
    Instantiate with list of target number of nodes per layer.
    For 1D regression need final layer to have 2 units
    """
    def __init__(self, decoder_layer_widths):
        super(Decoder, self).__init__()
        # add the hidden layers
        self.g = []
        for layer_width in decoder_layer_widths[:-1]:
            self.g.append(Dense(layer_width, activation='relu'))
        # no activation for the final layer
        self.g.append(Dense(decoder_layer_widths[-1], activation=None))

    def g_func(self, x):

        for layer in self.g:
            x = layer(x)

        return x

    def call(self, representation, inputs):
        """
        Takes in computed representation vector from encoder and x data to decode targets
        """
        # grab x_data
        x_data = inputs[-1]

        # need to concatenate representation to data, so each data set looks like [x_T, representation]

        # # need to reshape x_data
        # # grab shapes
        # batch_size, num_context_points = x_data.shape
        #
        # # reshape to [batch_size * num_points * 1]
        # x_data = tf.reshape(x_data, (batch_size, num_context_points, 1))

        # reshape representation vector and repeat it
        representation = tf.repeat(tf.expand_dims(representation, axis=1), x_data.shape[1], axis=1)
        # concatenate representation to inputs
        decoder_input = tf.keras.layers.concatenate([x_data, representation], axis=-1)

        decoder_output = self.g_func(decoder_input)

        # split into 2 tensors, one with means and one with stds
        # floor the variance to avoid pathological solutions
        means, log_stds = tf.split(decoder_output, 2, axis=-1)
        stds = 0.01 + 0.99 * tf.nn.softplus(log_stds)

        return means, stds


if __name__ == "__main__":
    # gps
    gp_test = GaussianProcess(100, 10, testing=False)
    data = gp_test.generate_curves()
    # model output sizes
    enc = Encoder(128)
    reps = enc((data.Inputs[0], data.Inputs[1]))
    masked_x = np.zeros(shape=data.Inputs[0].shape)
    masked_x[1:] = data.Inputs[0][1:]
    masked_y = np.zeros(shape=data.Inputs[1].shape)
    masked_y[1:] = data.Inputs[1][1:]
    masked_data = masked_x, masked_y
    reps_masked = enc((masked_x, masked_y))
    assert np.all(reps.numpy()[2, :] == reps_masked.numpy()[2, :])

    dec = Decoder(128)
    means, stds = dec(reps, data.Inputs)

    model = ConditionalNeuralProcess(layer_width=128)
    model(inputs=data.Inputs)
    model.summary()
