import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model
from Utils.keras_multi_head.multi_head_attention import MultiHeadAttention

class SelfAttentionEncoder(keras.layers.Layer):
    # Same implementation as in appendix B in the paper "Convolutional conditional neural processes"

    def __init__(self, encoder_layer_widths, num_heads, num_self_attention_blocks):
        super(SelfAttentionEncoder, self).__init__()
        self.num_self_attention_blocks = num_self_attention_blocks
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()
        self.d = [] # dense layers
        for layer_width in encoder_layer_widths:
            self.d.append(Dense(layer_width, activation='relu'))
        self.a = [] # self-attention blocks
        for _ in range(self.num_self_attention_blocks):
            self.a.append(MultiHeadAttention(head_num=num_heads)) 
    
    def sa_func(self,x):
        # self-attention

        # first two dense layers
        for dense_layer in self.d:
            x = dense_layer(x)

        # first self-attention block
        for attention_layer in self.a:
            x = self.norm(x)
            x_att = attention_layer([x,x,x])
            x = self.add([x_att,x])
            x = self.norm(x)

        return x
    
    def call(self, inputs):
        """
        Inputs is tuple (x_context, y_context, x_data), each shape [batch_size , num_points, dimension]
        Need to reshape inputs to be pairs of [x_context, y_context],
        First pass through NN,
        then compute the self attention
        """

        # process data for encoder
        x_context, y_context = inputs[0], inputs[1]

        # concatenate to form inputs [x_context, y_context], overall shape [batch_size, num_context, dim_x + dim_y]
        encoder_input = tf.concat([x_context, y_context], axis=-1)

        encoder_output = self.sa_func(encoder_input)

        return encoder_output
    
class AttentionDecoder(keras.layers.Layer):

    def __init__(self, decoder_layer_widths, embedding_layer_width, num_heads):
        super(AttentionDecoder, self).__init__()

        self.d1 = Dense(embedding_layer_width, activation='relu') # dense layer for the x inputs
        self.a1 = MultiHeadAttention(head_num=num_heads) 
        self.d = []
        for layer_width in decoder_layer_widths[:-1]:
            self.d.append(Dense(layer_width, activation='relu')) # dense layers for the decoder
        self.d.append(Dense(decoder_layer_widths[-1], activation=None))

    
    def ca_func(self, x_context, encoder_output, x_target):
        # cross-attention

        # pass context and target x inputs through the same dense layer
        x_context = self.d1(x_context)
        x_target = self.d1(x_target)

        # attention layer
        keys = x_context
        values = encoder_output
        queries = x_target

        # apply cross-attention
        x = self.a1([queries,keys, values])

        # final dense layers for the decoder
        for layer in self.d:
            x = layer(x)

        return x

    def call(self, encoder_output, inputs):
        """
        Inputs is tuple (x_context, y_context, x_data), each shape [batch_size , num_points, dimension]
        Need to reshape inputs to be pairs of [x_context, y_context],
        First pass the x inputs through a single layer dense NN,
        then compute the cross attention 
        then get the decoder output
        """
        x_context = inputs[0]
        x_target = inputs[-1]

        # apply the cross-attention and the decoder output
        decoder_output = self.ca_func(x_context, encoder_output, x_target)

        # split into 2 tensors, one with means and one with stds
        # floor the variance to avoid pathological solutions
        means, log_stds = tf.split(decoder_output, 2, axis=-1)
        stds = 0.01 + 0.99 * tf.nn.softplus(log_stds)

        return means, stds


