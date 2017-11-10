import tensorflow as tf
from deepsphinx.utils import FLAGS
import collections

class LSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, forget_bias=1.0, initializer=None,
                 noise_std=None, tile_size=1, reuse=None):
        super(LSTMCell, self).__init__(_reuse=reuse)
        self.initializer = initializer
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.noise_std = noise_std
        self.random_noise = tf.random_normal([tile_size * FLAGS.batch_size, 4 * num_units], stddev = self.noise_std)

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def call(self, inputs, state):
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, initializer=self.initializer):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = inputs.get_shape().as_list()[1]
            W_both = tf.get_variable('kernel',
                [x_size + self.num_units, 4 * self.num_units])
            bias = tf.get_variable('bias', [4 * self.num_units])

            batch_size = inputs.get_shape().as_list()[0]
            #W_both = W_both + tf.random_normal([batch_size] + W_both.get_shape().as_list(), stddev = self.noise_std)
            #bias = bias + tf.random_normal([batch_size] + bias.get_shape().as_list(), stddev = self.noise_std)

            #hidden = tf.matmul(tf.expand_dims(tf.concat([inputs, h], 1), 1), W_both) + tf.expand_dims(bias, 1)
            #hidden = tf.squeeze(hidden)

            inp = tf.concat([inputs, h], 1)
            print(inp)
            std_mult = tf.sqrt(tf.sqrt(tf.reduce_sum(inp * inp, 1)))
            print(std_mult)
            
            hidden = tf.matmul(inp, W_both) + bias
            hidden = hidden + self.random_noise * tf.expand_dims(std_mult, 1)

            i, j, f, o = tf.split(hidden, 4, 1)

            new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)
