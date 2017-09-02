# Modified BahdanauAttention to implement cutoff
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from deepsphinx.utils import FLAGS

# pylint: disable=too-few-public-methods
class BahdanauAttentionCutoff(tf.contrib.seq2seq.BahdanauAttention.__base__):
    '''Implements Bhadanau-style (additive) attention.
    This attention has two forms.  The first is Bhandanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    'Neural Machine Translation by Jointly Learning to Align and Translate.'
    ICLR 2015. https://arxiv.org/abs/1409.0473
    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    'Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks.'
    https://arxiv.org/abs/1602.07868
    To enable the second form, construct the object with parameter
    `normalize=True`.
    '''

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 score_mask_value=float('-inf'),
                 name='BahdanauAttention'):
        '''Construct the Attention mechanism.
        Args:
          num_units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          name: Name to use when creating ops.
        '''
        def probability_fn_cutoff(scores, previous_alignments):
            '''Only allow characters near previous alignments means and make all
            zero'''
            ran = tf.range(tf.to_float(
                tf.shape(previous_alignments)[1]), dtype=tf.float32)
            mean = (tf.reduce_sum(ran * previous_alignments, axis=1) /
                    tf.reduce_sum(previous_alignments, axis=1))
            mask = tf.logical_and(
                ran > mean - FLAGS.cutoff_range,
                ran < mean + FLAGS.cutoff_range)
            return tf.nn.softmax(tf.where(mask, scores, tf.ones_like(scores) * -1000))
        probability_fn = tf.nn.softmax

        def probability_fn_cutoff(score, _):
            return probability_fn(score)

        super(BahdanauAttentionCutoff, self).__init__(
            query_layer=Dense(
                num_units, name='query_layer', use_bias=False),
            memory_layer=Dense(
                num_units, name='memory_layer', use_bias=False),
            memory=memory,
            probability_fn=probability_fn_cutoff,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        dtype = tf.float32
        self.v = tf.get_variable(
            'attention_v', [self._num_units], dtype=dtype)
        if FLAGS.use_conv_feat_att:
            self.conv_filt = tf.get_variable(
                    'conv_filter',
                    shape = [200, 1, num_units])
        if self._normalize:
            # Scalar used in weight normalization
            self.g = tf.get_variable(
                'attention_g', dtype=dtype,
                initializer=tf.sqrt((1. / self._num_units)))
            # Bias added prior to the nonlinearity
            self.b = tf.get_variable(
                'attention_b', [self._num_units], dtype=dtype,
                initializer=tf.zeros_initializer())

    def __call__(self, query, previous_alignments):
        '''Score the query based on the keys and values.
        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          previous_alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        '''
        with tf.variable_scope(None, 'bahdanau_attention', [query]):
            processed_query = self.query_layer(
                query) if self.query_layer else query
            dtype = processed_query.dtype
            # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
            processed_query = tf.expand_dims(processed_query, 1)
            if FLAGS.use_conv_feat_att:
                conv_feat = tf.nn.conv1d(
                        tf.expand_dims(previous_alignments, 2),
                        self.conv_filt, 1, 'SAME')
            keys = self._keys
            if self._normalize:
                # normed_v = g * v / ||v||
                normed_v = self.g * self.v * tf.rsqrt(
                    tf.reduce_sum(tf.square(self.v)))
                score = tf.reduce_sum(
                    normed_v * tf.tanh(keys + processed_query + self.b), [2])
            else:
                if FLAGS.use_conv_feat_att:
                    score = tf.reduce_sum(self.v * tf.tanh(keys + processed_query + conv_feat),
                                      [2])
                else:
                    score = tf.reduce_sum(self.v * tf.tanh(keys + processed_query),
                                      [2])


        alignments = self._probability_fn(score, previous_alignments)
        return alignments

    def initial_alignments(self, batch_size, dtype):
        '''Returns all the alignment saturated in first block'''
        max_time = self._alignments_size
        alignments = _zero_state_tensors(max_time - 1, batch_size, dtype)
        return tf.concat([tf.fill([batch_size, 1], 1.0), alignments], 1)
