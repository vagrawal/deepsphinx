'''Tensorflow model for speech recognition'''
import tensorflow as tf
from deepsphinx.vocab import VOCAB_SIZE, VOCAB_TO_INT
from deepsphinx.utils import FLAGS
from deepsphinx.lm import LMCellWrapper
from deepsphinx.attention import BahdanauAttentionCutoff

def encoding_layer(
        input_lengths,
        rnn_inputs,
        keep_prob):
    ''' Encoding layer for the model.

    Args:
        input_lengths (Tensor): A tensor of input lenghts of instances in
            batches
        rnn_inputs (Tensor): Inputs

    Returns:
        Encoding output, LSTM state, output length
    '''
    for layer in range(FLAGS.num_conv_layers):
        filter = tf.get_variable(
            "conv_filter{}".format(layer + 1),
            shape=[FLAGS.conv_layer_width, rnn_inputs.get_shape()[2], FLAGS.conv_layer_size])
        rnn_inputs = tf.nn.conv1d(rnn_inputs, filter, 1, 'SAME')
    for layer in range(FLAGS.num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(
                FLAGS.rnn_size,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw,
                output_keep_prob=keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=rnn_inputs.get_shape()[2])

            cell_bw = tf.contrib.rnn.LSTMCell(
                FLAGS.rnn_size,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw,
                output_keep_prob=keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=rnn_inputs.get_shape()[2])

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                rnn_inputs,
                input_lengths,
                dtype=tf.float32)

            if layer != FLAGS.num_layers - 1:
                rnn_inputs = tf.concat(enc_output,2)
                rnn_inputs = rnn_inputs[:, ::2, :]
                input_lengths = (input_lengths + 1) // 2
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state, input_lengths

def get_dec_cell(
        enc_output,
        enc_output_lengths,
        use_lm,
        fst,
        tile_size,
        keep_prob):
    '''Decoding cell for attention based model

    Return:
        `RNNCell` Instance
    '''

    lstm = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    dec_cell_inp = tf.contrib.rnn.DropoutWrapper(
        lstm,
        output_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)
    lstm = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    dec_cell = tf.contrib.rnn.DropoutWrapper(
        lstm,
        output_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)

    dec_cell_out = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        num_proj=VOCAB_SIZE,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

    dec_cell = tf.contrib.rnn.MultiRNNCell(
        [dec_cell_inp] +
        [dec_cell] * (FLAGS.num_decoding_layers - 2) +
        [dec_cell_out])

    enc_output = tf.contrib.seq2seq.tile_batch(
        enc_output,
        tile_size)

    enc_output_lengths = tf.contrib.seq2seq.tile_batch(
        enc_output_lengths,
        tile_size)

    attn_mech = BahdanauAttentionCutoff(
        FLAGS.rnn_size,
        enc_output,
        enc_output_lengths,
        normalize=True,
        name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(
        dec_cell,
        attn_mech,
        VOCAB_SIZE,
        output_attention=True)

    if use_lm:
        dec_cell = LMCellWrapper(dec_cell, fst, 5)

    return dec_cell


#pylint: disable-msg=too-many-arguments
def training_decoding_layer(
        target_data,
        target_lengths,
        enc_output,
        enc_output_lengths,
        fst,
        keep_prob):
    ''' Training decoding layer for the model.

    Returns:
        Training logits
    '''
    target_data = tf.concat(
        [tf.fill([FLAGS.batch_size, 1], VOCAB_TO_INT['<s>']),
         target_data[:, :-1]], 1)

    dec_cell = get_dec_cell(
        enc_output,
        enc_output_lengths,
        FLAGS.use_train_lm,
        fst,
        1,
        keep_prob)

    initial_state = dec_cell.zero_state(
        dtype=tf.float32,
        batch_size=FLAGS.batch_size)

    target_data = tf.nn.embedding_lookup(
        tf.eye(VOCAB_SIZE),
        target_data)

    training_helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=target_data,
        sequence_length=target_lengths,
        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell,
        training_helper,
        initial_state)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(
        training_decoder,
        output_time_major=False,
        impute_finished=True)

    return training_logits


def inference_decoding_layer(
        enc_output,
        enc_output_lengths,
        fst,
        keep_prob):
    ''' Inference decoding layer for the model.

    Returns:
        Predictions
    '''

    dec_cell = get_dec_cell(
        enc_output,
        enc_output_lengths,
        FLAGS.use_inference_lm,
        fst,
        FLAGS.beam_width,
        keep_prob)

    initial_state = dec_cell.zero_state(
        dtype=tf.float32,
        batch_size=FLAGS.batch_size * FLAGS.beam_width)

    start_tokens = tf.fill(
        [FLAGS.batch_size],
        VOCAB_TO_INT['<s>'],
        name='start_tokens')

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        dec_cell,
        tf.eye(VOCAB_SIZE),
        start_tokens,
        VOCAB_TO_INT['</s>'],
        initial_state,
        FLAGS.beam_width)

    predictions, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder,
        output_time_major=False,
        maximum_iterations=FLAGS.max_output_len)

    return predictions

def seq2seq_model(
        input_data,
        target_data,
        input_lengths,
        target_lengths,
        fst,
        keep_prob):
    ''' Attention based model

    Returns:
        Logits, Predictions, Training operation, Cost, Step, Scores of beam
        search
    '''

    enc_output, _, enc_lengths = encoding_layer(
        input_lengths,
        input_data,
        keep_prob)

    with tf.variable_scope('decode'):

        training_logits = training_decoding_layer(
            target_data,
            target_lengths,
            enc_output,
            enc_lengths,
            fst,
            keep_prob)
    with tf.variable_scope('decode', reuse=True):
        predictions = inference_decoding_layer(
            enc_output,
            enc_lengths,
            fst,
            keep_prob)

    # Create tensors for the training logits and predictions
    training_logits = tf.identity(
        training_logits.rnn_output,
        name='logits')
    scores = tf.identity(
        predictions.beam_search_decoder_output.scores,
        name='scores')
    predictions = tf.identity(
        predictions.predicted_ids,
        name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(
        target_lengths,
        tf.reduce_max(target_lengths),
        dtype=tf.float32,
        name='masks')

    with tf.name_scope('optimization'):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            target_data,
            masks)

        tf.summary.scalar('cost', cost)

        step = tf.contrib.framework.get_or_create_global_step()

        # Optimizer
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [
            (tf.clip_by_value(grad, -5., 5.), var)
            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, step)

    return training_logits, predictions, train_op, cost, step, scores
