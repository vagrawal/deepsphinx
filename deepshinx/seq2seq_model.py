import tensorflow as tf
from vocab import vocab, vocab_to_int, vocab_size
from lm import LMCellWrapper
from attention import BahdanauAttentionCutoff

def encoding_layer(
        rnn_size,
        input_lengths,
        num_layers,
        rnn_inputs,
        keep_prob):
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(
                    rnn_size,
                    initializer=tf.random_uniform_initializer(
                        -0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell_fw,
                    output_keep_prob = keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32,
                    input_size=rnn_inputs.get_shape()[2])

            cell_bw = tf.contrib.rnn.LSTMCell(
                    rnn_size,
                    initializer=tf.random_uniform_initializer(
                        -0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell_bw,
                    output_keep_prob = keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32,
                    input_size=rnn_inputs.get_shape()[2])

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    rnn_inputs,
                    input_lengths,
                    dtype=tf.float32)

            if layer != num_layers - 1:
                rnn_inputs = tf.concat(enc_output,2)
                # Keep only every second element in the sequence
                rnn_inputs = rnn_inputs[:, ::2, :]
                input_lengths = (input_lengths + 1) / 2
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output,2)

    return enc_output, enc_state, input_lengths

def training_decoding_layer(
        output_data,
        output_lengths,
        rnn_size,
        enc_output,
        input_lengths,
        dec_cell,
        batch_size,
        start_token,
        LMfst):
    attn_mech = BahdanauAttentionCutoff(
            rnn_size,
            enc_output,
            input_lengths,
            normalize=True,
            name='BahdanauAttention')

    output_data = tf.concat(
            [tf.fill([batch_size, 1], start_token), output_data[:, :-1]], 1)

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(
            dec_cell,
            attn_mech,
            vocab_size,
            output_attention=True)

    # dec_cell = LMCellWrapper(dec_cell, LMfst, 5)

    initial_state = dec_cell.zero_state(
            dtype=tf.float32,
            batch_size=batch_size)

    output_data = tf.nn.embedding_lookup(
            tf.eye(vocab_size),
            output_data)

    training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=output_data,
            sequence_length=output_lengths,
            time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
            dec_cell,
            training_helper,
            initial_state)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished = True)

    return training_logits

def inference_decoding_layer(
        start_token,
        end_token,
        max_output_length,
        batch_size,
        beam_width,
        rnn_size,
        enc_output,
        input_lengths,
        dec_cell,
        LMfst):
    enc_output = tf.contrib.seq2seq.tile_batch(
            enc_output,
            beam_width)

    input_lengths = tf.contrib.seq2seq.tile_batch(
            input_lengths,
            beam_width)

    attn_mech = BahdanauAttentionCutoff(
            rnn_size,
            enc_output,
            input_lengths,
            normalize=True,
            name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(
            dec_cell,
            attn_mech,
            vocab_size,
            output_attention=True)

    # dec_cell = LMCellWrapper(dec_cell, LMfst, 5)

    initial_state = dec_cell.zero_state(
            dtype=tf.float32,
            batch_size=batch_size * beam_width)

    start_tokens = tf.fill(
            [batch_size],
            start_token,
            name='start_tokens')

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            dec_cell,
            tf.eye(vocab_size),
            start_tokens,
            end_token,
            initial_state,
            beam_width)

    predictions, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,
            output_time_major=False,
            maximum_iterations=max_output_length)

    return predictions

def seq2seq_model(
        input_data,
        output_data,
        keep_prob,
        input_lengths,
        output_lengths,
        max_output_length,
        rnn_size,
        num_layers,
        batch_size,
        beam_width,
        learning_rate,
        LMfst):

    enc_output, enc_state, enc_lengths = encoding_layer(
            rnn_size,
            input_lengths,
            num_layers,
            input_data,
            keep_prob)

    with tf.variable_scope('decoder'):
        # For some reason sharing LSTMCell between first and middle layers is
        # not working
        lstm = tf.contrib.rnn.LSTMCell(
                rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        dec_cell_inp = tf.contrib.rnn.DropoutWrapper(
                lstm,
                output_keep_prob = keep_prob,
                variational_recurrent=True,
                dtype=tf.float32)
        lstm = tf.contrib.rnn.LSTMCell(
                rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        dec_cell = tf.contrib.rnn.DropoutWrapper(
                lstm,
                output_keep_prob = keep_prob,
                variational_recurrent=True,
                dtype=tf.float32)

        dec_cell_out = tf.contrib.rnn.LSTMCell(
                rnn_size,
                num_proj=vocab_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

        dec_cell = tf.contrib.rnn.MultiRNNCell(
                [dec_cell_inp] + [dec_cell] * (num_layers - 2) + [dec_cell_out])

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(
                output_data,
                output_lengths,
                rnn_size,
                enc_output,
                enc_lengths,
                dec_cell,
                batch_size,
                vocab_to_int['<s>'],
                LMfst)
    with tf.variable_scope("decode", reuse=True):
        predictions = inference_decoding_layer(
                vocab_to_int['<s>'],
                vocab_to_int['</s>'],
                max_output_length,
                batch_size,
                beam_width,
                rnn_size,
                enc_output,
                enc_lengths,
                dec_cell,
                LMfst)

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
            output_lengths,
            tf.reduce_max(output_lengths),
            dtype=tf.float32,
            name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            output_data,
            masks)

        tf.summary.scalar('cost', cost)

        step = tf.contrib.framework.get_or_create_global_step()

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(
            tf.clip_by_value(grad, -5., 5.), var)
            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, step)

    return training_logits, predictions, train_op, cost, step, scores

