''' Language modelling for the tensorflow model'''
import tensorflow as tf
from deepsphinx.fst import fst_costs

class LMCellWrapper(tf.contrib.rnn.RNNCell):
    '''This class wraps a decoding cell to add LM scores'''

    def __init__(self, dec_cell, fst, max_states, reuse=None):
        super(LMCellWrapper, self).__init__(_reuse=reuse)
        self.dec_cell = dec_cell
        self.fst = fst
        self._output_size = dec_cell.output_size
        self.max_states = max_states
        # LSTM state, FST states and number of FST states
        self._state_size = (dec_cell.state_size,
                            tf.TensorShape((max_states)),
                            tf.TensorShape((max_states)),
                            tf.TensorShape((1)))

    @property
    def state_size(self):
        '''State size of the cell'''
        return self._state_size

    @property
    def output_size(self):
        '''Output size of the cell'''
        return self._output_size

    def __call__(self, inputs, state):
        '''Step once given the input and return score and next state'''
        cell_state, fst_states, state_probs, num_fst_states = state
        cell_out, cell_state = self.dec_cell(inputs, cell_state)

        def fst_costs_env(states, probs, num, inp):
            '''Python function'''
            return fst_costs(states, probs, num, inp, self.fst, self.max_states)

        func_appl = tf.py_func(fst_costs_env,
                               [fst_states, state_probs, num_fst_states,
                                tf.argmax(inputs, 1)],
                               [tf.int32, tf.float32, tf.int32, tf.float32],
                               stateful=False)
        next_state, next_state_probs, next_num_states, lm_scores = func_appl
        next_state.set_shape(fst_states.shape)
        next_num_states.set_shape(num_fst_states.shape)
        next_state_probs.set_shape(state_probs.shape)
        lm_scores.set_shape(cell_out.shape)
        fin_score = tf.nn.log_softmax(
            cell_out) + 0.5 * tf.nn.log_softmax(lm_scores)
        return fin_score, (cell_state, next_state, next_state_probs, next_num_states)

    def zero_state(self, batch_size, dtype):
        '''Zero state'''
        return (self.dec_cell.zero_state(batch_size, dtype),
                tf.zeros((batch_size, self.max_states), tf.int32),
                tf.zeros((batch_size, self.max_states), tf.float32),
                tf.ones((batch_size, 1), tf.int32))
