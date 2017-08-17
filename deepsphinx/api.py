""" A minimal API for the prediction for an audio file """
from deepsphinx.seq2seq_model import seq2seq_model
from deepsphinx.utils import FLAGS
from deepsphinx.vocab import VOCAB, VOCAB_SIZE
from deepsphinx.data import get_features
import tensorflow as tf

class Predict(object):
    """ Set flags and restore from checkpoint """
    def __init__(self, flags, checkpoint_path, lm_fst=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            flags['batch_size'] = 1
            # TODO: Use higher level API
            FLAGS.__dict__['__flags'] = flags
            self.input_length = tf.placeholder(tf.int32, shape=[1])
            self.input = tf.placeholder(tf.float32, shape=[1, None, FLAGS.nfilt * 3 + 1])
            _, self.predictions, _, _, _, _ = seq2seq_model(
                self.input,
                tf.placeholder(tf.int32, shape=[1, VOCAB_SIZE]),
                self.input_length,
                tf.placeholder(tf.int32, shape=[1]),
                lm_fst,
                1.0)
            self.sess = tf.Session(graph=self.graph)
            tf.train.Saver().restore(self.sess, checkpoint_path)

    @staticmethod
    def default_flags():
        return {'nfilt': 40,
                'max_output_len': 250,
                'rnn_size': 256,
                'num_layers': 3,
                'num_decoding_layers': 3,
                'batch_size': 1,
                'beam_width': 16,
                'cutoff_range': 200,
                'use_train_lm': False,
                'use_inference_lm': False,
                'learning_rate': 0.0
                }

    def predict(self, audio_file):
        """ Predict and return string output by beam search """
        feat = get_features(audio_file)
        pred = self.sess.run(self.predictions, feed_dict={
            self.input: [feat], self.input_length: [feat.shape[0]]})
        return ''.join([VOCAB[l] for l in pred[0, :, 0]])
