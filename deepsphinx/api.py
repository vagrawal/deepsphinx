''' A minimal API for the prediction for an audio file '''
from deepsphinx.seq2seq_model import seq2seq_model
from deepsphinx.utils import FLAGS
from deepsphinx.vocab import VOCAB, VOCAB_SIZE
from deepsphinx.data import get_features
import tensorflow as tf
from deepsphinx.flags import load_flags
import numpy as np

class Predict(object):
    ''' Set flags and restore from checkpoint '''
    def __init__(self, checkpoint_path, lm_fst=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            FLAGS.batch_size = 1
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

    def predict(self, audio_file):
        ''' Predict and return string output by beam search '''
        feat = get_features(audio_file)
        # Normalize by file
        mean = np.mean(feat, 0)
        var = np.var(feat, 0).clip(1e-5)
        feat = (feat - mean) / np.sqrt(var)
        pred = self.sess.run(self.predictions, feed_dict={
            self.input: [feat], self.input_length: [feat.shape[0]]})
        return ''.join([VOCAB[l] for l in pred[0, :, 0]]).split('<')[0]
