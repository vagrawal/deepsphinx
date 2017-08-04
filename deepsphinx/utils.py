"""Common utilities"""
import tensorflow as tf
import numpy as np

# https://github.com/zszyellow/WER-in-python/blob/master/wer.py
def wer(first, second):
    """This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    edit_dp = np.zeros(
        (len(first) + 1) * (len(second) + 1),
        dtype=np.uint8).reshape((len(first) + 1, len(second) + 1))
    for i in range(len(first) + 1):
        for j in range(len(second) + 1):
            if i == 0:
                edit_dp[0][j] = j
            elif j == 0:
                edit_dp[i][0] = i
    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            if first[i - 1] == second[j - 1]:
                edit_dp[i][j] = edit_dp[i - 1][j - 1]
            else:
                substitute = edit_dp[i - 1][j - 1] + 1
                insert = edit_dp[i][j - 1] + 1
                delete = edit_dp[i - 1][j] + 1
                edit_dp[i][j] = min(substitute, insert, delete)
    return (float(edit_dp[len(first)][len(second)]) /
            max(len(first), len(second)) * 100)

class FileOpen(tf.gfile.Open):
    """A custom class inheriting tf.gfile.Open for providing seek with whence

    Useful when using GCS
    """
    # pylint: disable=arguments-differ

    def seek(self, position, whence=0):
        if whence == 0:
            tf.gfile.Open.seek(self, position)
        elif whence == 1:
            tf.gfile.Open.seek(self, self.tell() + position)
        else:
            raise IOError

FLAGS = tf.app.flags.FLAGS
