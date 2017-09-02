'''Common utilities'''
import tensorflow as tf
import numpy as np

# https://github.com/zszyellow/WER-in-python/blob/master/wer.py
def wer(ref, hyp):
    '''This is a function that calculate the word error rate in ASR.
    You can use it like this: wer('what is it'.split(), 'what is'.split())
    '''
    # build the matrix
    edit_dp = np.zeros(
        (len(ref) + 1) * (len(hyp) + 1),
        dtype=np.uint8).reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                edit_dp[0][j] = j
            elif j == 0:
                edit_dp[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                edit_dp[i][j] = edit_dp[i - 1][j - 1]
            else:
                substitute = edit_dp[i - 1][j - 1] + 1
                insert = edit_dp[i][j - 1] + 1
                delete = edit_dp[i - 1][j] + 1
                edit_dp[i][j] = min(substitute, insert, delete)
    return float(edit_dp[len(ref)][len(hyp)]) / len(ref) * 100

FLAGS = tf.app.flags.FLAGS
