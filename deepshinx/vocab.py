import numpy as np

vocab = np.asarray(
        ['<eps>', '<s>', '</s>'] + list(" '.-ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['<backoff>'])
vocab_to_int = {}

for ch in vocab:
    vocab_to_int[ch] = len(vocab_to_int)

# backoff is not in vocabulary
vocab_size = len(vocab) - 1

# https://github.com/zszyellow/WER-in-python/blob/master/wer.py
def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    #build the matrix
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: d[0][j] = j
            elif j == 0: d[i][0] = i
    for i in range(1,len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return float(d[len(r)][len(h)]) / max(len(r), len(h)) * 100

